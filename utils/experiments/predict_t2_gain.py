# -*- coding: utf-8 -*-
"""预测 T2 收益：把 `curated_issues.solution` 也重排后，4 个未放行样本能救回几个？

## 问题

三臂实验（新系统臂 = Arm1，30 样本）里：检索段 30/30 都捞到自己的 curated 条目，
门控段只有 26/30 放行自己。T2 = 把 `curated_issues.solution` 也换成重建后的版本。
**这个改动能不能把那 4 个样本救回来？** —— 决定真机上 T2 值不值得跑。

## 做法

对每个样本的"自己那条 curated 候选"，在其**真实运行落盘的证据**
（`reports/analysis/<CVE>/<run>/second_pass/consolidated/*_r2.json`）上做**对照重放**：
同一份候选、同一个 haystack，只把 `solution` 从旧库换成新库，跑**生产门控函数**看决策。

### 生产调用顺序（重要，踩过坑）

`curated_issue` 通道在 `_handle_issue` 里的顺序是：

    candidate = _build_candidate_from_curated_issue(curated_hit, ...)
    _gate_candidate(candidate)                      # 直接门控，**中间没有** clone 那一步

克隆证据（权重最高的 0.5）是在 `_match_curated_issue()` **内部**算出来并写进
`curated_hit["matched_fields"] / structured_score` 的，`_build_candidate_from_curated_issue`
再把它搬进候选。所以：

* **不能**直接拿产物里的候选去 `_gate_candidate` —— 产物已是"旧 solution 的克隆证据
  *加上之后*"的状态，直接门控只会把旧结论原样复述一遍，看不到新 solution 的效果；
* 正确做法是**先反演回"克隆证据之前"的基线状态**（去掉 `error_code_clone`、`structured_score`
  减 0.5），再用 `_apply_error_code_clone_evidence()` 按新 solution 重算这一步 ——
  它与 `_match_curated_issue` 的克隆分支是**同一把尺子**（同一个 haystack = 被分析文件全文、
  同一套 token 化、同一个 `_is_contiguous_subseq`），因此数值上等价。
  `_apply_error_code_clone_evidence` 是 sqlite/weaviate 通道的生产调用，这里借它复现
  `_match_curated_issue` 的克隆分支（脚本里会断言两条路径结论一致）。

反演是**可逆**的：不带克隆命中时 `_match_curated_issue` 的加权只有
basename 0.15 + phenomenon 0.1 + root_cause 0.08 = **0.33 `< 0.4`**，那个
"未命中克隆则封顶 0.4"的 `min()` 永远不会咬合，所以 `structured_score` 可以精确还原。

### haystack

* `_candidate_code_fixed()` 用候选自带的 `_current_code`（产物里就是**服务器上被分析文件的
  全文**，实测与本地数据集同名文件字节数一致，本地 7836 == 产物 7836）；
* `_apply_error_code_clone_evidence()` 走 `_current_code_tokens(issue_file)`，要读盘。
  服务器路径 `/root/autodl-tmp/MAS/...` 本地不存在，直接传会静默退化成 `code_snippet`
  分片（**这正是"离线假阴性"的来源**）。所以这里按路径把 hash 缓存
  `agent._code_token_cache` 预置成"产物 `_current_code` 的 token"，
  保证两条路径用的 haystack 与生产**逐 token 相同**；同时核对本地文件 token 是否一致并报告。

## 用法

    python utils/experiments/predict_t2_gain.py
    python utils/experiments/predict_t2_gain.py --repeat 2          # 确定性自验
    python utils/experiments/predict_t2_gain.py --reverse-control    # 反向对照
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

# **复用**既有映射逻辑：curated 通道的 sqlite_id 是 curated_issues 主键，
# 必须先经 curated_issues.pattern_id 映射到 issue_patterns.id 才能和"自己的条目"比。
# 第一版就是直接比 sqlite_id，把 26 条自己的条目全算成别人（1/30 vs 26/30）。
# `owner_of` 就是 `ab_eval_runsets` 里那个、也是 `compare_arms` 复用的那一个。
from utils.experiments.ab_eval_runsets import owner_of  # noqa: E402

SERVER_PREFIX = "/root/autodl-tmp/MAS/"
DEFAULT_DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
ADMIT = {"formal_hit", "explanatory_hit"}


def read_only(db: Path):
    con = sqlite3.connect("file:%s?mode=ro" % Path(db).as_posix(), uri=True)
    try:
        yield con
    finally:
        con.close()


def rows(db: Path, sql: str, args=()):
    con = sqlite3.connect("file:%s?mode=ro" % Path(db).as_posix(), uri=True)
    try:
        return list(con.execute(sql, args))
    finally:
        con.close()


def to_local(server_path: str, ds: Path) -> str:
    """服务器路径 → 本地数据集路径。"""
    s = str(server_path or "").strip().replace("\\", "/")
    if not s:
        return ""
    if s.startswith(SERVER_PREFIX):
        return str((ROOT / s[len(SERVER_PREFIX):]).resolve())
    # 兜底：按 `before/<CVE>/...` 后缀找
    i = s.find("before/")
    if i >= 0:
        return str((ds / ".." / s[i:]).resolve())
    return s if Path(s).exists() else ""


def collect_instances(runs_file: Path, old_db: Path, id_by_title: dict, ci_to_pattern: dict) -> dict:
    """逐样本收集"自己那条 curated 候选"的全部真实证据实例（去重）。"""
    samples = [ln.strip() for ln in runs_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = {}
    for line in samples:
        cve, run = line.split("/", 1)
        own = id_by_title.get(cve.upper())
        d = ROOT / "reports/analysis" / cve / run / "second_pass" / "consolidated"
        rec = {"cve": cve, "run": run, "own_pattern_id": own, "ci_ids": sorted(
            ci for ci, p in ci_to_pattern.items() if own is not None and p == own),
            "artifact_own_in_new_findings": False, "instances": [], "r2_exists": d.exists(),
            "artifact_ci_admitted": set(), "artifact_admit_channels": {},
            "other_channel_candidates": []}
        if not d.exists():
            out[cve] = rec
            continue
        seen = set()
        for f in sorted(d.glob("*_r2.json")):
            j = json.loads(f.read_text(encoding="utf-8"))
            for nf in (j.get("new_findings") or []):
                ev = nf.get("evidence") or {}
                chan = str(ev.get("channel") or ev.get("primary_channel") or "")
                o = owner_of({"channel": chan, "sqlite_id": ev.get("sqlite_id")},
                             own, ci_to_pattern)
                if own is not None and o is not None and int(o) == int(own):
                    rec["artifact_own_in_new_findings"] = True
                    rec["artifact_admit_channels"][chan] = \
                        rec["artifact_admit_channels"].get(chan, 0) + 1
            for key in ("retrieval_evidence", "gap_retrieval_evidence"):
                for ev in (j.get(key) or []):
                    for c in (ev.get("candidates") or []):
                        if not isinstance(c, dict):
                            continue
                        chan = str(c.get("channel") or c.get("primary_channel") or "")
                        sid = c.get("sqlite_id")
                        o = owner_of({"channel": chan, "sqlite_id": sid}, own, ci_to_pattern)
                        if own is None or o is None or int(o) != int(own):
                            continue
                        # **只有 curated 通道**的候选才由 `curated_issues.solution` 驱动
                        # （sqlite 通道读的是 `issue_patterns.solution`，与 T2 无关）。
                        # 第一版漏了这个 channel 判据，把同 pattern_id 的 sqlite 候选也卷进来，
                        # 于是"旧 solution"那一臂实际喂错了 payload → 22/169 实例对不上产物。
                        if chan != "curated_issue":
                            rec["other_channel_candidates"].append({
                                "channel": chan, "sqlite_id": sid, "src_file": f.name,
                                "decision": c.get("gating_decision"),
                                "reason": c.get("rejection_reason") or "",
                                "matched_fields": list(c.get("matched_fields") or []),
                            })
                            continue
                        mf = tuple(c.get("matched_fields") or [])
                        ident = (int(sid), str(c.get("_analysis_file") or ev.get("issue_file")),
                                 str(c.get("file_pattern") or ""), mf,
                                 c.get("structured_score"), c.get("semantic_score"),
                                 c.get("anchor_score"), c.get("context_score"))
                        if ident in seen:
                            continue
                        seen.add(ident)
                        rec["instances"].append({
                            "src_file": f.name, "evidence_key": key,
                            "sqlite_id": int(sid),
                            "issue_file_server": str(c.get("_analysis_file") or ev.get("issue_file") or ""),
                            "candidate": c,
                            "artifact_decision": c.get("gating_decision"),
                            "artifact_reason": c.get("rejection_reason") or "",
                            "artifact_sx": None,   # 由调用方补
                            "artifact_file_pattern": c.get("file_pattern") or "",
                        })
        for inst in rec["instances"]:
            if inst["artifact_decision"] in ADMIT:
                rec["artifact_ci_admitted"].add(inst["sqlite_id"])
        rec["artifact_ci_admitted"] = sorted(rec["artifact_ci_admitted"])
        out[cve] = rec
    return out


# --------------------------------------------------------------------------- #
# 单实例重放
# --------------------------------------------------------------------------- #

def base_candidate(inst: dict, agent) -> tuple:
    """反演回"克隆证据之前"的基线候选 + 反演说明。

    `_match_curated_issue` 里克隆命中会 +0.5 并把 error_code_clone 写进 matched_fields；
    这里精确减回去，使候选回到"只有 basename/描述词面"的状态，供两条臂各自重算。
    """
    cand = copy.deepcopy(inst["candidate"])
    notes = []
    mf = list(cand.get("matched_fields") or [])
    if "error_code_clone" in mf:
        mf.remove("error_code_clone")
        cand["structured_score"] = max(0.0, float(cand.get("structured_score") or 0.0) - 0.5)
        notes.append("反演：旧 solution 曾命中克隆，已去掉 error_code_clone 并 -0.5")
    elif float(cand.get("structured_score") or 0.0) > 0.4 + 1e-9:
        notes.append("⚠ 异常：无 error_code_clone 但 structured>0.4（封顶规则本应压到 0.4）")
    # 克隆未命中时 `_match_curated_issue` 会封顶 0.4；实测弱证据和只有 0.33，永不咬合
    cand["matched_fields"] = mf
    cand["gating_decision"] = ""
    cand["rejection_reason"] = ""
    # 注：`_match_curated_issue` 在克隆命中时还会把 context_score 从 0 抬到 0.15/0.05。
    # 这里**不**模拟它：context 只进 curated 的 total_score（权重 0.15），
    # 而门控判定用的是 s(x)/semantic/anchor，与 context 无关 —— 不影响任何判定。
    return cand, notes


def replay(agent, inst: dict, solution: str, local_path: str, current_code: str) -> dict:
    """按生产语义重放一条候选：克隆证据 → 门控。返回可比较的结果字典。

    · 克隆证据：`_apply_error_code_clone_evidence`（与 `_match_curated_issue` 的克隆分支同一把尺子）
    · 门控：`_gate_candidate`（生产函数，不重写判据）
    """
    cand, notes = base_candidate(inst, agent)
    cand["solution"] = solution or ""
    issue = {"file": local_path, "code_snippet": current_code[:2000]}
    # 1) 克隆证据（生产顺序：必须在门控之前）
    agent._apply_error_code_clone_evidence(cand, local_path, issue)
    # 2) 门控（生产函数）
    agent._gate_candidate(cand)
    mf = list(cand.get("matched_fields") or [])
    sx = cand.get("unified_structured_score")
    if sx is None:
        sx = agent._unified_structured_score(mf)
    frags = agent._extract_error_code_fragments(solution or "")
    toks = agent._tokenize_code(current_code)
    return {
        "decision": cand.get("gating_decision") or "",
        "reason": cand.get("rejection_reason") or "",
        "sx": round(float(sx), 4),
        "admitted": (cand.get("gating_decision") in ADMIT),
        "matched_fields": mf,
        "structured_score": round(float(cand.get("structured_score") or 0.0), 4),
        "total_score": cand.get("total_score"),
        "semantic_score": round(float(cand.get("semantic_score") or 0.0), 4),
        "anchor_score": round(float(cand.get("anchor_score") or 0.0), 4),
        "code_fixed_scope": cand.get("code_fixed_scope") or "",
        "clone_hit": "error_code_clone" in mf,
        "n_frags": len(frags),
        "frag_lens": [len(f) for f in frags],
        "frag_hits": [agent._is_contiguous_subseq(f, toks) for f in frags],
        "notes": notes,
    }


def classify_blocker(old: dict, new: dict, new_solution: str, old_solution: str) -> str:
    if (new_solution or "") == (old_solution or ""):
        return "solution 在新库里**未变** —— T2 对这条不产生任何作用"
    if new["admitted"]:
        return "-"
    if new["n_frags"] == 0:
        return "针不可提取（token 数 < error_code_clone_min_tokens=4，或整段都是通用 token）"
    if not any(new["frag_hits"]):
        return "针仍不命中当前文件 → `_candidate_code_fixed` 判真"
    if new["code_fixed_scope"] == "different_file":
        return "跨文件（`_same_analysis_target` 为假）"
    if new["sx"] < 0.65 and new["semantic_score"] < 0.78:
        return "针命中了，但 s(x)=%.2f < θ_s=0.65 且语义分 %.2f < τ=0.78，分数仍不够" % (
            new["sx"], new["semantic_score"])
    return "其它（未分类）"


# --------------------------------------------------------------------------- #

def build_agent():
    from core.agents.ai_driven_second_pass_analysis_agent import (
        AIDrivenSecondPassAnalysisAgent,
    )
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    return agent


def prepare(args, agent):
    """装载两库、收集实例、核对 haystack、预置 token 缓存。"""
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   rows(args.old_db, "select id,title from issue_patterns")}
    ci_to_pattern = {int(i): int(p) for i, p in
                     rows(args.old_db, "select id,pattern_id from curated_issues")}
    old_sol = {int(i): (s or "") for i, s in rows(args.old_db, "select id,solution from curated_issues")}
    new_sol = {int(i): (s or "") for i, s in rows(args.new_db, "select id,solution from curated_issues")}

    data = collect_instances(args.runs, args.old_db, id_by_title, ci_to_pattern)

    haystack_report = []
    for cve, rec in data.items():
        for inst in rec["instances"]:
            server = inst["issue_file_server"]
            local = to_local(server, args.dataset_root)
            inst["local_file"] = local
            exists = bool(local) and Path(local).is_file()
            cc = str(inst["candidate"].get("_current_code") or "")
            frags_src = "产物 _current_code"
            if not cc and exists:
                # 证据瘦身后（`trim_run_evidence.py`）候选里的 `_current_code` 已被移出、
                # 原文留在同名 `.gz` 里。这里退回到**数据集里的那个被分析文件**——
                # 本来就是同一个文件（下面会说明这一步之后 token 一致性不再能独立核对）。
                try:
                    cc = Path(local).read_text(encoding="utf-8", errors="ignore")
                    frags_src = "数据集文件（产物 _current_code 已瘦身移出）"
                except Exception:
                    cc = ""
            inst["current_code"] = cc
            inst["haystack_source"] = frags_src
            same_tokens = None
            if exists and frags_src == "产物 _current_code":
                try:
                    local_txt = Path(local).read_text(encoding="utf-8", errors="ignore")
                    same_tokens = (agent._tokenize_code(local_txt) == agent._tokenize_code(cc))
                except Exception:
                    same_tokens = False
            haystack_report.append({"cve": cve, "server": server, "local": local,
                                    "local_exists": exists, "tokens_match_artifact": same_tokens,
                                    "artifact_len": len(cc), "source": frags_src})
            # **关键**：把 token 缓存预置成产物里那份 haystack（= 生产在同一 run 里读到的内容），
            # 使 `_apply_error_code_clone_evidence` 与 `_candidate_code_fixed` 逐 token 同源。
            key = ""
            if local:
                p = local if os.path.isabs(local) else os.path.join(os.getcwd(), local)
                try:
                    key = os.path.normcase(os.path.abspath(p))
                except Exception:
                    key = ""
            if key:
                agent._code_token_cache[key] = agent._tokenize_code(cc)
    return dict(id_by_title=id_by_title, ci_to_pattern=ci_to_pattern, old_sol=old_sol,
                new_sol=new_sol, data=data, haystack_report=haystack_report)


def predict(args, agent, loaded, new_db_override: Path):
    """跑一次完整预测。返回 (per-sample 行, per-instance 明细)。"""
    data = loaded["data"]
    old_sol, new_sol = loaded["old_sol"], loaded["new_sol"]
    rows_out, details = [], []
    if Path(new_db_override).resolve() != Path(args.new_db).resolve():
        new_sol = {int(i): (s or "") for i, s in
                   rows(new_db_override, "select id,solution from curated_issues")}

    for cve, rec in data.items():
        per_ci = {}
        for inst in rec["instances"]:
            sid = inst["sqlite_id"]
            o = replay(agent, inst, old_sol.get(sid, ""), inst["local_file"], inst["current_code"])
            n = replay(agent, inst, new_sol.get(sid, ""), inst["local_file"], inst["current_code"])
            d = {"cve": cve, "ci_id": sid, "file": inst["artifact_file_pattern"],
                 "local_file": inst["local_file"], "src_file": inst["src_file"],
                 "old": o, "new": n,
                 "artifact_decision": inst["artifact_decision"],
                 "artifact_reason": inst["artifact_reason"],
                 "artifact_matched_fields": list(inst["candidate"].get("matched_fields") or []),
                 "artifact_structured_score": round(float(inst["candidate"].get("structured_score") or 0.0), 4),
                 "artifact_total_score": inst["candidate"].get("total_score"),
                 "old_reproduces_artifact": (o["decision"] == inst["artifact_decision"]
                                             and o["reason"] == inst["artifact_reason"]),
                 "old_fields_match": (sorted(o["matched_fields"])
                                      == sorted(inst["candidate"].get("matched_fields") or [])),
                 "old_structured_match": abs(o["structured_score"] - float(
                     inst["candidate"].get("structured_score") or 0.0)) < 1e-6,
                 "old_total_match": (inst["candidate"].get("total_score") is None
                                     or abs(float(o["total_score"] or 0.0)
                                            - float(inst["candidate"].get("total_score") or 0.0)) < 1e-6),
                 "blocker": classify_blocker(o, n, new_sol.get(sid, ""), old_sol.get(sid, ""))}
            details.append(d)
            per_ci.setdefault(sid, []).append(d)

        ci_rows = []
        for sid in rec["ci_ids"]:
            ds = per_ci.get(sid, [])
            best_old = max(ds, key=lambda x: x["old"]["sx"]) if ds else None
            best_new = max(ds, key=lambda x: x["new"]["sx"]) if ds else None
            ci_rows.append({
                "ci_id": sid,
                "solution_changed": old_sol.get(sid, "") != new_sol.get(sid, ""),
                "n_instances": len(ds),
                "old_admitted": any(x["old"]["admitted"] for x in ds),
                "new_admitted": any(x["new"]["admitted"] for x in ds),
                "old_decisions": sorted({(x["old"]["decision"], x["old"]["reason"]) for x in ds}),
                "new_decisions": sorted({(x["new"]["decision"], x["new"]["reason"]) for x in ds}),
                "sx_old": best_old["old"]["sx"] if best_old else None,
                "sx_new": best_new["new"]["sx"] if best_new else None,
                "blocker": (next((x["blocker"] for x in ds if x["blocker"] != "-"), "-")
                            if not any(x["new"]["admitted"] for x in ds) else "-"),
            })
        own_ids = [r["ci_id"] for r in ci_rows]
        old_adm = any(r["old_admitted"] for r in ci_rows)
        new_adm = any(r["new_admitted"] for r in ci_rows)
        rows_out.append({
            "cve": cve, "ci_ids": own_ids, "has_own_candidate": bool(own_ids),
            "artifact_admitted": rec["artifact_own_in_new_findings"],
            "artifact_admit_channels": rec["artifact_admit_channels"],
            "other_channel_candidates": rec["other_channel_candidates"],
            "other_channel_admitted": any(
                x["decision"] in ADMIT for x in rec["other_channel_candidates"]),
            "old_admitted": old_adm, "new_admitted": new_adm,
            "rescued": (not old_adm) and new_adm,
            "lost": old_adm and (not new_adm),
            "ci_rows": ci_rows,
        })
    return rows_out, details


def own_entry_coverage(loaded) -> list:
    """检查：每个样本**自己那几条** curated 条目，是否都在 Arm1 的候选里出现过。

    这一步排除"视野盲区"：如果某条自己的条目在旧 run 里根本没成为候选，
    那么新 solution 让它变强之后会**新出现**，而重放表里看不到它。
    """
    out = []
    for cve, rec in loaded["data"].items():
        seen = {i["sqlite_id"] for i in rec["instances"]}
        missing = [c for c in rec["ci_ids"] if c not in seen]
        out.append({"cve": cve, "ci_ids": rec["ci_ids"], "in_candidates": sorted(seen),
                    "missing": missing})
    return out


def fp_census(agent, loaded, args, details) -> dict:
    """副作用普查（**上界**）：新 solution 会不会让"外来的" curated 条目也命中而变成新放行。

    curated 通道的 s(x) 只可能由两个字段构成（`_match_curated_issue` 只产这两个计分字段，
    且 curated 候选**不走** `_apply_file_function_anchors`）：
        s(x) = 0.5·[error_code_clone] + 0.2·[basename_match]
    所以外来条目要越过 θ_s=0.65，必须**同时**满足：新针命中该文件 **且** basename 相同
    （`_normalize_source_basename` 比的是裸小写文件名，同一 run 里同名不同目录会撞车）。
    这里把这个必要条件当普查条件（因此结果是**上界**，不是精确误报数）。
    """
    ci_rows_new = rows(args.new_db,
                       "select id, pattern_id, file_path, solution from curated_issues")
    old_sol = loaded["old_sol"]
    ci_to_pattern_new = {int(i): int(p) for i, p, _f, _s in ci_rows_new}

    # 样本 → {文件 → (tokens, 同文件外的候选判据)}
    by_sample = {}
    for d in details:
        by_sample.setdefault(d["cve"], {})
    for cve, rec in loaded["data"].items():
        for inst in rec["instances"]:
            f = inst["local_file"]
            if f in by_sample[cve]:
                continue
            toks = agent._tokenize_code(inst["current_code"])
            by_sample[cve][f] = {"tokens": toks, "text": " ".join(toks),
                                 "base": agent._normalize_source_basename(f)}

    # 旧 run 里出现过的 (sample, curated_id) → 决策
    seen_pairs = {}
    for d in details:
        seen_pairs[(d["cve"], d["ci_id"])] = d["artifact_decision"]

    hits = []
    for cve, files in by_sample.items():
        own = loaded["data"][cve]["own_pattern_id"]
        for f, fd in files.items():
            for cid, pat, fpath, sol in ci_rows_new:
                cid, pat = int(cid), int(pat)
                if pat == own:
                    continue                      # 自己的条目已在主表里逐条量过
                if agent._normalize_source_basename(str(fpath or "")) != fd["base"]:
                    continue                      # basename 不同 → s(x) 只到 0.5，过不了 0.65
                frags = agent._extract_error_code_fragments(sol or "")
                if not frags or not any((" ".join(fr) in fd["text"]) for fr in frags):
                    continue
                prev = seen_pairs.get((cve, cid))
                old_frags = agent._extract_error_code_fragments(old_sol.get(cid, "") or "")
                old_hit = bool(old_frags) and any(
                    (" ".join(fr) in fd["text"]) for fr in old_frags)
                hits.append({"cve": cve, "file": Path(f).name, "curated_id": cid,
                             "pattern_id": pat, "solution_changed": sol != old_sol.get(cid, ""),
                             "needle_hit_before": old_hit, "artifact_decision": prev or "(旧 run 里不是候选)"})
    new_hits = [h for h in hits if not h["needle_hit_before"] or h["artifact_decision"] == "(旧 run 里不是候选)"]
    newly_admitted = [h for h in new_hits if h["artifact_decision"] not in ADMIT]
    return {"hits": hits, "new_hits": new_hits, "newly_admitted": newly_admitted,
            "n_files": sum(len(v) for v in by_sample.values())}


def digest(rows_out, details) -> str:
    payload = json.dumps({"rows": rows_out,
                          "details": [{k: v for k, v in d.items() if k != "notes"}
                                      for d in details]},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "reports/arm1_runs.txt")
    ap.add_argument("--old-db", type=Path, default=ROOT / "reports/mas_live.db")
    ap.add_argument("--new-db", type=Path, default=ROOT / "reports/mas_rebuild_candidate_v2.db")
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DS)
    ap.add_argument("--json-out", type=Path, default=ROOT / "reports/t2_gain_prediction.json")
    ap.add_argument("--repeat", type=int, default=2, help="重跑次数，用于确定性自验")
    ap.add_argument("--reverse-control", action="store_true",
                    help="反向对照：新臂改用旧库，CVE-2018-20854 必须回落 code_already_fixed")
    args = ap.parse_args()

    agent = build_agent()
    loaded = prepare(args, agent)

    # ---------- haystack 一致性 ----------
    seen, hs_rows = set(), []
    for h in loaded["haystack_report"]:
        if h["server"] in seen:
            continue
        seen.add(h["server"])
        hs_rows.append(h)

    print("=" * 108)
    print("0) haystack 一致性（产物 `_current_code` = 生产在被分析文件里读到的全文）")
    print("=" * 108)
    n_missing = sum(1 for h in hs_rows if not h["local_exists"])
    n_mismatch = sum(1 for h in hs_rows if h["local_exists"] and h["tokens_match_artifact"] is False)
    print("  服务器被分析文件 %d 个；本地可定位 %d 个；本地文件与产物 token 不一致 %d 个"
          % (len(hs_rows), len(hs_rows) - n_missing, n_mismatch))
    if n_missing or n_mismatch:
        for h in hs_rows:
            if not h["local_exists"] or h["tokens_match_artifact"] is False:
                print("    ⚠ %s %s exists=%s tokens_match=%s"
                      % (h["cve"], h["local"], h["local_exists"], h["tokens_match_artifact"]))
    print("  注：重放一律用**产物里的 `_current_code`** 当 haystack（已预置进 token 缓存），"
          "本地文件只作交叉核对。")

    # ---------- 主预测 ----------
    rows_out, details = predict(args, agent, loaded, args.new_db)
    dg = digest(rows_out, details)

    print()
    print("=" * 108)
    print("1) 对照成立性：旧 solution 重放 vs 运行产物里的真实决策")
    print("=" * 108)
    bad = [d for d in details if not d["old_reproduces_artifact"]]
    print("  实例数 %d，逐条一致 %d，不一致 %d" % (len(details), len(details) - len(bad), len(bad)))
    for d in bad:
        print("    ✗ %-16s ci=%-4d 文件=%-46s 重放=(%s,%s) 产物=(%s,%s)"
              % (d["cve"], d["ci_id"], Path(d["local_file"]).name[:46],
                 d["old"]["decision"], d["old"]["reason"],
                 d["artifact_decision"], d["artifact_reason"]))
    # 更严的一致性：不只是决策，连证据字段与三个分数都对上
    nf = sum(1 for d in details if not d["old_fields_match"])
    ns = sum(1 for d in details if not d["old_structured_match"])
    nt = sum(1 for d in details if not d["old_total_match"])
    print("  更严校验：matched_fields 不一致 %d 条；structured_score 不一致 %d 条；"
          "total_score 不一致 %d 条" % (nf, ns, nt))
    for d in details:
        if not (d["old_fields_match"] and d["old_structured_match"] and d["old_total_match"]):
            print("    ✗ %-16s ci=%-4d 重放 mf=%s/%s structured=%s/%s total=%s/%s"
                  % (d["cve"], d["ci_id"], d["old"]["matched_fields"], d["artifact_matched_fields"],
                     d["old"]["structured_score"], d["artifact_structured_score"],
                     d["old"]["total_score"], d["artifact_total_score"]))
    # 样本级
    mism = [r for r in rows_out if r["has_own_candidate"] and r["old_admitted"] != r["artifact_admitted"]]
    print("  样本级（「门控放行自己」）：重放旧臂 %d/30，产物 %d/30，不一致 %d 个"
          % (sum(1 for r in rows_out if r["old_admitted"]),
             sum(1 for r in rows_out if r["artifact_admitted"]), len(mism)))
    for r in mism:
        print("    ✗ %s 重放=%s 产物=%s" % (r["cve"], r["old_admitted"], r["artifact_admitted"]))

    # 必须复现的 4 个未放行样本
    print()
    print("=" * 108)
    print("2) 硬性对照：4 个未放行样本，旧 solution 那一路必须复现产物里的拒因")
    print("=" * 108)
    EXPECT = {"CVE-2018-20854": "code_already_fixed",
              "CVE-2017-17053": "low_confidence_or_generic",
              "CVE-2018-6057": None,
              "CVE-2002-2443": "low_confidence_or_generic"}
    ctrl_ok = True
    for cve, want in EXPECT.items():
        ds = [d for d in details if d["cve"] == cve]
        reasons = sorted({d["old"]["reason"] for d in ds})
        got_artifact = sorted({d["artifact_reason"] for d in ds})
        agree = all(d["old_reproduces_artifact"] for d in ds)
        ok = agree and (want is None or want in reasons)
        ctrl_ok = ctrl_ok and ok
        print("  [%s] %-16s 旧臂拒因=%-46s 产物拒因=%s"
              % ("OK" if ok else "NG", cve, ",".join(reasons) or "-", ",".join(got_artifact) or "-"))
    if not ctrl_ok:
        print("\n  **对照组不成立** —— 离线重建与真实流水线不一致，")
        print("  按纪律**停止预测**：下面的数字不能当结论用，先修重建。")

    # ---------- 预测表 ----------
    print()
    print("=" * 108)
    print("3) 预测表（只看「自己那条 curated 候选」；实例级取最好 %s 者）" % "s(x)")
    print("=" * 108)
    print("  %-16s %-5s %-6s %-9s %-9s %-8s %-7s %-8s %s"
          % ("CVE", "ci", "sol变", "旧决策", "新决策", "s(x)旧→新", "旧放行", "预测", "卡点"))
    for r in rows_out:
        if not r["has_own_candidate"]:
            print("  %-16s %s" % (r["cve"], "**该样本没有自己的 curated 候选（单列）**"))
            continue
        for c in r["ci_rows"]:
            old_d = "/".join(sorted({x[0] for x in c["old_decisions"]})) or "-"
            new_d = "/".join(sorted({x[0] for x in c["new_decisions"]})) or "-"
            verdict = ("救回" if (not c["old_admitted"] and c["new_admitted"])
                       else ("**丢失**" if (c["old_admitted"] and not c["new_admitted"])
                             else ("保持放行" if c["new_admitted"] else "仍不放行")))
            print("  %-16s %-5d %-6s %-9s %-9s %-8s %-7s %-8s %s"
                  % (r["cve"], c["ci_id"], "是" if c["solution_changed"] else "否",
                     old_d, new_d,
                     "%s→%s" % (c["sx_old"], c["sx_new"]),
                     "是" if c["old_admitted"] else "否", verdict,
                     c["blocker"][:44]))

    print()
    print("  说明：`sol变`=该 curated 条目的 solution 在新库 v2 里是否变了。"
          "未变 → T2 对它没有任何作用（不是「救不回」，是「没动它」）。")

    # ---------- 汇总 ----------
    n = len(rows_out)
    art = sum(1 for r in rows_out if r["artifact_admitted"])
    old_n = sum(1 for r in rows_out if r["old_admitted"])
    new_n = sum(1 for r in rows_out if r["new_admitted"])
    rescued = [r["cve"] for r in rows_out if r["rescued"]]
    lost = [r["cve"] for r in rows_out if r["lost"]]
    no_own = [r["cve"] for r in rows_out if not r["has_own_candidate"]]

    print()
    print("=" * 108)
    print("4) 汇总预测")
    print("=" * 108)
    print("  口径定义（两个口径都报，避免「26/30」到底指哪个说不清）：")
    print("    A. **只看自己那条 curated 候选**（本任务要的口径，逐条离线重放）")
    print("    B. compare_arms 口径 = 任意通道解析到自己条目即算（含 sqlite/weaviate 通道）")
    print("  注：sqlite 通道读的是 `issue_patterns.solution`，**不由 T2 改动驱动**，")
    print("      所以只有口径 A 才能回答「T2 能救回几个」。")
    print()
    print("  口径 A（curated 通道，逐条重放）：")
    print("    · 旧 solution 离线重放（对照）      : %2d/%d" % (old_n, n))
    print("    · 新 solution 离线重放（T2 预测）   : %2d/%d" % (new_n, n))
    print("    · 净变化                            : %+d" % (new_n - old_n))
    print("    · 救回：%s" % (rescued or "无"))
    print("    · 丢失：%s" % (lost or "无"))
    print()
    print("  口径 B（任意通道，产物实测）：")
    print("    · Arm1 运行产物里自己条目被放行     : %2d/%d" % (art, n))
    print("    · 其中走非 curated 通道的样本       : %s"
          % ([r["cve"] for r in rows_out if r["other_channel_admitted"]] or "无"))
    n_other = sum(1 for r in rows_out if r["other_channel_candidates"])
    print("    · 有「同 pattern 的 sqlite/weaviate 候选」的样本: %d 个（其 solution 来自 "
          "`issue_patterns`，与 T2 无关；本任务不计入口径 A）" % n_other)
    for r in rows_out:
        if r["other_channel_candidates"]:
            ch = {}
            for x in r["other_channel_candidates"]:
                k = "%s(%s)" % (x["channel"], x["decision"])
                ch[k] = ch.get(k, 0) + 1
            print("        %-16s %s" % (r["cve"], ch))
    print("    · 各样本放行通道分布                : %s"
          % ", ".join("%s:%s" % (r["cve"], r["artifact_admit_channels"])
                      for r in rows_out if r["artifact_admit_channels"]) or "无")
    if no_own:
        print("    · 无自己的 curated 候选（单列，不参与计数）：%s" % no_own)

    print()
    print("  【一句话预测】把 `curated_issues.solution` 也重排后，"
          "30 样本里「门控放行自己那条 curated 候选」会从 %d/30 变成 **%d/30**（净 +%d）。"
          % (old_n, new_n, new_n - old_n))

    # ---------- 视野盲区 + 副作用普查 ----------
    print()
    print("=" * 108)
    print("4b) 视野盲区核对：自己的条目有没有「旧 run 里还不是候选、新 solution 后才冒出来」？")
    print("=" * 108)
    cov = own_entry_coverage(loaded)
    miss = [(c["cve"], c["missing"]) for c in cov if c["missing"]]
    print("  36 条自己的 curated 条目里，Arm1 候选里没出现过的：%s" % (miss or "无"))
    print("  → %s" % ("主表覆盖全部自己的条目，没有盲区" if not miss
                      else "⚠ 有盲区，主表低估了，需补量"))

    cen = fp_census(agent, loaded, args, details)
    print()
    print("=" * 108)
    print("4c) 副作用普查（**上界**）：外来 curated 条目会不会因新 solution 也命中而新放行？")
    print("=" * 108)
    print("  判据：curated 通道 s(x)=0.5·[克隆]+0.2·[basename 同]（curated 候选不走 "
          "`_apply_file_function_anchors`），")
    print("        所以外来条目必须「新针命中 且 裸文件名相同」才可能到 0.7 ≥ θ_s=0.65。")
    print("  扫描范围：30 样本共 %d 个被分析文件 × 新库 247 条 curated。" % cen["n_files"])
    print("  · 满足上述必要条件的 (样本,外来条目,文件) 三元组：%d 个" % len(cen["hits"]))
    print("  · 其中旧 solution 那一路**不命中**（即 T2 带来的新命中）：%d 个" % len(cen["new_hits"]))
    for h in cen["new_hits"][:20]:
        print("      %-16s %-46s curated_id=%-4d 旧 run 决策=%s"
              % (h["cve"], h["file"][:46], h["curated_id"], h["artifact_decision"]))
    print("  · 其中旧 run 里不是候选 / 未被放行的（= 新增误报上界）：%d 个" % len(cen["newly_admitted"]))
    print("  注意：这是**上界**，还没扣掉 `_derive_new_findings` 里")
    print("        「候选 error_type 出现在 issue 描述中就丢弃」这一条过滤，也没有重跑检索段——")
    print("        真实新增误报数 ≤ 这个数，**具体几个不确定**。")

    # ---------- 未救回样本的卡点 ----------
    print()
    print("=" * 108)
    print("5) 4 个未放行样本：逐条卡点")
    print("=" * 108)
    for cve in ("CVE-2018-20854", "CVE-2017-17053", "CVE-2018-6057", "CVE-2002-2443"):
        r = next((x for x in rows_out if x["cve"] == cve), None)
        if r is None:
            continue
        print("  %s：旧放行=%s → 新放行=%s （%s）"
              % (cve, r["old_admitted"], r["new_admitted"],
                 "救回" if r["rescued"] else "未救回"))
        for c in r["ci_rows"]:
            print("     ci=%-4d sol变=%-5s s(x) %s→%s  旧=(%s) 新=(%s)"
                  % (c["ci_id"], c["solution_changed"], c["sx_old"], c["sx_new"],
                     ",".join("%s%s" % (a, ("/" + b) if b else "") for a, b in c["old_decisions"]) or "-",
                     ",".join("%s%s" % (a, ("/" + b) if b else "") for a, b in c["new_decisions"]) or "-"))
            ds = [x for x in details if x["cve"] == cve and x["ci_id"] == c["ci_id"]]
            if ds:
                print("          针诊断（共 %d 条实例，min_tokens=%d）："
                      % (len(ds), agent.error_code_clone_min_tokens))
                print("            旧：片段 token 数=%s 命中=%s；克隆证据=%s"
                      % (ds[0]["old"]["frag_lens"], ds[0]["old"]["frag_hits"],
                         "有" if any(x["old"]["clone_hit"] for x in ds) else "无"))
                print("            新：片段 token 数=%s 命中=%s；克隆证据=%s"
                      % (ds[0]["new"]["frag_lens"], ds[0]["new"]["frag_hits"],
                         "有" if any(x["new"]["clone_hit"] for x in ds) else "无"))
            if c["blocker"] != "-":
                print("          卡点：%s" % c["blocker"])

    # ---------- 确定性 ----------
    print()
    print("=" * 108)
    print("6) 自验")
    print("=" * 108)
    print("  结果摘要 sha256[:16] = %s" % dg)
    if args.repeat > 1:
        same = True
        for i in range(args.repeat - 1):
            rows2, details2 = predict(args, agent, loaded, args.new_db)
            d2 = digest(rows2, details2)
            same = same and (d2 == dg)
            print("  重复运行 #%d 摘要 = %s  %s" % (i + 2, d2, "一致" if d2 == dg else "**不一致**"))
        print("  [%s] 确定性：重复 %d 次输出一致" % ("OK" if same else "NG", args.repeat))

    if args.reverse_control:
        print()
        print("  反向对照：把「新库」换回旧库（--new-db = 旧库），"
              "CVE-2018-20854 必须回落到 code_already_fixed")
        rows_rev, details_rev = predict(args, agent, loaded, args.old_db)
        ds = [d for d in details_rev if d["cve"] == "CVE-2018-20854"]
        reasons = sorted({d["new"]["reason"] for d in ds})
        ok = all(d["new"]["reason"] == "code_already_fixed" for d in ds) and bool(ds)
        n_rev = sum(1 for r in rows_rev if r["new_admitted"])
        print("  [%s] CVE-2018-20854 新臂（假新库=旧库）拒因 = %s"
              % ("OK" if ok else "NG", ",".join(reasons) or "-"))
        print("       此时全样本新臂放行 = %d/30（应回到 %d/30 = 旧臂）" % (n_rev, old_n))
        print("       救回名单 = %s（应为空）"
              % ([r["cve"] for r in rows_rev if r["rescued"]] or "无"))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(
        {"control_ok": ctrl_ok, "digest": dg,
         "haystack": hs_rows,
         "own_entry_coverage": own_entry_coverage(loaded),
         "fp_census": fp_census(agent, loaded, args, details),
         "samples": rows_out,
         "instances": details,
         "totals": {"n": n, "artifact_admitted": art, "old_admitted": old_n,
                    "new_admitted": new_n, "rescued": rescued, "lost": lost,
                    "no_own_candidate": no_own}},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n  明细已落盘：%s" % args.json_out)


if __name__ == "__main__":
    main()
