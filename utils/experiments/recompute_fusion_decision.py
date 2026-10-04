#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**融合判定重算器**：用已有产物 + 本地库向量，离线重算"换任意 (λ, θ) 会放行哪几条"。

## 🔴 状态（2026-10-04）：**尚未通过校验，不要用它给 λ/θ 定价**

拿真机数据校验（fusion ON、λ=1.5、θ=0.7 的 smoke 批次）结果：
**放行数一致（14 vs 14），但 134 条 DNF 候选里有 14 条的语义项对不上**，差值 0.04~0.66。

观察到的关键反例（说明我对生产口径的理解还缺一块）：同一个证据块里、**同一个 `sid`
（`sid=54`）的两条候选（curated_issue / weaviate）记录的语义项竟然不同**
（0.6832 vs 0.4886），而重算对两者给出同一个值。按现在的理解
（语义项 = max over 该查询已查各层 of f(sim, μ, σ)），同块同 sid 必然同值 ——
所以生产那一步至少还依赖了某个我没识别到的量（怀疑与 gap 通道
"一个 chunk 一条查询、但证据块与 `gap_code_chunks[bi]` 不是 1:1 对应"有关）。
在弄清之前，**它算出来的任何 λ/θ 定价都不可信**。

**替代做法（已采纳）**：λ/θ 的问题改用**真臂**回答 —— 直接用生产代码在不同 λ 下跑同一样本集，
读产物里的放行/own/非 own。多花 GPU 时间，但生产代码就是唯一权威口径，
不存在"离线复现口径"这一类风险（正是《07》纪律 6 的由来）。

## 为什么可以离线重算

融合**只改判定、不改候选集合**。判定需要三样东西，全都能重建：

| 需要 | 从哪来 |
|---|---|
| `s(x)` 结构化分 | 候选里的 `unified_structured_score`（**只有走到 DNF 的候选才有**，正好等于判定要考虑的集合） |
| 语义项 `s_sem` | 用**生产函数**重算：查询文本 = `_build_query_text(issue)`（产物里 `issues[]` 带 `llm_semantic`）→ `_query_embed(text, 层)`；库侧 = **dump 里真实存的层向量**；再按生产口径算"整层分布 μ/σ"与相对分 |
| 否决 / 守卫 | `_fusion_veto_blocks(matched_fields)`（生产函数）；守卫拦掉的候选**没有** `unified_structured_score`，因此天然被排除 |

## ⚠️ 一个必须换算对的地方（这里错了整份结论就废）

生产写的是 `similarity = 1.0 - distance / 2.0`，而 Weaviate 的**余弦距离 = 1 − cos**
⇒ 生产的"相似度" = **`(1 + cos) / 2`**，**不是 cos 本身**。
所以本地用 dump 向量算相似度时，必须做同样的换算。

## 两种用法

1. **校验**（拿真机数据验它）：`--lam` 给成**当时那臂实际用的 λ**，把重算的语义项/放行数与产物里记录的值比；
2. **定价**（拿它给新参数估价）：给别的 λ/θ，重算"会多放行几条、是 own 还是别的条目"。

用法：
    python -X utf8 utils/experiments/recompute_fusion_decision.py \
        --runs reports/kbself_fix_runs.txt --lam 1.5 --theta 0.7 --validate
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

DUMP = ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl"
DB = ROOT / "reports/mas_rebuild_candidate_v2.db"
LAYERS = ("semantic", "code_pattern", "solution", "full")


def load_kb(dump: Path):
    """{层: (id 列表, 向量矩阵)}，向量来自 dump（生产存的）。"""
    rows = defaultdict(list)
    for line in dump.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        L = str(r.get("vector_layer") or "").strip().lower()
        v = r.get("_vector") or r.get("vector")
        if L in LAYERS and v:
            rows[L].append((int(r["sqlite_id"]), [float(x) for x in v]))
    out = {}
    for L, items in rows.items():
        out[L] = ([i for i, _ in items], [v for _, v in items])
    return out


def cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--theta", type=float, default=0.7)
    ap.add_argument("--theta-s", type=float, default=0.65, dest="theta_s")
    ap.add_argument("--dump", type=Path, default=DUMP)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--validate", action="store_true",
                    help="与产物里记录的 fusion_semantic_term / gating_decision 逐条比对")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import (
        AIDrivenSecondPassAnalysisAgent,
    )
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    agent.gate_fusion_lambda = args.lam
    agent.gate_fusion_theta = args.theta
    agent.gate_fusion_enabled = True

    kb = load_kb(args.dump)
    print("库向量层：%s" % {L: len(v[0]) for L, v in kb.items()})

    con = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    con.close()

    runs = [ln.strip() for ln in args.runs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    term_diff = []            # |重算 term − 记录 term|
    mismatch_detail = []      # 不一致候选的明细（诊断用）
    counts = Counter()
    new_own, new_other = [], []
    ev_cache: dict = {}

    for rel in runs:
        cve = rel.split("/")[0]
        own = id_by_title.get(cve.upper())
        for f in sorted(glob.glob(str(ROOT / "reports/analysis" / rel / "second_pass" / "**" / "*_r2.json"),
                                 recursive=True)):
            try:
                j = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception:
                continue
            issues = j.get("issues") or []
            art_file = j.get("file") or ""
            gap_chunks = j.get("gap_code_chunks") or []
            sem_lookup = agent._semantic_lookup_from_issues(issues)
            for key in ("retrieval_evidence", "gap_retrieval_evidence"):
                blocks = j.get(key) or []
                for bi, b in enumerate(blocks):
                    ck = (rel, bi, key)
                    if ck in ev_cache:
                        stats = ev_cache[ck]
                    else:
                        if key == "gap_retrieval_evidence":
                            # ⚠️ **gap 补漏通道的查询文本是"分片级"的**：生产用
                            # `_code_chunk_as_issue(chunk, semantic_lookup)` 造一个伪 issue
                            # （分片前 500 字 + 与该分片行区间重叠的首轮 llm_semantic），
                            # 再交给 `_build_query_text`。第一版重算器错用了 issue 级文本 ⇒ 对不上。
                            chunk = (gap_chunks[bi] if bi < len(gap_chunks)
                                     else {"text": str(b.get("code_chunk") or "")})
                            if not isinstance(chunk, dict):
                                chunk = {"text": str(chunk)}
                            issue_like = agent._code_chunk_as_issue(chunk, sem_lookup)
                            qfile = str(chunk.get("file") or art_file)
                            text = agent._build_query_text(
                                issue_like, qfile, str(issue_like.get("description") or ""),
                                str(issue_like.get("source") or "source_code_chunk"))
                        else:
                            # 主通道：用**原始 issue 字典**（与生产 `_collect_evidence(issue, ...)` 一致）
                            issue = issues[bi] if bi < len(issues) else {}
                            if issue and b.get("issue_description") and \
                                    str(issue.get("description") or "")[:40] != \
                                    str(b.get("issue_description") or "")[:40]:
                                issue = next((x for x in issues
                                              if str(x.get("description") or "")[:40] ==
                                              str(b.get("issue_description") or "")[:40]), issue)
                            qfile = str((issue or {}).get("file") or art_file)
                            text = agent._build_query_text(
                                issue, qfile, str(b.get("issue_description") or ""),
                                str(b.get("issue_source") or ""))
                        qv = {L: agent._query_embed(text, L) for L in kb}
                        stats = {}
                        for L, (ids, vecs) in kb.items():
                            sims, by_id = [], {}
                            for sid, v in zip(ids, vecs):
                                s = (1.0 + cos(qv[L], v)) / 2.0     # 生产口径：1 − distance/2
                                sims.append(s)
                                by_id[sid] = s
                            mu = sum(sims) / len(sims)
                            sd = (sum((x - mu) ** 2 for x in sims) / (len(sims) - 1)) ** 0.5
                            stats[L] = {"mu": mu, "sigma": sd, "sims_by_id": by_id}
                        ev_cache[ck] = stats
                    for c in (b.get("candidates") or []):
                        if not isinstance(c, dict) or "unified_structured_score" not in c:
                            continue      # 没走到 DNF（被守卫拦了）⇒ 判定不考虑它
                        counts["cand"] += 1
                        s = float(c.get("unified_structured_score") or 0.0)
                        ch = str(c.get("channel") or "").strip().lower()
                        sid = c.get("kb_pattern_id") if ch == "curated_issue" else c.get("sqlite_id")
                        try:
                            sid = int(sid) if sid is not None else None
                        except (TypeError, ValueError):
                            sid = None
                        # 语义项：生产函数（含否决），取各层最大
                        if agent._fusion_veto_blocks(c.get("matched_fields")):
                            term = 0.0
                        else:
                            terms = []
                            for L, st in stats.items():
                                if sid is None or sid not in st["sims_by_id"]:
                                    continue
                                terms.append(agent._fuse_semantic_term(
                                    st["sims_by_id"][sid], st["mu"], st["sigma"],
                                    agent.gate_fusion_z_scale))
                            term = max(terms) if terms else 0.0
                        if args.validate and "fusion_semantic_term" in c:
                            rec_term = float(c.get("fusion_semantic_term") or 0.0)
                            term_diff.append(abs(term - rec_term))
                            if abs(term - rec_term) > 1e-6 and len(mismatch_detail) < 12:
                                mismatch_detail.append({
                                    "run": rel, "block": "%s[%d]" % (key, bi),
                                    "ch": c.get("channel"), "sid": sid,
                                    "s": round(s, 3),
                                    "term_rec": round(rec_term, 4), "term_calc": round(term, 4),
                                    "layer": c.get("vector_layer"),
                                    "terms_by_layer": {L: round(agent._fuse_semantic_term(
                                        st["sims_by_id"][sid], st["mu"], st["sigma"],
                                        agent.gate_fusion_z_scale), 4)
                                        for L, st in stats.items()
                                        if sid is not None and sid in st["sims_by_id"]},
                                })
                        fused = s + args.lam * term
                        if s >= args.theta_s:
                            new_dec = "formal_hit"
                        elif fused >= args.theta:
                            new_dec = "formal_hit"
                            (new_own if (own is not None and sid == own) else new_other).append(
                                {"cve": cve, "sid": sid, "s": round(s, 3), "term": round(term, 4),
                                 "fused": round(fused, 3), "ch": ch,
                                 "mf": sorted(set(c.get("matched_fields") or []))})
                            counts["new_admit"] += 1
                        else:
                            new_dec = c.get("gating_decision") or "discarded_hit"
                        if args.validate:
                            rec = c.get("gating_decision")
                            adm_new = new_dec in ("formal_hit", "explanatory_hit")
                            adm_rec = rec in ("formal_hit", "explanatory_hit")
                            if adm_new != adm_rec:
                                counts["dec_mismatch"] += 1
                            counts["adm_rec" if adm_rec else "disc_rec"] += 1
                            counts["adm_new" if adm_new else "disc_new"] += 1

    print("\n候选（走到 DNF 的）：%d" % counts["cand"])
    if args.validate and term_diff:
        nz = [d for d in term_diff if d > 1e-6]
        print("语义项逐条比对：%d 条可比，其中**不一致 %d 条**，最大偏差 %.6f"
              % (len(term_diff), len(nz), max(term_diff)))
        if nz:
            print("  ⚠️ 偏差 >1e-6 ⇒ 重算器与生产不同源，**不能**用它给别的 λ 定价")
            print("  --- 不一致候选明细（前 %d 条）---" % len(mismatch_detail))
            for m in mismatch_detail:
                print("    %-28s %-16s sid=%-4s s=%.2f 记录=%.4f 重算=%.4f 层=%s"
                      % (m["block"], m["ch"], m["sid"], m["s"], m["term_rec"],
                         m["term_calc"], m["layer"]))
                print("        各层重算: %s" % m["terms_by_layer"])
        else:
            print("  ✅ 逐条一致（≤1e-6）⇒ 重算器与生产同源")
        print("放行判定比对：记录放行 %d / 重算放行 %d，不一致 %d 条"
              % (counts["adm_rec"], counts["adm_new"], counts["dec_mismatch"]))
    print("\n(λ=%.2f, θ=%.2f) 下【新增放行】(老规则不放行、融合放行)：%d 条（own %d / 其他 %d）"
          % (args.lam, args.theta, counts["new_admit"], len(new_own), len(new_other)))
    if new_own:
        print("  own 新增（= 召回）：")
        for x in new_own[: args.show]:
            print("    %-16s sid=%-4s s=%.2f term=%.4f fused=%.3f %s %s"
                  % (x["cve"], x["sid"], x["s"], x["term"], x["fused"], x["ch"],
                     [m for m in x["mf"] if m not in ("phenomenon_in_description",
                                                      "root_cause_in_description")]))
    if new_other:
        print("  其他条目新增（在库内样本上=多放行的面）：")
        for x in new_other[: args.show]:
            print("    %-16s sid=%-4s s=%.2f term=%.4f fused=%.3f %s"
                  % (x["cve"], x["sid"], x["s"], x["term"], x["fused"], x["ch"]))

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"lam": args.lam, "theta": args.theta, "counts": dict(counts),
             "new_own": new_own, "new_other": new_other,
             "term_max_abs_diff": (max(term_diff) if term_diff else None)},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n明细已落盘: %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
