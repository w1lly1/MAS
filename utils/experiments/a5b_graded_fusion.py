# -*- coding: utf-8 -*-
"""**A5b**：把 A5 的结论落到"分级 s_lex + λ·s_sem + 单阈值 θ"，并回答"哪些参数不承重"。

## 与 A5 的三处不同（都是上一轮讨论直接要求的）
1. **`s_lex` 从二值换成分级**（复算生产字段，含 **1.0 上限**与**合并同义字段**）：
   ```
   s_lex = min(1.0, 0.5·[针命中] + 0.2·[同文件身份] + 0.25·[类名在代码里]
                    + 0.25·[函数名在代码里] + 0.15·[类名在描述且文件对上])
   ```
   * `[同文件身份]`：把生产的 `file_basename_anchor`(0.2) 与 `basename_match`(0.2) **合并成一个**——
     它们本来是同一件事的两个通道版本（重复计分，属过拟合面）；这里用**严格的 basename 相等**口径。
   * 生产版 `file_basename_anchor` 还有"basename 出现在 error_description 里"这类**很松**的写法，
     本脚本把它作为**对照组**单列（`loose_file`），用来量"这条松判据带来多少判定差异"。
2. **加入权重消融**：把每个权重 ±50%／置 0，统计**判定翻转的 (样本, 候选) 对数**。
   翻转 0 ⇒ 该参数**不承重**（可删/可固定），这是对"过拟合"的可执行回答。
3. **`s_sem` 换成 LLM 配对判定**（`same=1.0 / related=0.5 / unrelated=0`），并且**只对向量 top-M 重排**；
   同时保留向量 z 归一化作为对照。

## 三段式运行（LLM 慢，分开跑、结果落盘）
    python -X utf8 utils/experiments/a5b_graded_fusion.py --stage grade     # 分级 s_lex + 消融 + 向量档融合扫描
    python -X utf8 utils/experiments/a5b_graded_fusion.py --stage llm --llm-topk 3   # 判定并缓存（约 1 小时/135 次）
    python -X utf8 utils/experiments/a5b_graded_fusion.py --stage fusion    # 用 LLM 语义项扫 (兑换率, 严格度)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from utils.experiments.a5_semantic_path_backtest import (  # noqa: E402
    JUDGE_PROMPT, KB, _load_qwen, basename_of, stats_of,
)
from utils.kb_coverage import SOURCE_EXT  # noqa: E402

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
CACHE_ITEMS = ROOT / "reports/a5_items_cache.json"
CACHE_LLM = ROOT / "reports/a5b_llm_verdicts.json"

#: 分级 s_lex 的权重（合并同义字段后的版本）；消融就是改这张表
W_BASE = {"needle": 0.5, "file_identity": 0.2, "class_in_code": 0.25,
          "func_in_code": 0.25, "class_desc_anchor": 0.15}
FUNC_RE = re.compile(r"\b([a-z_][a-z0-9_]{3,})\s+function\b")


def load_kb_rows(dump: Path) -> dict:
    rows = {}
    for line in dump.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = int(r["sqlite_id"])
        rows.setdefault(sid, {}).update({
            "file_pattern": r.get("file_pattern") or "",
            "class_pattern": r.get("class_pattern") or "",
            "error_description": r.get("error_description") or "",
            "solution": r.get("solution") or "",
            "llm_semantic": r.get("llm_semantic") or "",
            "error_type": r.get("error_type") or "",
        })
    return rows


def sample_file_text(cve: str, fname: str) -> str:
    d = DS / "before" / cve
    if not d.is_dir():
        return ""
    for f in d.rglob(fname):
        if f.is_file():
            return f.read_text(encoding="utf-8", errors="ignore")
    files = [x for x in sorted(d.rglob("*")) if x.is_file() and x.suffix.lower() in SOURCE_EXT]
    return files[0].read_text(encoding="utf-8", errors="ignore") if files else ""


def grade_item(agent, item, kb_rows, weights=None, loose_file=False):
    """复算该样本对**全部 200 条**的 matched_fields 与分级 s_lex。"""
    w = dict(W_BASE if weights is None else weights)
    text = sample_file_text(item["cve"], item["file"])
    low = text.lower()
    toks = agent._tokenize_code(text)
    fbase = basename_of(item["file"])
    graded = {}
    for sid in item["sims"]:
        row = kb_rows.get(int(sid)) or {}
        fields = []
        # ① 针命中（主匹配键）
        frags = agent._extract_error_code_fragments(row.get("solution", ""))
        if frags and any(agent._is_contiguous_subseq(fr, toks) for fr in frags):
            fields.append("needle")
        # ② 同文件身份（严格：basename 相等）
        kb_base = basename_of(row.get("file_pattern", ""))
        same = bool(kb_base and fbase and kb_base == fbase)
        if same:
            fields.append("file_identity")
        elif loose_file:
            # 生产版 `file_basename_anchor` 的松口径：basename 出现在 error_description 里，
            # 或路径互相包含。单列为对照组，用来量它带来多少差异。
            ed = str(row.get("error_description", "")).lower()
            fp = str(row.get("file_pattern", "")).lower()
            if (fbase and fbase in ed) or (fp and kb_base and kb_base in low):
                fields.append("file_identity")
        # ③ 类名出现在被分析代码里
        cls = str(row.get("class_pattern", "")).strip().lower()
        if cls and cls in low:
            fields.append("class_in_code")
        elif cls and cls in str(row.get("error_description", "")).lower() and fbase in low:
            fields.append("class_desc_anchor")
        # ④ 函数名（从 error_description 抽）出现在代码里
        m = FUNC_RE.search(str(row.get("error_description", "")).lower())
        if m and m.group(1) in low:
            fields.append("func_in_code")
        s = min(1.0, sum(w.get(f, 0.0) for f in fields))     # ← 生产也有 1.0 上限
        graded[int(sid)] = {"s_lex": round(s, 4), "fields": fields}
    return graded


def admit_set(item, graded, rule, lam, theta, sem_key):
    """admit ⇔ s_lex + λ·s_sem ≥ θ（**只有一个阈值**）"""
    out = []
    for sid, g in graded.items():
        s_sem = float((item.get(sem_key) or {}).get(str(sid), 0.0) or 0.0)
        if g["s_lex"] + lam * s_sem >= theta:
            out.append(int(sid))
    return out


def add_vector_sem(items, key="s_sem_vec"):
    """语义项（向量版）：按"该查询在全库上的分布"标准化后映射到 [0,1]（z=4 → 1.0）。"""
    for it in items:
        mu, sd = stats_of(it["sims"])
        d = {}
        for sid, v in it["sims"].items():
            z = (v - mu) / sd if sd > 1e-9 else 0.0
            d[str(sid)] = max(0.0, min(1.0, z / 4.0))
        it[key] = d
    return items


def report(items, graded_all, lam, theta, sem_key, kb_rows):
    pos = [i for i in items if i["kind"] == "pos"]
    neg = [i for i in items if i["kind"] == "neg"]
    own_hit = sum(1 for i in pos if i["own"] in admit_set(i, graded_all[i["cve"]], None, lam, theta, sem_key))
    tot = same = cross = 0
    for i in neg:
        adm = admit_set(i, graded_all[i["cve"]], None, lam, theta, sem_key)
        tot += len(adm)
        fbase = basename_of(i["file"])
        for s in adm:
            kbase = basename_of((kb_rows.get(s) or {}).get("file_pattern", ""))
            if kbase and fbase and kbase == fbase:
                same += 1
            else:
                cross += 1
    return {"pos": own_hit, "pos_n": len(pos), "neg": tot, "same": same, "cross": cross, "neg_n": len(neg)}


def name_of(items, sid) -> str:
    for i in items:
        if i["own"] == sid:
            return i["cve"]
    return str(sid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path,
                    default=ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl")
    ap.add_argument("--stage", choices=("grade", "llm", "fusion"), default="grade")
    ap.add_argument("--llm-topk", type=int, default=3)
    ap.add_argument("--json-out", type=Path, default=ROOT / "reports/a5b_result.json")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    items = json.loads(CACHE_ITEMS.read_text(encoding="utf-8"))
    for it in items:
        it["sims"] = {int(k): v for k, v in (it.get("sims") or {}).items()}
    kb_rows = load_kb_rows(args.dump)
    items = add_vector_sem(items)
    print("样本 %d（正 %d / 负 %d），KB 条目 %d"
          % (len(items), sum(1 for i in items if i["kind"] == "pos"),
             sum(1 for i in items if i["kind"] == "neg"), len(kb_rows)))

    # ---------- 分级 s_lex（两种口径：合并后严格版 / 生产的松版） ----------
    graded_all, graded_loose = {}, {}
    for it in items:
        graded_all[it["cve"]] = grade_item(agent, it, kb_rows)
        graded_loose[it["cve"]] = grade_item(agent, it, kb_rows, loose_file=True)
    # 统计证据字段分布（让人看清 s_lex 到底由什么构成）
    from collections import Counter
    cnt = Counter()
    for g in graded_all.values():
        for v in g.values():
            for f in v["fields"]:
                cnt[f] += 1
    print("\n证据字段命中次数（分级版，全部 200 条 × %d 个样本）: %s" % (len(items), dict(cnt)))

    if args.stage in ("grade", "fusion"):
        print("\n" + "=" * 104)
        print("A5b 融合扫描：score = s_lex + λ·s_sem ≥ θ      （语义项 = %s）"
              % ("LLM 判定" if args.stage == "fusion" else "向量 z 归一化"))
        print("=" * 104)
        sem_key = "s_sem_llm" if args.stage == "fusion" else "s_sem_vec"
        if args.stage == "fusion" and not CACHE_LLM.is_file():
            print("  ⚠️ 还没有 LLM 判定缓存（先跑 --stage llm）")
            return 1
        if args.stage == "fusion":
            verdicts = json.loads(CACHE_LLM.read_text(encoding="utf-8"))
            for it in items:
                it["s_sem_llm"] = {str(s): v for s, v in (verdicts.get(it["cve"]) or {}).items()}
        print("  %-26s %10s %10s %10s %12s" % ("(λ, θ)", "正命中", "负放行", "同文件", "跨文件"))
        rows = []
        for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            for theta in (0.65, 0.7, 0.9, 1.0, 1.2, 1.5):
                r = report(items, graded_all, lam, theta, sem_key, kb_rows)
                rows.append({"lam": lam, "theta": theta, **r})
                print("  %-26s %7d/%-2d %10d %10d %12d"
                      % ("λ=%.1f, θ=%.2f" % (lam, theta), r["pos"], r["pos_n"], r["neg"],
                         r["same"], r["cross"]))
        print("\n  【对照】生产的松口径 file_basename_anchor（λ=0, θ=0.65）")
        r = report(items, graded_loose, 0.0, 0.65, sem_key, kb_rows)
        print("    正命中 %d/%d，负放行 %d（同文件 %d / 跨文件 %d）"
              % (r["pos"], r["pos_n"], r["neg"], r["same"], r["cross"]))

    # ---------- 权重消融 ----------
    if args.stage == "grade":
        print("\n" + "=" * 104)
        print("权重消融：固定 λ=1.0, θ=1.0（向量档），逐个扰动权重，数**判定翻转的 (样本,条目) 对**")
        print("=" * 104)
        base_adm = {it["cve"]: set(admit_set(it, graded_all[it["cve"]], None, 1.0, 1.0, "s_sem_vec"))
                    for it in items}
        for key in W_BASE:
            for factor, label in ((1.5, "+50%"), (0.5, "-50%"), (0.0, "置0")):
                w = dict(W_BASE)
                w[key] = w[key] * factor
                alt = {it["cve"]: set(admit_set(it, grade_item(agent, it, kb_rows, weights=w),
                                                None, 1.0, 1.0, "s_sem_vec")) for it in items}
                flips = sum(len(base_adm[c] ^ alt[c]) for c in base_adm)
                print("  %-20s %-6s → 判定翻转 %4d 对" % (key, label, flips))
        # 合并同义字段 vs 不合并（用 0.4 的双计分口径模拟"没合并"）
        w_unmerged = dict(W_BASE)
        w_unmerged["file_identity"] = 0.4
        alt = {it["cve"]: set(admit_set(it, grade_item(agent, it, kb_rows, weights=w_unmerged),
                                        None, 1.0, 1.0, "s_sem_vec")) for it in items}
        flips = sum(len(base_adm[c] ^ alt[c]) for c in base_adm)
        print("  %-20s %-6s → 判定翻转 %4d 对" % ("file_identity 双计分", "(0.4)", flips))
        loose_alt = {it["cve"]: set(admit_set(it, graded_loose[it["cve"]], None, 1.0, 1.0, "s_sem_vec"))
                     for it in items}
        flips = sum(len(base_adm[c] ^ loose_alt[c]) for c in base_adm)
        print("  %-20s %-6s → 判定翻转 %4d 对" % ("file 松口径", "(prod)", flips))

    # ---------- LLM 判定阶段 ----------
    if args.stage == "llm":
        tok, model, torch = _load_qwen()
        verdicts = json.loads(CACHE_LLM.read_text(encoding="utf-8")) if CACHE_LLM.is_file() else {}
        done = 0
        for it in items:
            cve = it["cve"]
            verdicts.setdefault(cve, {})
            ranked = sorted(it["sims"].items(), key=lambda kv: -kv[1])
            picks = [s for s, _ in ranked[:args.llm_topk]]
            if it["own"] and it["own"] not in picks:
                picks[-1] = it["own"]
            for sid in picks:
                if str(sid) in verdicts[cve]:
                    continue
                row = kb_rows.get(sid) or {}
                idx = int((it.get("argmax") or {}).get(str(sid), 0) or 0)
                chunks = it.get("chunks") or []
                chunk_txt = (chunks[idx]["text"] if idx < len(chunks) else "")[:1500]
                p = JUDGE_PROMPT % (chunk_txt, row.get("error_type", ""),
                                    (row.get("error_description") or "")[:600],
                                    (row.get("solution") or "")[:600],
                                    (row.get("llm_semantic") or "")[:600])
                ids = tok([tok.apply_chat_template([{"role": "user", "content": p}],
                                                   tokenize=False, add_generation_prompt=True)],
                          return_tensors="pt")
                with torch.no_grad():
                    out = model.generate(**ids, max_new_tokens=16, do_sample=False)
                reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
                val = 0.0
                if "same_defect" in reply:
                    val = 1.0
                elif "related" in reply:
                    val = 0.5
                verdicts[cve][str(sid)] = val
                done += 1
                print("     [%s] %-16s sid=%-4d sim=%.3f base=%-26s → %.1f"
                      % (it["kind"], cve, sid, it["sims"][sid],
                         basename_of(row.get("file_pattern", ""))[:26], val))
                CACHE_LLM.write_text(json.dumps(verdicts, ensure_ascii=False), encoding="utf-8")
        print("\n本次新增判定 %d 条，累计 %d 条 → %s"
              % (done, sum(len(v) for v in verdicts.values()), CACHE_LLM))

    args.json_out.write_text(json.dumps({"stage": args.stage}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
