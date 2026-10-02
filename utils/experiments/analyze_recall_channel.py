# -*- coding: utf-8 -*-
"""回答一个核心问题：**召回到底是"词元匹配"还是"语义匹配"来的？**

## 判据从代码里读出来的（不是猜字段名）
五个候选构造器只有三条召回路径（`_collect_evidence` 里）：

| 通道 | 怎么召回的 | 归类 |
|---|---|---|
| `weaviate` | 向量检索（带 `semantic_score` = 向量相似度） | **语义** |
| `sqlite` | 预取的 KB 条目过**字面/结构匹配器**（`_match_pattern`：error_type/file_pattern/class_pattern/函数名/描述词面…） | **词元** |
| `curated_issue` | 取出全部 curated 条目过 `_match_curated_issue()`：主键是 **`error_code_clone`**（"修复前代码的连续词元序列"在被分析文件里命中）+ basename + 描述词面 | **词元** |

两条通道**没有任何向量参与**，命中一律来自字符串/词元比较。所以：
**按通道拆 = 按"词元召回 / 语义召回"拆。**

## 另外两个正交的量（一并报出来，避免只看通道会漏信息）
* `semantic_score > 0`：候选的**得分**里向量相似度真的贡献了（可能是跨通道合并来的）；
* `matched_fields` 非空：候选的**得分**里有字面/结构证据。

## 用法
    python -X utf8 utils/experiments/analyze_recall_channel.py \
        --arms "Arm2基线=reports/arm2_runs.txt" "T2优化后=reports/t2_runs.txt" \
        --db reports/mas_live.db [--root reports/analysis]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

SEMANTIC_CHANNEL = "weaviate"          # 唯一走向量的通道
LEXICAL_CHANNELS = {"sqlite", "curated_issue"}


def iter_r2(run_dir: Path, include_full_layer: bool = False):
    """兼容两种落盘布局，**默认只读 second_pass 那一套**。

    ⚠️ 踩过的坑：最初把 `fullLayer/consolidated/*_r1.json`（另一阶段的产物）也算进来，
    结果"同一批证据被数了两遍"——服务器上的 run 目录两套都有、本地还原的只有一套，
    于是 T2 的数字正好是三臂的 2 倍（922 vs 461 条证据），**口径不一致就没法比**。
    （发现方式：`2×` 太整齐了，怀疑度量而不是相信结果。）
    """
    seen = set()
    patterns = ["second_pass/consolidated/*_r2.json", "*_r2.json"]
    if include_full_layer:
        patterns.append("fullLayer/consolidated/*_r1.json")
    for pattern in patterns:
        for f in sorted(run_dir.glob(pattern)):
            if f.name.endswith(".gz") or f in seen:
                continue
            seen.add(f)
            yield f


def load_arm(runs_file: Path, root: Path, id_by_title, ci_to_pattern,
             include_full_layer: bool = False):
    samples = [l.strip() for l in runs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    per_sample = []
    for line in samples:
        cve, run = line.split("/", 1)
        own = id_by_title.get(cve.upper())
        rec = {"cve": cve, "own": own, "cand_by_channel": Counter(),
               "sem_scores": [], "lex_ev": 0, "own_hits": [],
               "admitted_by_channel": Counter(), "reject_by_channel": {},
               "n_evidence": 0, "n_cand": 0}
        d = root / cve / run
        for f in iter_r2(d, include_full_layer):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key in ("retrieval_evidence", "gap_retrieval_evidence"):
                for ev in (j.get(key) or []):
                    rec["n_evidence"] += 1
                    for c in (ev.get("candidates") or []):
                        if not isinstance(c, dict):
                            continue
                        rec["n_cand"] += 1
                        ch = str(c.get("channel") or c.get("primary_channel") or "?")
                        rec["cand_by_channel"][ch] += 1
                        sem = float(c.get("semantic_score") or 0.0)
                        if sem > 0:
                            rec["sem_scores"].append(sem)
                        if c.get("matched_fields"):
                            rec["lex_ev"] += 1
                        sid = c.get("sqlite_id")
                        resolved = ci_to_pattern.get(int(sid)) if ch == "curated_issue" and sid is not None else sid
                        if own is not None and resolved is not None and int(resolved) == int(own):
                            rec["own_hits"].append({"channel": ch, "semantic": sem,
                                                    "matched": list(c.get("matched_fields") or []),
                                                    "decision": c.get("gating_decision")})
                        if c.get("gating_decision") in {"formal_hit", "explanatory_hit"}:
                            rec["admitted_by_channel"][ch] += 1
                        else:
                            d = rec["reject_by_channel"].setdefault(ch, Counter())
                            d[str(c.get("rejection_reason") or "(空)")] += 1
        per_sample.append(rec)
    return per_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="形如 名称=run列表路径")
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_live.db")
    ap.add_argument("--root", type=Path, default=ROOT / "reports/analysis",
                    help="run 目录所在的根（服务器上可指向解包出来的目录）")
    ap.add_argument("--show-own", action="store_true", help="逐样本打印「自己条目」的召回路径")
    ap.add_argument("--include-full-layer", action="store_true",
                    help="把 fullLayer 那一套产物也算进来（默认不算：否则同一证据会被数两遍，"
                         "而服务器有、本地还原的没有 → 两边口径不可比）")
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    ci_to_pattern = {int(i): int(p) for i, p in
                     con.execute("select id, pattern_id from curated_issues")}
    con.close()

    for item in args.arms:
        name, path = item.split("=", 1)
        runs = Path(path)
        if not runs.is_absolute():
            runs = ROOT / runs
        data = load_arm(runs, args.root, id_by_title, ci_to_pattern, args.include_full_layer)

        n_cand = sum(r["n_cand"] for r in data)
        by_ch = Counter()
        for r in data:
            by_ch.update(r["cand_by_channel"])
        sem_any = sum(len(r["sem_scores"]) for r in data)
        lex_any = sum(r["lex_ev"] for r in data)

        print("=" * 100)
        print("臂：%s（%d 个样本；%d 条证据记录；%d 个候选）" % (name, len(data), sum(r["n_evidence"] for r in data), n_cand))
        print("=" * 100)
        print("① 候选按**召回通道**（= 词元 vs 语义）")
        for ch, n in by_ch.most_common():
            kind = "语义（向量）" if ch == SEMANTIC_CHANNEL else ("词元/字面" if ch in LEXICAL_CHANNELS else "未知")
            print("     %-16s %7d  %5.1f%%   [%s]" % (ch, n, 100.0 * n / max(n_cand, 1), kind))
        sem_n = by_ch.get(SEMANTIC_CHANNEL, 0)
        lex_n = sum(by_ch.get(c, 0) for c in LEXICAL_CHANNELS)
        print("     → **语义召回 %d（%.1f%%） / 词元召回 %d（%.1f%%）**"
              % (sem_n, 100.0 * sem_n / max(n_cand, 1), lex_n, 100.0 * lex_n / max(n_cand, 1)))

        print("② 得分构成（与通道正交，两个都看才不漏）")
        print("     有语义分(semantic_score>0) 的候选: %d（%.1f%%）"
              % (sem_any, 100.0 * sem_any / max(n_cand, 1)))
        print("     有词元/结构证据(matched_fields) 的候选: %d（%.1f%%）"
              % (lex_any, 100.0 * lex_any / max(n_cand, 1)))

        own_recalled = [r for r in data if r["own_hits"]]
        print("③ **自己条目的召回路径**（这是「存量知识有没有被召回」最直接的量）")
        print("     自己的条目出现在候选里的样本: %d / %d" % (len(own_recalled), len(data)))
        cls = Counter()
        for r in own_recalled:
            chans = {h["channel"] for h in r["own_hits"]}
            if SEMANTIC_CHANNEL in chans:
                cls["语义（向量）"] += 1
            if chans & LEXICAL_CHANNELS:
                cls["词元/字面"] += 1
        for k, v in cls.most_common():
            print("       经 %-12s 被召回: %d 个样本" % (k, v))
        if args.show_own:
            for r in own_recalled:
                for h in r["own_hits"]:
                    print("       %-16s ch=%-14s sem=%.3f 决策=%-14s matched=%s"
                          % (r["cve"], h["channel"], h["semantic"], h["decision"], h["matched"]))

        adm = Counter()
        rej = {}
        for r in data:
            adm.update(r["admitted_by_channel"])
            for ch, reasons in r["reject_by_channel"].items():
                d = rej.setdefault(ch, Counter())
                d.update(reasons)
        if adm:
            tot_adm = sum(adm.values())
            print("④ 被放行（formal/explanatory）的候选通道分布: 共 %d 条" % tot_adm)
            for ch, n in adm.most_common():
                kind = "语义" if ch == SEMANTIC_CHANNEL else "词元"
                print("     %-16s %5d  %5.1f%%   [%s]" % (ch, n, 100.0 * n / tot_adm, kind))
        print("⑤ **每条通道的「放行率」与拒绝原因**（解释「语义为什么占比低」）")
        for ch, n in by_ch.most_common():
            a = adm.get(ch, 0)
            print("     %-16s 候选 %6d  放行 %4d（%.2f%%）" % (ch, n, a, 100.0 * a / max(n, 1)))
            for reason, c in rej.get(ch, Counter()).most_common(4):
                print("        拒因 %-26s %6d（%.1f%%）" % (reason or "(空)", c, 100.0 * c / max(n, 1)))
        print()


if __name__ == "__main__":
    main()
