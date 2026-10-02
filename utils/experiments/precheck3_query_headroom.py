#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3 预检 3：**查询侧的余量** —— "用语义描述当查询文本"到底有没有换来更好的检索结果。

## 背景（为什么这个问题要单独问）

三臂实验已经给出**主指标无差异**的结论，并且**处理确实生效**（补漏通道用到语义的比例：
Arm1 32.2% / Arm3 32.5% / Arm2 0%）。于是问题变成：**处理生效了，为什么没换来结果？**
只有三种可能，本脚本量的是第一种：

| 可能 | 怎么量 |
|---|---|
| **① 查询侧换了文本，但检索结果几乎没变**（相似度分布、候选质量都一样） | 比较两臂 gap 证据里候选的 `semantic_score` 分布、通道构成、命中字段数 |
| ② 检索确实变好了，但门控把它们全拒了 | 比较两臂候选的拒绝理由分布（本脚本一并打印） |
| ③ 检索变好了、也放行了，但**总量太小**，淹在上万条候选里看不见 | 比较 gap 通道 vs 其它通道的规模占比 |

## 口径

* 只比**同一类证据块**（`gap_retrieval_evidence`），因为两臂差异只出在补漏通道的**查询文本**上；
* 用**逐样本配对**比较（同一 CVE 的同一 run 结构），避免样本不同造成的偏差；
* 判据全部来自产物里已有的字段，**不复算**任何分数（复算会引入"两套实现"的风险）。

用法：
    python -X utf8 utils/experiments/precheck3_query_headroom.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARMS = {
    "Arm1 语义查询（开）": ROOT / "reports/arm1_runs.txt",
    "Arm2 纯代码查询（关）": ROOT / "reports/arm2_runs.txt",
    "Arm3 语义查询（切库臂）": ROOT / "reports/arm3_runs.txt",
}


def blocks_of(run_rel: str):
    d = ROOT / "reports/analysis" / run_rel / "second_pass" / "consolidated"
    for f in sorted(glob.glob(os.path.join(str(d), "*_r2.json"))):
        try:
            j = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("gap_retrieval_evidence", "retrieval_evidence"):
            for b in (j.get(key) or []):
                if isinstance(b, dict):
                    yield key, b


def summarize(runs_file: Path) -> dict:
    runs = [ln.strip() for ln in runs_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = {"runs": len(runs), "gap_blocks": 0, "gap_cands": 0, "gap_with_fields": 0,
           "gap_sims": [], "gap_channels": {}, "gap_reject": {}, "all_blocks": 0, "all_cands": 0,
           "per_run_gap_cands": []}
    for rel in runs:
        c_in_run = 0
        for key, b in blocks_of(rel):
            cands = b.get("candidates") or []
            out["all_blocks" if key == "retrieval_evidence" else "gap_blocks"] += 1
            if key == "retrieval_evidence":
                out["all_cands"] += len(cands)
                continue
            out["gap_cands"] += len(cands)
            c_in_run += len(cands)
            for c in cands:
                out["gap_channels"][str(c.get("channel") or "?")] = \
                    out["gap_channels"].get(str(c.get("channel") or "?"), 0) + 1
                if c.get("matched_fields"):
                    out["gap_with_fields"] += 1
                s = c.get("semantic_score")
                if isinstance(s, (int, float)):
                    out["gap_sims"].append(float(s))
                r = str(c.get("rejection_reason") or "")
                if r:
                    out["gap_reject"][r] = out["gap_reject"].get(r, 0) + 1
        out["per_run_gap_cands"].append(c_in_run)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-reject", type=int, default=4)
    args = ap.parse_args()

    print("=" * 100)
    print("A3 预检 3：查询侧余量 —— 补漏通道的候选质量，两臂（语义查询 vs 纯代码查询）对比")
    print("=" * 100)
    res = {}
    for name, f in ARMS.items():
        if not f.is_file():
            print("  跳过 %s（缺 %s）" % (name, f.name))
            continue
        res[name] = summarize(f)
    print("%-22s %8s %10s %12s %10s %12s %12s"
          % ("臂", "run 数", "gap 块", "gap 候选", "有命中字段", "相似度中位", "相似度最高"))
    for name, r in res.items():
        sims = r["gap_sims"]
        print("%-22s %8d %10d %12d %10.1f%% %12s %12s"
              % (name, r["runs"], r["gap_blocks"], r["gap_cands"],
                 100.0 * r["gap_with_fields"] / max(1, r["gap_cands"]),
                 ("%.3f" % median(sims)) if sims else "-",
                 ("%.3f" % max(sims)) if sims else "-"))

    names = list(res)
    if len(names) >= 2:
        a, b = res[names[0]], res[names[-1]]
        print("\n--- 逐项对照（%s vs %s）---" % (names[0], names[-1]))
        print("  gap 候选总数      : %d vs %d" % (a["gap_cands"], b["gap_cands"]))
        print("  平均每样本 gap 候选: %.1f vs %.1f"
              % (mean(a["per_run_gap_cands"] or [0]), mean(b["per_run_gap_cands"] or [0])))
        print("  有命中字段占比    : %.1f%% vs %.1f%%"
              % (100.0 * a["gap_with_fields"] / max(1, a["gap_cands"]),
                 100.0 * b["gap_with_fields"] / max(1, b["gap_cands"])))
        if a["gap_sims"] and b["gap_sims"]:
            print("  相似度 中位/均值/最高: %.3f/%.3f/%.3f  vs  %.3f/%.3f/%.3f"
                  % (median(a["gap_sims"]), mean(a["gap_sims"]), max(a["gap_sims"]),
                     median(b["gap_sims"]), mean(b["gap_sims"]), max(b["gap_sims"])))
        print("\n  通道构成：")
        for nm, r in res.items():
            tot = max(1, r["gap_cands"])
            print("    %-22s %s" % (nm, {k: "%.0f%%" % (100.0 * v / tot)
                                        for k, v in sorted(r["gap_channels"].items(),
                                                           key=lambda kv: -kv[1])}))
        print("\n  拒绝理由（前 %d）：" % args.show_reject)
        for nm, r in res.items():
            top = sorted(r["gap_reject"].items(), key=lambda kv: -kv[1])[: args.show_reject]
            print("    %-22s %s" % (nm, top))

    out = ROOT / "reports/a3_query_headroom.json"
    out.write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "gap_sims"}
                               for k, v in res.items()}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("\n明细已落盘: %s（相似度明细略）" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
