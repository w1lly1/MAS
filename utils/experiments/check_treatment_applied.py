# -*- coding: utf-8 -*-
"""处理生效性检查（**做任何 A/B 结论之前先跑这个**）。

今天已经踩过两次"改了却没生效"：一次是首轮没记行号、一次是汇总白名单丢了字段。
所以对比出现"两臂一模一样"时，第一件事不是下结论，而是问：
**处理组里那个改动真的作用到本次运行了吗？**

检查项（逐臂）：
  * gap 证据里 `query_semantic_used` 为真的比例（= 补漏通道有多少查询用上了语义）
  * 每个 run 的 gap 证据条数（判断通道是否在跑）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def scan(arm: str, runs_file: Path) -> None:
    strings = [ln.strip() for ln in runs_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total_gap = total_used = 0
    per_run = []
    for line in strings:
        cve, run = line.split("/", 1)
        d = ROOT / "reports/analysis" / cve / run / "second_pass" / "consolidated"
        if not d.exists():
            continue
        gap = used = 0
        for f in d.glob("*_r2.json"):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for e in (j.get("gap_retrieval_evidence") or []):
                gap += 1
                if e.get("query_semantic_used"):
                    used += 1
        total_gap += gap
        total_used += used
        per_run.append((cve, gap, used))
    print("=== %s ===" % arm)
    print("  gap 证据总数 %d，其中用上语义 %d（%.1f%%）"
          % (total_gap, total_used, 100.0 * total_used / max(1, total_gap)))
    zero = [c for c, g, u in per_run if g and u == 0]
    print("  gap 有证据但**完全没用上语义**的样本: %d 个%s"
          % (len(zero), ("  " + ", ".join(zero[:6])) if zero else ""))
    print("  没有任何 gap 证据的样本: %d 个" % len([c for c, g, u in per_run if g == 0]))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="形如 名称=reports/arm2_runs.txt")
    args = ap.parse_args()
    for item in args.pairs:
        name, path = item.split("=", 1)
        scan(name, Path(path))


if __name__ == "__main__":
    main()
