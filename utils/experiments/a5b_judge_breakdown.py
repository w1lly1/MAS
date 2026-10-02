#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 A5b 的 LLM 配对判定缓存拆成一张"判别力表"（离线，纯统计）。

要回答的问题：这个判定器到底能不能分清"自己那条"和"别的条目"？
拆四个格子：pos/neg 样本 × 自己那条/其它候选，各自数 `same_defect` / `related` / `unrelated`。

用法：
    python -X utf8 utils/experiments/a5b_judge_breakdown.py
    python -X utf8 utils/experiments/a5b_judge_breakdown.py --verdicts reports/a5b_llm_verdicts.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = {1.0: "same_defect", 0.5: "related", 0.0: "unrelated"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=ROOT / "reports/a5_items_cache.json")
    ap.add_argument("--verdicts", type=Path, default=ROOT / "reports/a5b_llm_verdicts.json")
    args = ap.parse_args()

    items = {it["cve"]: it for it in json.loads(args.items.read_text(encoding="utf-8"))}
    verd = json.loads(args.verdicts.read_text(encoding="utf-8"))

    cells: dict[tuple, Counter] = defaultdict(Counter)
    for cve, per in verd.items():
        it = items.get(cve)
        if it is None:
            continue
        own = it.get("own")
        for sid_s, val in per.items():
            sid = int(sid_s)
            who = "自己那条" if own and sid == int(own) else "其它候选"
            cells[(it["kind"], who)][LABEL.get(val, "?")] += 1

    print("判定总数：%d 条（%d 个样本）"
          % (sum(sum(c.values()) for c in cells.values()), len(verd)))
    print()
    print("| 样本类型 | 候选 | same_defect(1.0) | related(0.5) | unrelated(0) | 合计 |")
    print("|---|---|---|---|---|---|")
    for kind in ("pos", "neg"):
        for who in ("自己那条", "其它候选"):
            c = cells.get((kind, who))
            if not c:
                continue
            n = sum(c.values())
            print("| %s | %s | %d（%.0f%%） | %d（%.0f%%） | %d | %d |"
                  % (kind, who, c["same_defect"], 100.0 * c["same_defect"] / n,
                     c["related"], 100.0 * c["related"] / n, c["unrelated"], n))
    print()
    tp = cells[("pos", "自己那条")]["same_defect"]
    n_own = sum(cells[("pos", "自己那条")].values())
    fp = (cells[("pos", "其它候选")]["same_defect"]
          + cells[("neg", "其它候选")]["same_defect"]
          + cells[("neg", "自己那条")]["same_defect"])
    n_other = (sum(cells[("pos", "其它候选")].values())
               + sum(cells[("neg", "其它候选")].values())
               + sum(cells[("neg", "自己那条")].values()))
    print("以 same_defect 为“命中”：正样本自己那条 %d/%d = %.0f%%，其它候选 %d/%d = %.0f%%"
          % (tp, n_own, 100.0 * tp / max(1, n_own), fp, n_other, 100.0 * fp / max(1, n_other)))
    print("⇒ 严格的“自己那条 vs 其它”判别力 = 差值 %.0f 个百分点（越大越能分）"
          % (100.0 * tp / max(1, n_own) - 100.0 * fp / max(1, n_other)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
