# -*- coding: utf-8 -*-
"""通道互补性：**"有多少样本是靠语义通道才被放行的"**（词元 × 语义互补的直接证据）。

## 为什么单独量这件事

研究目标不是"哪个通道更强"，而是**语义词元互补**：词法（针/同文件/类名在码…）负责精确，
语义负责"说法不一样但讲的是同一件事"。所以要回答的是：

* 有多少样本的**自己条目**是**只靠语义通道**才被放行的（词法没捞到/没放行）
  ⇒ 这是互补性最硬的证据；
* 有多少是**只靠词法**（语义没起作用）；
* 有多少是**两条通道都**放行了（互补没被用到，但也不算反例）；
* 非自己条目的放行（误报面）分别来自哪条通道 ⇒ 哪条通道更危险。

## 输入

`extract_arm_summary.py` 产出的决策摘要 `<tag>.jsonl`（每样本一行，含每条放行记录的通道）。
这样本脚本**不需要 run 产物**，因此可以对着留档摘要长期复算（磁盘删了也能算）。

通道口径：`curated_issue` / `sqlite` = 词法侧（结构化匹配），`weaviate` = 语义侧（向量检索）。
一行放行记录属于哪侧由它的 `channel` 决定；同一条目被两条通道各放行一次时，两侧都记。

用法：
    python -X utf8 utils/experiments/analyze_channel_complementarity.py \
        --summaries reports/arm_summaries/baseline_fix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LEXICAL = {"curated_issue", "sqlite"}


def side_of(channel: str) -> str:
    ch = str(channel or "").strip().lower()
    if ch in LEXICAL:
        return "lexical"
    if ch == "weaviate":
        return "semantic"
    return "other"


def analyze(path: Path) -> dict:
    res = {"path": path.name, "samples": 0, "own_admitted": 0,
           "buckets": Counter(), "examples": defaultdict(list),
           "fp_by_channel": Counter(), "own_by_channel": Counter(),
           "missed": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        res["samples"] += 1
        own_sides = set()
        for a in row.get("admitted") or []:
            side = side_of(a.get("channel"))
            if a.get("is_own"):
                res["own_by_channel"][a.get("channel")] += 1
                own_sides.add(side)
            else:
                res["fp_by_channel"][a.get("channel")] += 1
        if row.get("sample_own_admitted"):
            res["own_admitted"] += 1
            if own_sides == {"semantic"}:
                bucket = "semantic_only"
            elif own_sides == {"lexical"}:
                bucket = "lexical_only"
            elif "semantic" in own_sides and "lexical" in own_sides:
                bucket = "both"
            else:
                bucket = "other_only"
            res["buckets"][bucket] += 1
            res["examples"][bucket].append(row["cve"])
        elif row.get("own_in_cand"):
            res["missed"].append(row["cve"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="通道互补性（语义能不能补词元的漏）")
    ap.add_argument("--summaries", nargs="+", required=True, type=Path)
    args = ap.parse_args()

    for p in args.summaries:
        path = p if p.is_absolute() else ROOT / p
        if not path.is_file():
            raise SystemExit("摘要不存在: %s" % path)
        r = analyze(path)
        print("=" * 92)
        print("臂 %s（%d 个样本）" % (r["path"], r["samples"]))
        print("=" * 92)
        print("  自己条目被放行的样本: %d / %d" % (r["own_admitted"], r["samples"]))
        b = r["buckets"]
        print("    只靠**语义**通道: %d 条  ← 互补性证据（词法这几条没放行）" % b.get("semantic_only", 0))
        if r["examples"].get("semantic_only"):
            print("       %s" % " ".join(sorted(r["examples"]["semantic_only"])))
        print("    只靠**词法**通道: %d" % b.get("lexical_only", 0))
        print("    两条通道都放行: %d" % b.get("both", 0))
        if b.get("other_only"):
            print("    其它通道: %d" % b["other_only"])
        print("  自己条目放行记录按通道: %s" % dict(r["own_by_channel"]))
        print("  非自己条目放行记录按通道（误报面）: %s" % (dict(r["fp_by_channel"]) or "无"))
        if r["missed"]:
            print("  有候选但没放行的样本: %s" % " ".join(r["missed"]))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
