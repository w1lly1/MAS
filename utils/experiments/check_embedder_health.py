#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**embedder 健康检验**：查询向量是不是"与文本无关的兜底向量"。

## 原理（很硬，不靠日志）

`codebert_embedder._fallback_embed()` 在模型加载失败时返回一个
**只在前 3 维非零**的"校验和向量"（长度、ord 和 %991、%313），且**不经白化**；
再经 C2 查询偏移一减，结果**几乎就是那个常量方向**。

于是有一个无法伪造的指纹：

> **若查询向量与文本无关 ⇒ 不同样本、不同 issue 的 weaviate top-k 命中集合会几乎完全一样。**

反过来，正常 embedder 下"文本不同 → 向量不同 → 邻居不同"，命中集合应当几乎每次都不同。

本脚本就量这个：取样若干 run，统计 (a) 不同的命中集合数、(b) 出现最多的集合占比、
(c) 单次查询内相似度的"散开程度"（兜底向量会把散开度压成常数）。

用法：
    python -X utf8 utils/experiments/check_embedder_health.py --runs reports/arm1_runs.txt --limit 12
    python -X utf8 utils/experiments/check_embedder_health.py --runs reports/kbself_fix_runs.txt --root reports/analysis
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT / "reports/analysis")
    ap.add_argument("--limit", type=int, default=12, help="取样多少个样本")
    args = ap.parse_args()

    runs = [ln.strip() for ln in args.runs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    runs = runs[: args.limit]
    sets: Counter = Counter()
    spreads = []
    n_q = 0
    examples = []
    for rel in runs:
        for f in sorted(glob.glob(str(args.root / rel / "second_pass" / "**" / "*_r2.json"),
                                 recursive=True)):
            try:
                j = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception:
                continue
            for key in ("retrieval_evidence", "gap_retrieval_evidence"):
                for b in (j.get(key) or []):
                    hits = b.get("weaviate_hits") or []
                    if len(hits) < 2:
                        continue
                    key_t = tuple(sorted({int(h.get("sqlite_id") or 0) for h in hits}))
                    sets[key_t] += 1
                    sims = sorted((float(h.get("similarity") or 0.0) for h in hits), reverse=True)
                    spreads.append(sims[0] - sims[-1])
                    n_q += 1
                    if len(examples) < 2:
                        examples.append((rel.split("/")[0], [round(s, 4) for s in sims[:6]],
                                         list(key_t)[:8]))

    print("文件: %s（取样 %d 个样本）" % (args.runs.name, len(runs)))
    print("可判定的查询数（有 ≥2 条 weaviate 命中）: %d" % n_q)
    if n_q == 0:
        print("  ⇒ 本次取样没有 weaviate 命中，无法判定（换一批 run）")
        return 0
    print("不同的 top-k 命中集合数: %d" % len(sets))
    top_set, top_n = sets.most_common(1)[0]
    print("出现最多的集合占比: %d/%d = %.0f%%" % (top_n, n_q, 100.0 * top_n / n_q))
    if spreads:
        spreads.sort()
        print("单次查询内相似度散开度（最高−最低）: 中位 %.4f｜最小 %.4f｜最大 %.4f"
              % (spreads[len(spreads) // 2], spreads[0], spreads[-1]))
    for cve, sims, ids in examples:
        print("  样例 %-16s sims=%s ids=%s" % (cve, sims, ids))

    bad = (len(sets) <= 2 and top_n / n_q > 0.8) or (spreads and max(spreads) - min(spreads) < 1e-6)
    print()
    if bad:
        print("❌ **判定：查询向量与文本无关（兜底向量）** —— 这一臂的向量/语义通道无意义")
    else:
        print("✅ 判定：查询向量随文本变化（embedder 正常）")
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
