# -*- coding: utf-8 -*-
"""语义通道的"天花板"在哪：**是门控把它拦了，还是它本来就没捞到相关的？**

## 为什么这是分水岭
要提高语义贡献，有两条完全不同的路：
* **开门的（门控侧）**：语义确实捞到了高相似的候选，只是被跨文件守卫拦下 → 改门控就行；
* **改库/改查询的（数据侧）**：语义候选的相似度**本来就不高** → 门控开了也没用，
  得重新设计索引文本/查询文本，或者承认这类样本在数据集里根本没有"语义重现"。

判别方法：看 weaviate 通道候选的 **semantic_score 分布**——
* 若大量候选 ≥ τ（0.65）却被拒 → 门控责任；
* 若**几乎没有**候选到 τ → 检索/数据责任。

用法: python -X utf8 utils/experiments/probe_semantic_channel_headroom.py --arms "Arm1=reports/arm1_runs.txt"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

TAU = 0.65          # 配置里的相似度门限 τ（semantic 通道要求 v(x) ≥ τ）
CODE_ANCHORS = {"class_pattern_in_code", "function_name_in_code",
                "file_basename_anchor", "file_pattern"}


def pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="形如 名称=run列表")
    ap.add_argument("--root", type=Path, default=ROOT / "reports/analysis")
    args = ap.parse_args()

    for item in args.arms:
        name, path = item.split("=", 1)
        runs = Path(path)
        if not runs.is_absolute():
            runs = ROOT / runs
        scores, admitted, with_anchor, rejected_by, cands = [], 0, 0, {}, []
        n_total = 0
        for line in runs.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cve, run = line.split("/", 1)
            d = args.root / cve / run / "second_pass/consolidated"
            for f in sorted(d.glob("*_r2.json")):
                if f.name.endswith(".gz"):
                    continue
                try:
                    j = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for key in ("retrieval_evidence", "gap_retrieval_evidence"):
                    for ev in (j.get(key) or []):
                        for c in (ev.get("candidates") or []):
                            if not isinstance(c, dict):
                                continue
                            if str(c.get("channel")) != "weaviate":
                                continue
                            n_total += 1
                            s = float(c.get("semantic_score") or 0.0)
                            scores.append(s)
                            if c.get("gating_decision") in {"formal_hit", "explanatory_hit"}:
                                admitted += 1
                            anch = bool(set(c.get("matched_fields") or []) & CODE_ANCHORS)
                            cands.append({"sem": s, "anchor": anch})
                            if anch:
                                with_anchor += 1
                            else:
                                r = str(c.get("rejection_reason") or "(未拒)")
                                rejected_by[r] = rejected_by.get(r, 0) + 1

        print("=" * 96)
        print("臂：%s   weaviate（语义）通道候选 %d 个" % (name, n_total))
        print("=" * 96)
        if not n_total:
            print("  没有语义候选")
            continue
        over = sum(1 for s in scores if s >= TAU)
        print("  semantic_score 分布: min=%.3f  p50=%.3f  p90=%.3f  p99=%.3f  max=%.3f"
              % (min(scores), pct(scores, 0.5), pct(scores, 0.9), pct(scores, 0.99), max(scores)))
        print("  ≥ τ(%.2f) 的候选: %d（%.2f%%）   ← 这些才是「门控若开门就能救」的量"
              % (TAU, over, 100.0 * over / n_total))
        print("  带代码级锚点（类名/函数名/文件名/文件路径）的候选: %d（%.2f%%）"
              % (with_anchor, 100.0 * with_anchor / n_total))
        print("  被放行: %d（%.2f%%）" % (admitted, 100.0 * admitted / n_total))
        print("  无代码锚点者的拒因分布:")
        for r, c in sorted(rejected_by.items(), key=lambda kv: -kv[1])[:5]:
            print("     %-30s %6d（%.1f%%）" % (r, c, 100.0 * c / n_total))
        # 结论倾向
        if over / n_total < 0.01:
            print("\n  ⇒ **倾向：数据/检索侧责任** —— 几乎没有语义候选达到 τ，"
                  "开门控也救不回多少；要动的是索引文本/查询文本（或承认数据集里没有语义重现）")
        else:
            print("\n  ⇒ **倾向：门控侧责任** —— 有一批高相似候选被拦，放宽跨文件守卫能直接见效")

        print("\n  【若按分布重新标定 τ】各阈值下会新增多少候选（含「带代码锚点」者）：")
        for t in (0.58, 0.60, 0.61, 0.62, 0.625):
            sel = [c for c in cands if c["sem"] >= t]
            anchors = sum(1 for c in sel if c["anchor"])
            print("     τ=%.3f → 候选 %6d（%.2f%%），其中带代码锚点 %d"
                  % (t, len(sel), 100.0 * len(sel) / n_total, anchors))


if __name__ == "__main__":
    main()
