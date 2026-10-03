#!/usr/bin/env bash
# **embedder 健康检验**：查询向量到底是不是"与文本无关的兜底向量"。
#
# 原理（很硬）：如果查询向量是常量/兜底向量，那么它不随查询文本变化 ⇒
#   **不同样本、不同 issue 的 weaviate top-k 命中集合会几乎一样**。
# 反之若 embedder 正常，不同查询的命中集合应当差别明显。
#
# 用法: bash srv_check_embedder_health.sh <runs 列表> [每臂取样条数]
set -u
cd /root/autodl-tmp/MAS || exit 1
RUNS="${1:?用法: srv_check_embedder_health.sh <runs 列表> [取样条数]}"
N="${2:-20}"

venv/bin/python - "$RUNS" "$N" <<'PY'
import json, sys, glob, os
from collections import Counter

runs = [ln.strip() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip()][: int(sys.argv[2])]
sets = Counter()          # top-k 命中集合 → 出现次数
spreads = []              # 单次查询内相似度的"散开程度"
n_q = 0
examples = []
for rel in runs:
    for f in sorted(glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*_r2.json"),
                              recursive=True)):
        try:
            j = json.loads(open(f, encoding="utf-8").read())
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
                if len(examples) < 3:
                    examples.append((rel.split("/")[0], [round(s, 4) for s in sims], list(key_t)[:8]))

print("取样：%d 个样本，%d 次查询（有 ≥2 条 weaviate 命中的）" % (len(runs), n_q))
print("不同的 top-k 命中集合数：%d" % len(sets))
print("出现最多的命中集合（可能说明「与文本无关」）：")
for k, v in sets.most_common(3):
    print("   出现 %3d 次  %s" % (v, list(k)[:10]))
if spreads:
    spreads.sort()
    print("单次查询内相似度最高−最低：中位 %.4f，最小 %.4f，最大 %.4f"
          % (spreads[len(spreads)//2], spreads[0], spreads[-1]))
print("样例：")
for cve, sims, ids in examples:
    print("   %-16s sims=%s ids=%s" % (cve, sims, ids))
print()
print("判读：")
print("  · 若「不同命中集合数」远小于查询数（比如 1~3 个），且它们出现次数占绝大多数 ⇒ **查询向量与文本无关**（embedder 失效）；")
print("  · 正常情况：命中集合应当**几乎每次都不同**（文本不同 → 向量不同 → 邻居不同）。")
PY
