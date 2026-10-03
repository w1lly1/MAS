#!/bin/bash
# 诊断：融合的语义项到底有没有真的算出来（还是分布统计没拿到、被静默记 0）。
#
# 背景：如果 `_fusion_layer_similarity_stats()` 拿不到分布（查询失败/条数<2），
# `s_sem` 会返回 0，于是 `fusion_score == s(x)`，看起来"融合没效果" ——
# 但那其实是**实现/环境问题**，不是设计结论。必须证据分开。
set -u
cd /root/autodl-tmp/MAS || exit 1
RUNS=${1:-reports/kbself_on_runs.txt}

venv/bin/python - "$RUNS" <<'PY'
import json, sys, glob, os
from collections import Counter

runs = [ln.strip() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip()]
stat = Counter()
formula = Counter()
term_vals = Counter()
fs_vals = Counter()
stats_n = Counter()
samples_with_fusion = 0
for rel in runs:
    files = glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*_r2.json"), recursive=True)
    touched = 0
    for f in files:
        try:
            j = json.loads(open(f, encoding="utf-8").read())
        except Exception:
            continue
        for key in ("retrieval_evidence", "gap_retrieval_evidence"):
            for b in (j.get(key) or []):
                for c in (b.get("candidates") or []):
                    if not isinstance(c, dict):
                        continue
                    stat["cand"] += 1
                    if "fusion_score" in c:
                        touched += 1
                        fs_vals[round(float(c["fusion_score"]), 4)] += 1
                        term_vals[round(float(c.get("fusion_semantic_term") or 0.0), 4)] += 1
                        stats_n[int(c.get("fusion_stats_n") or 0)] += 1
                        gb = c.get("gate_branch")
                        if gb:
                            stat["branch_" + str(gb)] += 1
                    gf = c.get("gate_formula")
                    if gf:
                        formula["含融合分支" if "lambda" in gf else "旧公式"] += 1
    if touched:
        samples_with_fusion += 1
print("样本数:", len(runs), " 有融合字段的样本:", samples_with_fusion)
print("候选总数:", stat["cand"])
print("有 fusion_score 的候选:", sum(fs_vals.values()))
print("gate_branch 分布:", {k[7:]: v for k, v in stat.items() if k.startswith("branch_")})
print("公式分布:", dict(formula))
print("fusion_stats_n 分布:", dict(stats_n))
print("fusion_semantic_term 取值分布（前 6）:", dict(term_vals.most_common(6)))
print("fusion_score 取值分布（前 8）:", dict(fs_vals.most_common(8)))
PY
