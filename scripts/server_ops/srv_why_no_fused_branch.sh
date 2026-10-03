#!/bin/bash
# 查"融合分为何没走到 fused_semantic 分支"：把 fusion_score 高的候选原样打出来看。
set -u
cd /root/autodl-tmp/MAS || exit 1
RUN="${1:?用法: srv_why_no_fused_branch.sh <run 相对路径>}"

venv/bin/python - "$RUN" <<'PY'
import json, sys, glob, os
rel = sys.argv[1]
rows = []
for f in glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*.json"), recursive=True):
    try:
        j = json.loads(open(f, encoding="utf-8").read())
    except Exception:
        continue
    if not isinstance(j, dict):
        continue
    for key in ("retrieval_evidence", "gap_retrieval_evidence"):
        for b in (j.get(key) or []):
            for c in (b.get("candidates") or []):
                if isinstance(c, dict) and float(c.get("fusion_score") or 0) >= 0.7:
                    rows.append(c)
print("fusion_score >= θ(0.7) 的候选:", len(rows))
for c in rows[:10]:
    print("  ---")
    print("   fusion_score=%.4f  term=%.4f  unified_s=%s  channel=%s  layer=%s  sid=%s"
          % (float(c.get("fusion_score") or 0), float(c.get("fusion_semantic_term") or 0),
             c.get("unified_structured_score"), c.get("channel"), c.get("vector_layer"),
             c.get("sqlite_id")))
    print("   gate_branch=%r  gating_decision=%r  rejection_reason=%r  stats_n=%s"
          % (c.get("gate_branch"), c.get("gating_decision"), c.get("rejection_reason"),
             c.get("fusion_stats_n")))
    print("   matched_fields=%s" % (c.get("matched_fields"),))
    print("   gate_formula=%r" % (c.get("gate_formula"),))
PY
