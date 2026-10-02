#!/bin/bash
# 用评测工具**自己的解析器**核实：干净层这一臂的候选是"产生了但被拒"，还是"根本没跑"。
# 为什么不用自己写的读法：产物结构有两层包装，手写读法会静默数出 0（我第一遍就数错了）。
set -u
cd /root/autodl-tmp/MAS || exit 1
venv/bin/python - <<'PY'
import sys
sys.path.insert(0, ".")
from utils.experiments.compare_arms import load_arm

arm = load_arm("reports/held_clean30_runs.txt")
print("样本数:", len(arm))
n_cand = n_adm = n_low = 0
reasons = {}
decisions = {}
for row in arm:
    cands = row.get("candidates") or []
    n_cand += len(cands)
    for c in cands:
        d = str(c.get("gating_decision") or "")
        decisions[d] = decisions.get(d, 0) + 1
        r = str(c.get("rejection_reason") or "")
        if r:
            reasons[r] = reasons.get(r, 0) + 1
    n_adm += len(row.get("admitted") or [])
    n_low += len(row.get("low_confidence") or [])
print("候选总数:", n_cand, " 放行:", n_adm, " 低置信保留:", n_low)
print("判定分布:", decisions)
print("拒绝理由分布（前 8）:", dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:8]))
print("（候选数 > 0 而放行 0 ⇒ 是门控拒掉的，不是「没跑」）")
PY
echo "VERIFY_BY_LOAD_ARM_DONE"
