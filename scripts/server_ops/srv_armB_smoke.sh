#!/bin/bash
# ③ 第 5 步：**门控融合的生效性检查**（1 个样本冒烟）。
#
# 为什么单独一步、而且要能失败：不看这个就比 A/B，等于不知道开关有没有接上 ——
# 结果"没差异"时无法区分"融合没用"和"开关没生效"。
#
# 用法: bash srv_armB_smoke.sh [λ] [θ]      （默认 1.5 0.7）
# 退出码: 0 = 生效；2 = 未生效（产物里没有融合字段）
set -u
LAM="${1:-1.5}"
THETA="${2:-0.7}"
cd /root/autodl-tmp/MAS || exit 1
LOG=/root/autodl-tmp/smoke_fusion_on.log

echo "=== 1) 打开开关 ==="
bash /root/autodl-tmp/srv_set_gate_fusion.sh on "$LAM" "$THETA" 2>&1 | grep -E 'gate_fusion|weaviate_top_k|配置 sha'

echo
echo "=== 2) 跑 1 个样本（held_clean30_smoke1.json）==="
date +%H:%M:%S
./venv/bin/python mas.py batch -c utils/experiments/held_clean30_smoke1.json > "$LOG" 2>&1
echo "退出码=$? 日志末尾:"
tail -3 "$LOG" | cut -c1-140

echo
echo "=== 3) 找这次 run ==="
RUN=$(ls -dt reports/analysis/CVE-2016-5191/*/ 2>/dev/null | head -1)
echo "  $RUN"
if [ -z "$RUN" ]; then echo "没找到 run 目录"; exit 2; fi

echo
echo "=== 4) 生效性判据（产物里必须出现融合字段）==="
N_SCORE=$(grep -rl 'fusion_score' "$RUN" 2>/dev/null | wc -l)
N_BRANCH=$(grep -rl 'gate_branch' "$RUN" 2>/dev/null | wc -l)
N_VETO=$(grep -rl 'fusion_semantic_term' "$RUN" 2>/dev/null | wc -l)
echo "  含 fusion_score        : $N_SCORE 个文件"
echo "  含 gate_branch         : $N_BRANCH 个文件"
echo "  含 fusion_semantic_term: $N_VETO 个文件"
echo "  gate_formula 示例:"
grep -rho 'admit = F(x) & ( s(x)>=theta_s[^"]*' "$RUN" 2>/dev/null | head -1
echo "  fusion_score 取值样例:"
grep -rho '"fusion_score": [0-9.]*' "$RUN" 2>/dev/null | head -5
echo "  判定分支分布:"
grep -rho '"gate_branch": "[a-z_]*"' "$RUN" 2>/dev/null | sort | uniq -c | head -5

echo
echo "=== 5) 门控判定分布（该样本的候选最终怎么判的）==="
grep -rho '"gating_decision": "[a-z_]*"' "$RUN" 2>/dev/null | sort | uniq -c | head -6

if [ "$N_SCORE" -gt 0 ] && [ "$N_BRANCH" -gt 0 ]; then
  echo
  echo "✅ 生效性检查通过（开关确实起作用了）"
  exit 0
fi
echo
echo "❌ 生效性检查**不通过** —— 产物里没有融合字段，A/B 不能往下做"
exit 2
