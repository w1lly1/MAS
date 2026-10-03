#!/bin/bash
# 等一个批次跑完，然后按**臂标签**做收尾评测（run 清单 / 库外口径评测 / 完整性审计）。
#
# 用法: bash srv_wait_then_eval_arm.sh <tag> <批次日志> [期望样本数]
#   tag 例: clean30_off / clean30_on / kbself_on
#   产出: reports/held_clean30_<tag>_runs.txt / _eval.txt / _audit.json
#
# 为什么必须带 tag：③ 有多个臂跑同一批样本，**不能共用一个输出文件名** ——
# 否则后跑的臂会把先跑的臂的评测证据覆盖掉（昨晚 B6 的基线就是这么存下来的）。
set -u
TAG="${1:?用法: srv_wait_then_eval_arm.sh <tag> <批次日志> [期望样本数]}"
LOG="${2:?缺批次日志路径}"
EXPECT="${3:-30}"
cd /root/autodl-tmp/MAS || exit 1

echo "=== 0) 等批次结束（最多 90 分钟）==="
for i in $(seq 1 90); do
  if ! pgrep -f 'mas.py batch' >/dev/null 2>&1; then
    echo "  批次进程已结束（等了约 $((i-1)) 分钟）"
    break
  fi
  sleep 60
done
date +%H:%M:%S

echo
echo "=== 1) 生成 run 清单 ==="
# 命名按 tag 走（原来写死 `held_clean30_` 前缀，overlap15 那类臂会很难认）
venv/bin/python -X utf8 utils/experiments/make_run_list.py \
  --out "reports/${TAG}_runs.txt" --logs "$LOG" 2>&1 | tail -3
N=$(wc -l < "reports/${TAG}_runs.txt")
echo "  清单行数: $N（期望 $EXPECT）"

echo
echo "=== 2) 库外口径评测（--db 指线上那份 mas.db）==="
venv/bin/python -X utf8 utils/experiments/eval_held_runs.py \
  --arms "${TAG}=reports/${TAG}_runs.txt" \
  --db infrastructure/database/mas.db --max-rows 40 2>&1 | tail -70 \
  | tee "reports/${TAG}_eval.txt"

echo
echo "=== 3) 完整性审计 ==="
venv/bin/python -X utf8 utils/experiments/audit_run_completeness.py \
  --reports-root reports/analysis \
  --batch-summary "reports/${TAG}_runs.txt" \
  --json-out "reports/${TAG}_audit.json" 2>&1 | tail -12

echo
echo "=== 4) 该臂的开关生效性线索（产物里该出现/不该出现融合字段）==="
# 路径必须与第 1 步一致（`reports/${TAG}_runs.txt`）—— 原来这里还留着旧前缀，
# 导致 FIRST 为空、$D 退化成空串，抽查变成"在空路径上 grep"（白跑一次，见 chain_lam3.log）。
FIRST=$(head -1 "reports/${TAG}_runs.txt")
D="reports/analysis/${FIRST}"
echo "  抽查 run: $FIRST"
if [ -n "$FIRST" ] && [ -d "$D" ]; then
  echo "  含 fusion_score 的文件数: $(grep -rl 'fusion_score' "$D" 2>/dev/null | wc -l)"
  echo "  含 gate_branch 的文件数:  $(grep -rl 'gate_branch' "$D" 2>/dev/null | wc -l)"
  bash /root/autodl-tmp/srv_check_fusion_evidence.sh "$D" 2>/dev/null || true
else
  echo "  ⚠️ 找不到该臂的 run 目录（$D），这一步不做结论"
fi

echo
echo "=== 5) 磁盘 ==="
df -h /root/autodl-tmp | tail -1
echo "ARM_EVAL_DONE tag=${TAG}"
