#!/bin/bash
# 等一个**库内（kb-self）**批次跑完，然后按库内口径收尾：run 清单 → compare_arms → 审计 → 生效性。
#
# 用法: bash srv_wait_then_compare_arm.sh <tag> <批次日志> [期望样本数]
#
# 为什么不用 held 那套：库内样本的口径是"自己的条目有没有被放行"（召回），
# 而 held 口径是"放行数=误报面"。**用错口径会把召回读成误报。**
set -u
TAG="${1:?用法: srv_wait_then_compare_arm.sh <tag> <批次日志> [期望样本数]}"
LOG="${2:?缺批次日志路径}"
EXPECT="${3:-30}"
cd /root/autodl-tmp/MAS || exit 1

echo "=== 0) 等批次结束（按 pid，最多 120 分钟）==="
# 坑 49：这里原来是 `while pgrep -f 'mas.py batch'`，会匹配到调用方自己的命令行 ⇒ 死等
bash /root/autodl-tmp/srv_wait_batch.sh 120
date +%H:%M:%S

echo
echo "=== 1) 生成 run 清单 ==="
venv/bin/python -X utf8 utils/experiments/make_run_list.py \
  --out "reports/${TAG}_runs.txt" --logs "$LOG" 2>&1 | tail -3
echo "  清单行数: $(wc -l < "reports/${TAG}_runs.txt")（期望 $EXPECT）"

echo
echo "=== 2) 库内口径（compare_arms：检索段 / 门控段分开量）==="
venv/bin/python -X utf8 utils/experiments/compare_arms.py \
  --arms "${TAG}=reports/${TAG}_runs.txt" --db infrastructure/database/mas.db 2>&1 \
  | tail -45 | tee "reports/${TAG}_compare.txt"

echo
echo "=== 3) 完整性审计 ==="
venv/bin/python -X utf8 utils/experiments/audit_run_completeness.py \
  --reports-root reports/analysis \
  --batch-summary "reports/${TAG}_runs.txt" \
  --json-out "reports/${TAG}_audit.json" 2>&1 | tail -10

echo
echo "=== 4) 生效性（该臂该不该有融合字段）==="
# 注意：`$FIRST` 本身就是"CVE/run-id"（相对 reports/analysis），**不要再 `%/*`** ——
# 那样取到的是 CVE 目录（含该 CVE 的全部历史 run），诊断会扫错范围、数出 0（我踩过一次）。
FIRST=$(head -1 "reports/${TAG}_runs.txt")
bash /root/autodl-tmp/srv_check_fusion_evidence.sh "reports/analysis/${FIRST}" || true

echo
echo "=== 5) 融合分支放行了几条（只在开臂有意义）==="
echo "  含 fused_semantic 的候选行数: $(grep -rho 'fused_semantic' "reports/analysis/${FIRST}" 2>/dev/null | wc -l)"
echo
echo "=== 6) 磁盘 ==="
df -h /root/autodl-tmp | tail -1
echo "KBSELF_ARM_DONE tag=${TAG}"
