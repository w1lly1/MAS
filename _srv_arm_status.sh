#!/bin/bash
# Arm 进度巡检：逐项计数、完成/未确认/失败、最新一行、耗时、磁盘。
# 用法: bash _srv_arm_status.sh <logfile> [total]
set -u
LOG="${1:-/root/autodl-tmp/arm1_new_new.log}"
TOTAL="${2:-30}"
cd /root/autodl-tmp/MAS || exit 1

DONE=$(grep -cE '^  \[[0-9]+/[0-9]+\]' "$LOG" 2>/dev/null || echo 0)
OK=$(grep -c '✅ 完成' "$LOG" 2>/dev/null || echo 0)
# 只数**条目级**提示：批次汇总行也含"未确认完成"这个词（"...未确认完成 0..."），
# 直接 grep 它会多算 1 个（实测把最后一个样本误报成 partial）。
PARTIAL=$(grep -c '（记为 partial' "$LOG" 2>/dev/null || echo 0)
BAD=$(grep -cE '❌' "$LOG" 2>/dev/null || echo 0)

echo "=== 进度 ==="
echo "  已开始处理: $DONE / $TOTAL    完成: $OK    未确认完成: $PARTIAL    报错行: $BAD"
PID=$(pgrep -f 'mas.py batch' | head -1)
if [ -n "${PID:-}" ]; then
  echo "  进程: 运行中 pid=$PID  已运行 $(ps -o etime= -p "$PID" | tr -d ' ')"
else
  echo "  进程: 已结束"
fi
echo
echo "=== 逐项状态（最近的 8 行）==="
grep -E '^  \[[0-9]+/[0-9]+\]|✅ 完成|未确认完成|Run ID' "$LOG" 2>/dev/null | tail -8 | cut -c1-150
echo
echo "=== 日志最后 3 行（看当前在干什么）==="
tail -3 "$LOG" | cut -c1-160
echo
echo "=== 是否已有 partial / 真正报错 ==="
grep -nE '未确认完成（记为 partial|❌' "$LOG" 2>/dev/null | tail -5 || echo "  (无)"
echo
echo "=== 汇总行（跑完才有）==="
grep -E '批量分析结束' "$LOG" 2>/dev/null | tail -1 || echo "  (还没结束)"
echo
echo "=== 资源 ==="
df -h /root/autodl-tmp | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "reports/analysis 目录数: $(ls reports/analysis | wc -l)"
