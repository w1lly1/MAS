#!/bin/bash
# 任意批次的进度快查（只读）。用法: bash srv_arm_progress.sh <批次日志> [起始时间 "YYYY-MM-DD HH:MM"]
set -u
LOG="${1:?用法: srv_arm_progress.sh <批次日志> [起始时间]}"
SINCE="${2:-}"
cd /root/autodl-tmp/MAS || exit 1

echo "现在        : $(date '+%H:%M:%S')"
echo "日志        : $LOG"
echo "已派发(Run ID 行数): $(grep -c 'Run ID' "$LOG" 2>/dev/null || echo 0)"
echo "批次进程数  : $(pgrep -f 'mas.py batch' | wc -l)   （0 = 已结束）"
if [ -n "$SINCE" ]; then
  echo "起始时间后新建的 run 目录: $(find reports/analysis -maxdepth 2 -mindepth 2 -type d -newermt "$SINCE" 2>/dev/null | wc -l)"
fi
echo "记为 partial: $(grep -c '（记为 partial' "$LOG" 2>/dev/null || echo 0)"
echo "磁盘        : $(df -h /root/autodl-tmp | tail -1)"
echo "--- 末尾 3 行 ---"
tail -3 "$LOG" 2>/dev/null | cut -c1-150
