#!/bin/bash
# 某臂的完成度（按 CVE 计）：用法 bash srv_arm_done.sh <批次日志> <起始时间 ISO，如 2026-10-03T12:42> [期望样本数]
set -u
LOG="${1:?用法: srv_arm_done.sh <批次日志> <起始ISO> [期望数]}"
SINCE="${2:?缺起始时间}"
EXPECT="${3:-30}"
cd /root/autodl-tmp/MAS || exit 1

DONE=$(find reports/analysis -maxdepth 1 -mindepth 1 -type d -newermt "$SINCE" | wc -l)
RUNS=$(find reports/analysis -maxdepth 2 -mindepth 2 -type d -newermt "$SINCE" | wc -l)
echo "现在          : $(date '+%H:%M:%S')（起始 $SINCE）"
echo "本臂产出 run 目录: $RUNS 个"
echo "本臂覆盖 CVE   : $DONE / $EXPECT"
echo "批次进程      : $(pgrep -f 'venv/bin/python mas[.]py batch' | wc -l) 个（0 = 已结束）"
echo "已派发(Run ID) : $(grep -c 'Run ID' "$LOG" 2>/dev/null || echo 0)"
echo "记为 partial  : $(grep -c '（记为 partial' "$LOG" 2>/dev/null || echo 0)"
echo "磁盘          : $(df -h /root/autodl-tmp | tail -1)"
echo "末尾 2 行     :"
tail -2 "$LOG" | cut -c1-130
