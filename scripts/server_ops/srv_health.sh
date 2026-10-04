#!/bin/bash
# 跑批期间的**巡检**（只读）：进度 + 磁盘 + 内存 + 负载 + GPU + 服务 + 日志里的报错。
# 用法: bash srv_health.sh <批次日志> [起始时间 ISO，如 2026-10-03T16:49] [期望样本数]
set -u
LOG="${1:?用法: srv_health.sh <批次日志> [起始ISO] [期望数]}"
SINCE="${2:-}"
EXPECT="${3:-30}"
cd /root/autodl-tmp/MAS || exit 1

echo "=== 时间 / 进度 ==="
echo "现在          : $(date '+%H:%M:%S')${SINCE:+（起始 $SINCE）}"
if [ -n "$SINCE" ]; then
  echo "本臂 run 目录 : $(find reports/analysis -maxdepth 2 -mindepth 2 -type d -newermt "$SINCE" | wc -l) 个"
  echo "本臂覆盖 CVE  : $(find reports/analysis -maxdepth 1 -mindepth 1 -type d -newermt "$SINCE" | wc -l) / $EXPECT"
fi
echo "批次进程      : $(pgrep -f 'venv/bin/python mas[.]py batch' | wc -l) 个（0 = 已结束）"
echo "已派发(Run ID): $(grep -c 'Run ID' "$LOG" 2>/dev/null || echo 0)"
echo "记为 partial  : $(grep -c '（记为 partial' "$LOG" 2>/dev/null || echo 0)"

echo
echo "=== 磁盘 ==="
df -h /root/autodl-tmp | tail -1
echo "  本臂产物大小: $(du -sh --exclude='*.gz' reports/analysis 2>/dev/null | cut -f1)"
echo "  剩余可用    : $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')（低于 2G 要立刻处理）"

echo
echo "=== 内存 / 负载 ==="
free -g | head -2
echo "  load: $(cat /proc/loadavg)"
echo "  核数: $(nproc)"

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

echo
echo "=== 服务 ==="
echo "  Weaviate 进程: $(pgrep -f 'weaviate --host' | wc -l) 个"
echo "  Weaviate 就绪: HTTP $(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8080/v1/.well-known/ready 2>/dev/null)"

echo
echo "=== 日志里的异常（近 200 行内计数）==="
tail -200 "$LOG" 2>/dev/null | grep -ciE 'traceback|exception|❌|error -' || echo 0
echo "  最近 2 行:"
tail -2 "$LOG" | cut -c1-140
echo "HEALTH_DONE"
