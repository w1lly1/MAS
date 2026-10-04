#!/bin/bash
# 收工前的停机准备：优雅停 Weaviate（不用 kill -9），并打一份"停机快照"。
#
# 为什么强调优雅停：上次为了腾磁盘用了 kill -9，下次启动日志出现
# hnsw_load_commit_log_corruption（WAL 突然结束）。实测数据没丢，但那个告警会让人怀疑数据，
# 而且每次都要额外花时间用数字证明无损 —— 能避免就避免。
set -u
cd /root/autodl-tmp/MAS || exit 1

echo "=== 1) 优雅停止 Weaviate（SIGTERM，等它把 WAL/raft 冲干净）==="
pkill -TERM -f 'weaviate --host' && echo "  已发送 SIGTERM" || echo "  Weaviate 本来就没在跑"
for i in $(seq 1 20); do
  if ! pgrep -f 'weaviate --host' >/dev/null 2>&1; then echo "  已在 ${i}0 秒内退出"; break; fi
  sleep 10
done
if pgrep -f 'weaviate --host' >/dev/null 2>&1; then
  echo "  ⚠️ 还活着，等它自己退（不升级成 -9，宁可多等）"
else
  echo "  ✅ 已退出"
fi
echo "  日志末尾:"
tail -3 /root/autodl-tmp/weaviate.log 2>/dev/null | cut -c1-160

echo
echo "=== 2) 停机快照（写进 reports/shutdown_snapshot_$(date +%Y%m%d_%H%M).txt）==="
SNAP="reports/shutdown_snapshot_$(date +%Y%m%d_%H%M).txt"
{
  echo "停机快照 $(date '+%Y-%m-%d %H:%M:%S')"
  echo "容器名: $(hostname)"
  echo "代码: $(git log --oneline -1)"
  echo "未跟踪改动: $(git status --porcelain | wc -l) 项"
  echo "配置 sha256[:16]: $(sha256sum infrastructure/config/ai_agent_config.json | cut -c1-16)"
  echo "知识库 mas.db: $(sha256sum infrastructure/database/mas.db | cut -c1-16)"
  echo "磁盘: $(df -h /root/autodl-tmp | tail -1)"
  echo "Weaviate 进程: $(pgrep -f 'weaviate --host' | wc -l) 个（应为 0）"
  echo "批次进程: $(pgrep -f 'venv/bin/python mas[.]py batch' | wc -l) 个（应为 0）"
  echo "本批（干净层30）评测: $(cat reports/held_clean30_eval.txt 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
} | tee "$SNAP"
echo
echo "SHUTDOWN_PREP_DONE"
