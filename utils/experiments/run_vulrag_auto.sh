#!/bin/bash
# Vul-RAG 风格基线：先冒烟(8样本)，通过后自动启动全量(400样本)。
set -u
cd /root/autodl-tmp/MAS || exit 1
source venv/bin/activate
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== 冒烟测试 (8 样本) 开始 $(date) ==="
python -u utils/experiments/vulrag_baseline.py --limit 8 --device gpu \
  > /root/autodl-tmp/vulrag_smoke.log 2>&1
SMOKE_EXIT=$?

if [ "$SMOKE_EXIT" -ne 0 ] || grep -q "Traceback" /root/autodl-tmp/vulrag_smoke.log; then
  echo "SMOKE_FAIL" > /root/autodl-tmp/vulrag_status.txt
  echo "冒烟失败 exit=$SMOKE_EXIT，日志见 vulrag_smoke.log"
  tail -25 /root/autodl-tmp/vulrag_smoke.log
  exit 1
fi
if ! grep -q "kb 召回" /root/autodl-tmp/vulrag_smoke.log; then
  echo "SMOKE_INCOMPLETE" > /root/autodl-tmp/vulrag_status.txt
  echo "冒烟未产出汇总，日志见 vulrag_smoke.log"
  tail -25 /root/autodl-tmp/vulrag_smoke.log
  exit 1
fi

echo "=== 冒烟通过，自动启动全量 (400 样本) $(date) ==="
tail -6 /root/autodl-tmp/vulrag_smoke.log
nohup python -u utils/experiments/vulrag_baseline.py --limit 400 --device gpu \
  > /root/autodl-tmp/vulrag_full.log 2>&1 </dev/null &
FULL_PID=$!
echo "RUNNING:$FULL_PID" > /root/autodl-tmp/vulrag_status.txt
echo "全量已启动 pid=$FULL_PID，日志 vulrag_full.log"
