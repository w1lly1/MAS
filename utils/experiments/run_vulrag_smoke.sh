#!/bin/bash
# Vul-RAG 风格基线冒烟测试（8 样本），输出到日志。
set -u
cd /root/autodl-tmp/MAS || exit 1
source venv/bin/activate
export HF_HOME=/root/autodl-tmp/hf-cache
nohup python -u utils/experiments/vulrag_baseline.py --limit 8 --device gpu \
  > /root/autodl-tmp/vulrag_smoke.log 2>&1 </dev/null &
echo "smoke pid $!"
