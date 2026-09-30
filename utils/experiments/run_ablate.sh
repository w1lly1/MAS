#!/bin/bash
# 离线视图消融：16 分片并行（每分片单线程 torch，充分利用 64 可用核）。
set -u
cd /root/autodl-tmp/MAS || exit 1
source venv/bin/activate
export HF_HOME=/root/autodl-tmp/hf-cache
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

SHARDS=16
# 清理旧分片结果，避免重复
rm -f reports/ablate_views_singlepass_results_shard*.jsonl
rm -f reports/ablate_views_evidence_shard*.jsonl
rm -f reports/ablate_views_sqlite_patterns_shard*.json

for i in $(seq 0 $((SHARDS - 1))); do
  nohup python -u utils/experiments/ablate_views_singlepass.py \
    --limit 200 --shard-id "$i" --shard-count "$SHARDS" \
    > "/root/autodl-tmp/ablate_shard$i.log" 2>&1 </dev/null &
  echo "launched shard $i (pid $!)"
done
echo "ALL_16_LAUNCHED"
