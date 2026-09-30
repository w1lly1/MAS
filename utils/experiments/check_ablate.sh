#!/bin/bash
# 查看 16 分片进度：每分片最近一行 + 完成计数。
SHARDS=16
done_count=0
for i in $(seq 0 $((SHARDS - 1))); do
  log="/root/autodl-tmp/ablate_shard$i.log"
  if [ -f "$log" ]; then
    last=$(tail -1 "$log" 2>/dev/null | cut -c1-140)
    echo "shard$i: $last"
    if grep -q "本进程完成\|汇总" "$log" 2>/dev/null; then
      done_count=$((done_count + 1))
    fi
  else
    echo "shard$i: (无日志)"
  fi
done
echo "----- 已完成分片: $done_count/$SHARDS -----"
echo "----- 已出结果 CVE 数 -----"
wc -l /root/autodl-tmp/MAS/reports/ablate_views_singlepass_results_shard*.jsonl 2>/dev/null | tail -1
