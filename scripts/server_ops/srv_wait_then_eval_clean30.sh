#!/bin/bash
# 等 B6 批次跑完（按"批次进程是否还在"判断，最多等 60 分钟），然后自动做收尾评测。
# 单独写成一步的原因：批次是 nohup 起的，巡检脚本退出不代表批次结束。
set -u
for i in $(seq 1 60); do
  if ! pgrep -f "mas.py batch" >/dev/null 2>&1; then
    echo "批次进程已结束（等了约 $((i-1)) 分钟）"
    break
  fi
  sleep 60
done
date +%H:%M:%S
tail -5 /root/autodl-tmp/batch_held_clean30.log | cut -c1-160
echo
bash /root/autodl-tmp/srv_eval_clean30.sh
