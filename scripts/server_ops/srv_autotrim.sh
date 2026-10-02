#!/bin/bash
# 每批跑完**自动瘦身**：只要某批的 run 清单出现，就对那批的证据做 trim（无损：原文留 .gz）。
#
# 为什么要自动：held 样本的证据单个可到 227MB，几十个样本就能把磁盘打满；
# 而"批次边界"需要有人盯着 —— 之前就是因为没人盯，编排脚本崩了 26 分钟都没发现。
# 这里用 `run 清单文件存在` 当边界信号（编排脚本每跑完一批就会生成它）。
#
# 用法: nohup bash /root/autodl-tmp/srv_autotrim.sh > /root/autodl-tmp/autotrim.log 2>&1 &
set -u
cd /root/autodl-tmp/MAS || exit 1

say () { echo "[$(date +%H:%M:%S)] $*"; }
BATCHES="held_fp4 held_overlap15 held_kb30"

say "开始监视批次边界（每 2 分钟看一次）"
for i in $(seq 1 300); do
  for n in $BATCHES; do
    f="/root/autodl-tmp/${n}_runs.txt"
    mark="/root/autodl-tmp/.trimmed_${n}"
    if [ -f "$f" ] && [ ! -f "$mark" ]; then
      say "$n 的 run 清单出现 → 开始瘦身"
      nice -n 19 ionice -c3 ./venv/bin/python -X utf8 utils/experiments/trim_run_evidence.py \
        --runs "$f" --apply > "/root/autodl-tmp/trim_${n}.log" 2>&1
      rc=$?
      tail -2 "/root/autodl-tmp/trim_${n}.log"
      say "$n 瘦身结束 rc=$rc，磁盘可用 $(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')MB"
      touch "$mark"
    fi
  done
  if grep -q CHAIN_HELD_DONE /root/autodl-tmp/held_chain.log 2>/dev/null; then
    say "编排脚本已报完成，监视结束"
    break
  fi
  sleep 120
done
say "AUTOTRIM_DONE"
