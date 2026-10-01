#!/bin/bash
# 启动一个批次（nohup，立刻返回；长跑靠巡检脚本看进度）
# 用法: bash _srv_start_batch.sh <config> <logfile>
set -u
CFG="${1:?用法: _srv_start_batch.sh <config> <logfile>}"
LOG="${2:-/root/autodl-tmp/batch.log}"
cd /root/autodl-tmp/MAS || exit 1
: > "$LOG"
nohup ./venv/bin/python mas.py batch -c "$CFG" >> "$LOG" 2>&1 &
echo "started pid=$! log=$LOG"
echo "config=$CFG items=$(./venv/bin/python -c "import json;print(len(json.load(open('$CFG'))['items']))")"
