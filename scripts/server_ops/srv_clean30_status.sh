#!/bin/bash
# B6（干净层 30 样本）批次进度快查：进程在不在、跑了多少、还有多少。
set -u
cd /root/autodl-tmp/MAS || exit 1
LOG=/root/autodl-tmp/batch_held_clean30.log
echo "现在        : $(date +%H:%M:%S)（批次起于 22:41）"
echo "批次进程数  : $(pgrep -f 'venv/bin/python mas[.]py batch' | wc -l)   （0 = 已结束）"
echo "Run ID 行数 : $(grep -c 'Run ID' "$LOG")"
echo "新建 run 目录: $(find reports/analysis -maxdepth 2 -mindepth 2 -type d -newermt '2026-10-02 22:41' | wc -l)"
echo "覆盖 CVE 数 : $(find reports/analysis -maxdepth 1 -mindepth 1 -type d -newermt '2026-10-02 22:41' | wc -l) / 30"
echo "--- 末尾 3 行 ---"
tail -3 "$LOG" | cut -c1-150
