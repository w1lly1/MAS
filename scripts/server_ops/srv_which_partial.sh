#!/bin/bash
# 查某一批里到底哪些样本没跑完（partial），以及它们缺什么。
# 用法: bash srv_which_partial.sh held_fp4
set -u
NAME="${1:?用法: srv_which_partial.sh <批次名>}"
LIST="/root/autodl-tmp/${NAME}_runs.txt"
LOG="/root/autodl-tmp/${NAME}_log.txt"
ROOT=/root/autodl-tmp/MAS

echo "===== 1) run 清单（$LIST）====="
cat "$LIST" 2>/dev/null || echo "（没有清单）"

echo
echo "===== 2) 逐个 run 的完成情况 ====="
while IFS=/ read -r cve run; do
  [ -n "${cve:-}" ] || continue
  d="$ROOT/reports/analysis/$cve/$run"
  if [ -f "$d/run_summary.json" ]; then
    echo "  ✅ 完成     $cve/$run"
  elif [ -d "$d/agents" ]; then
    echo "  ❌ **未完成** $cve/$run（有 agents/ 但没有 run_summary.json）"
  else
    echo "  ?  找不到产物 $cve/$run"
  fi
done < "$LIST"

echo
echo "===== 3) 批处理日志里缺类的诊断（R1 修复后会打印"还差哪几类"）====="
grep -E "还差|缺|partial|未确认完成" "$LOG" | tail -12

echo
echo "===== 4) 批次收尾统计 ====="
grep -E "批量分析结束" "$LOG" | tail -2
