#!/bin/bash
# 某臂的**逐样本完成状态**（只读）。用法: bash srv_arm_samples_state.sh <起始时间 ISO>
# 例: bash srv_arm_samples_state.sh 2026-10-03T20:25
set -u
SINCE="${1:?用法: srv_arm_samples_state.sh <起始ISO>}"
cd /root/autodl-tmp/MAS || exit 1

ok=0; run=0
echo "--- 已完成（有 run_summary.json）---"
while IFS= read -r d; do
  cve=$(echo "$d" | cut -d/ -f3)
  if [ -f "$d/run_summary.json" ]; then
    echo "  OK   $cve   $(basename "$d")"
    ok=$((ok+1))
  else
    echo "  跑着 $cve   $(basename "$d")   目录建于 $(date -r "$d" '+%H:%M')"
    run=$((run+1))
  fi
done < <(find reports/analysis -maxdepth 2 -mindepth 2 -type d -newermt "$SINCE" | sort)

echo
echo "已完成 $ok 个；未完成 $run 个；批次进程 $(pgrep -f 'venv/bin/python mas[.]py batch' | wc -l) 个"
