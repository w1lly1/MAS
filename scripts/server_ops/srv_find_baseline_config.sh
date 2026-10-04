#!/bin/bash
# 找出：哪个实验配置的样本集与 baseline_fix 臂**完全一致**（λ 真臂必须用同一批样本，
# 否则"与 fusion OFF 基线对比"不成立）。
set -u
cd /root/autodl-tmp/MAS || exit 1

cut -d/ -f1 reports/baseline_fix_runs.txt | tr -d ' \r' | sort -u > /tmp/_base_cves.txt
echo "基线 CVE 数: $(wc -l < /tmp/_base_cves.txt)"
echo

for f in utils/experiments/*.json; do
  grep -oE 'CVE-[0-9]{4}-[0-9]+' "$f" 2>/dev/null | sort -u > /tmp/_c.txt || true
  n=$(wc -l < /tmp/_c.txt)
  [ "$n" -eq 0 ] && continue
  if diff -q /tmp/_c.txt /tmp/_base_cves.txt >/dev/null 2>&1; then
    verdict="✅ 与基线一致"
  else
    verdict="✗ 不一致"
  fi
  # 顺带把该配置的 items 数打出来
  items=$(grep -c 'target_dir' "$f" 2>/dev/null || echo 0)
  echo "$(printf '%-58s' "$f") 唯一CVE $n  items≈$items  $verdict"
done
echo
echo "FIND_CONFIG_DONE"
