#!/bin/bash
# 清磁盘：Weaviate 在 90% 会转只读，3 臂 × 30 样本会产生大量 run 目录，先腾地方。
# 策略：保留**最近 25 个** run 目录，其余删除；先干跑打印，`--apply` 才真删。
set -u
cd /root/autodl-tmp/MAS/reports || exit 1
APPLY="${1:-dry}"

echo "reports 总占用: $(du -sh . 2>/dev/null | cut -f1)"
echo "analysis 目录数: $(ls analysis 2>/dev/null | wc -l)"
echo "最大的 8 个 run 目录:"
du -sm analysis/* 2>/dev/null | sort -rn | head -8 | awk '{printf "   %6d MB  %s\n", $1, $2}'

ls -t analysis | tail -n +26 > /tmp/to_delete.txt
echo
echo "将删除 $(wc -l < /tmp/to_delete.txt) 个较老的 run 目录（保留最近 25 个）"
SIZE=$(while IFS= read -r d; do du -sm "analysis/$d" 2>/dev/null; done < /tmp/to_delete.txt | awk '{s+=$1} END {print s+0}')
echo "预计释放: ${SIZE} MB"

if [ "$APPLY" = "--apply" ]; then
  while IFS= read -r d; do rm -rf "analysis/$d"; done < /tmp/to_delete.txt
  echo "已删除。"
  echo "reports 现在: $(du -sh . 2>/dev/null | cut -f1)"
  df -h /root/autodl-tmp | tail -1
else
  echo "（干跑；加 --apply 才真删）"
fi
