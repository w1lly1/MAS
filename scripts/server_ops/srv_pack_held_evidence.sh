#!/bin/bash
# 把 held 两批的**评测所需证据**打成一个小包，便于拉回本地归档（服务器随时可能被释放）。
#
# 装什么：每个 run 的 `run_summary.json` + `second_pass/consolidated/*_r2.json`（**精简版**，
# 工具真正读的就是它），以及 `second_pass_debug.log`（用于排查"为什么这条被拒"）。
# **不装**原始 `.gz` 留档（单个 60~500MB，两批合起来十几 G）——本地要重算指标用精简版足够；
# 真需要原文时再从服务器取（脚本会打印原文总大小，便于你决定要不要一起拉）。
#
# 用法: bash /root/autodl-tmp/srv_pack_held_evidence.sh
set -u
cd /root/autodl-tmp/MAS/reports/analysis || exit 1

OUT=/root/autodl-tmp/held_evidence_20261002.tgz
LIST=/tmp/_held_runs.txt
cat /root/autodl-tmp/held_fp4_runs.txt /root/autodl-tmp/held_overlap15_runs.txt 2>/dev/null > "$LIST"

echo "要打包的 run: $(wc -l < "$LIST") 个"
PATHS=/tmp/_held_paths.txt
: > "$PATHS"
while IFS=/ read -r cve run; do
  [ -n "${cve:-}" ] || continue
  [ -d "$cve/$run" ] && echo "$cve/$run" >> "$PATHS"
done < "$LIST"
echo "存在的目录: $(wc -l < "$PATHS") 个"

echo "打包（只含 run_summary + 精简 r2 + debug 日志）"
tar czf "$OUT" \
  --exclude='*.gz' \
  -T <(while read -r d; do
        echo "$d/run_summary.json"
        ls "$d"/second_pass/consolidated/*_r2.json 2>/dev/null
        echo "$d/second_pass_debug.log"
      done < "$PATHS" | while read -r f; do [ -e "$f" ] && echo "$f"; done)
echo "包大小: $(du -sh "$OUT" | cut -f1)  条目数: $(tar tzf "$OUT" | wc -l)"

echo
echo "（参考）这些 run 的**原始 .gz 留档**总大小："
while read -r d; do find "$d" -name '*.gz' -printf '%s\n' 2>/dev/null; done < "$PATHS" \
  | awk '{s+=$1} END {printf "  %.0f MB\n", s/1048576}'
echo "PACK_OK"
