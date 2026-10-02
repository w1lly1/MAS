#!/bin/bash
# 查清"reports/analysis 里那些 run 目录"与"已压缩归档"的关系，判断哪些可以安全删除。
#
# 背景：held 批次正在跑，磁盘按约 0.8G/样本下降（r2 证据单个可达 60-70MB）。
# 中途爆盘会让 Weaviate 转只读、后续样本失败 —— 比"先删冗余副本"严重得多。
# 但删之前必须证明：**这些目录在归档里有逐字节相同的副本**。
set -u
cd /root/autodl-tmp/MAS/reports || exit 1

TGZ=/root/autodl-tmp/arm_artifacts_full_20261002.tgz
KEEP_LIST=/root/autodl-tmp/t2_runs.txt

echo "===== 0) 现状 ====="
df -h /root/autodl-tmp | tail -1
du -sh analysis 2>/dev/null

echo
echo "===== 1) 分析目录里的 run 目录总数 ====="
find analysis -maxdepth 3 -name run_summary.json | wc -l

echo
echo "===== 2) 三臂清单里的 run 在 analysis 里还有多少、占多大 ====="
cat /root/autodl-tmp/arm1_runs.txt /root/autodl-tmp/arm2_runs.txt /root/autodl-tmp/arm3_runs.txt 2>/dev/null > /tmp/_arms.txt
FOUND=0; MISSING=0; TOTAL_MB=0
while IFS= read -r line; do
  [ -n "$line" ] || continue
  d="analysis/$line"
  if [ -d "$d" ]; then
    FOUND=$((FOUND+1))
    mb=$(du -sm "$d" 2>/dev/null | cut -f1)
    TOTAL_MB=$((TOTAL_MB + ${mb:-0}))
  else
    MISSING=$((MISSING+1))
  fi
done < /tmp/_arms.txt
echo "  存在 $FOUND 个（合计 ${TOTAL_MB}MB），不存在 $MISSING 个"

echo
echo "===== 3) 抽一个 run 做逐字节比对（analysis vs 归档）====="
SAMPLE_LINE=$(head -1 /tmp/_arms.txt)
CVE=$(echo "$SAMPLE_LINE" | cut -d/ -f1)
RUN=$(echo "$SAMPLE_LINE" | cut -d/ -f2)
LIVE="analysis/$CVE/$RUN"
echo "  样本: $CVE/$RUN"
if [ -d "$LIVE" ]; then
  LIVE_FILE=$(ls "$LIVE"/second_pass/consolidated/*_r2.json 2>/dev/null | head -1)
  echo "  本地 r2: $(basename "${LIVE_FILE:-无}")  大小 $(stat -c%s "${LIVE_FILE:-/dev/null}" 2>/dev/null)"
  echo "  归档内同 run 的文件："
  tar tzvf "$TGZ" 2>/dev/null | grep "$RUN" | head -4
else
  echo "  analysis 里没有这个 run（可能已被清过）"
fi

echo
echo "===== 4) 可安全释放的候选（不在 T2/held 任何清单里的老 run）====="
cat "$KEEP_LIST" /root/autodl-tmp/held_fp4_runs.txt /root/autodl-tmp/held_overlap15_runs.txt /root/autodl-tmp/held_kb30_runs.txt 2>/dev/null > /tmp/_keep.txt
CNT=0; MB=0
while IFS= read -r cve_dir; do
  cve=$(basename "$cve_dir")
  while IFS= read -r run_dir; do
    key="$cve/$(basename "$run_dir")"
    grep -qxF "$key" /tmp/_keep.txt 2>/dev/null && continue
    grep -qxF "$key" /tmp/_arms.txt 2>/dev/null && continue
    CNT=$((CNT+1))
    mb=$(du -sm "$run_dir" 2>/dev/null | cut -f1)
    MB=$((MB + ${mb:-0}))
  done < <(find "$cve_dir" -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
done < <(find analysis -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
echo "  非三臂、非 T2/held 的老 run：$CNT 个，合计 ${MB}MB"
