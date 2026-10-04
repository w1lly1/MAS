#!/bin/bash
# **先瘦身（抽决策摘要）再删**：把旧臂的 run 产物压成可留档的决策摘要，
# 与已归档结论做正向对照，只有对得上才允许删原目录。
#
# 为什么这么绕：400 样本约需 8 GB，盘只剩 5.7 GB。删之前必须证明"删掉的是可复现的"，
# 否则就是把证据变成"当时好像是这样"（《03》坑 41/46 是同一类错误）。
#
# 用法：
#   bash srv_slim_arms.sh                 # 只抽摘要 + 打印对照（不动任何东西）
#   bash srv_slim_arms.sh --apply-delete  # 对照通过后才真删（死 embedder 臂）
set -u
cd /root/autodl-tmp/MAS || exit 1
APPLY=0
[ "${1:-}" = "--apply-delete" ] && APPLY=1

DB=infrastructure/database/mas.db
# 死 embedder（2026-10-03 及以前）的臂：向量侧结论已作废 ⇒ 属于"可删"对象
DEAD_TAGS="ab_g0 ab_g1 ab_smoke_kb8 ab_smoke_v2 held_clean30 held_clean30_clean30_off held_clean30_clean30_on clean30_lam3 kbself_on kbself_on3 kbself_fix overlap15_lam3"
# 活 embedder 的臂 / 正在跑的臂：**有效证据，永不删**
KEEP_TAGS="baseline_fix clean30_live overlap15_live"

echo "===== 1) 抽决策摘要 ====="
for tag in $KEEP_TAGS $DEAD_TAGS; do
  f="reports/${tag}_runs.txt"
  [ -f "$f" ] || { echo "  跳过 $tag（无 run 清单）"; continue; }
  venv/bin/python -X utf8 utils/experiments/extract_arm_summary.py \
    --tag "$tag" --runs "$f" --db "$DB" 2>&1 | sed 's/^/  /'
done

echo
echo "===== 2) 摘要 vs 已归档结论（人工核对用）====="
for tag in $KEEP_TAGS $DEAD_TAGS; do
  agg="reports/arm_summaries/${tag}_aggregate.json"
  [ -f "$agg" ] || continue
  echo "--- $tag"
  venv/bin/python -X utf8 - "$agg" <<'PY'
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))
print("    摘要: 样本 %d / own 放行 %d / 放行总 %d / 非 own %d"
      % (a["samples"], a["own_admitted"], a["total_admitted"], a["non_own_admitted"]))
PY
  for arch in "reports/${tag}_eval.txt" "reports/${tag}_compare.txt"; do
    [ -f "$arch" ] || continue
    echo "    归档 $arch:"
    grep -E '放行总数|自己条目|own 放行|放行来源通道|总放行' "$arch" 2>/dev/null | head -4 | sed 's/^/      /'
  done
done

echo
echo "===== 3) 删除 ====="
if [ "$APPLY" != "1" ]; then
  echo "（未加 --apply-delete，只抽摘要不删。确认第 2 步逐臂对得上再执行）"
  df -h /root/autodl-tmp | tail -1
  echo "SLIM_ARMS_DRYRUN_DONE"
  exit 0
fi

before_mb=$(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')
for tag in $DEAD_TAGS; do
  f="reports/${tag}_runs.txt"
  [ -f "$f" ] || continue
  agg="reports/arm_summaries/${tag}_aggregate.json"
  if [ ! -f "$agg" ]; then
    echo "!! $tag 没有摘要，拒绝删它的 run 目录"; continue
  fi
  n=0
  while IFS= read -r line; do
    line=$(echo "$line" | tr -d ' \r')
    [ -z "$line" ] && continue
    d="reports/analysis/$line"
    if [ -d "$d" ]; then rm -rf "$d"; n=$((n+1)); fi
  done < "$f"
  echo "  删除 $tag: $n 个 run 目录（摘要已留档）"
done
after_mb=$(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')
echo "  可用空间: ${before_mb}MB -> ${after_mb}MB（释放 $((after_mb-before_mb))MB）"
du -sh reports reports/arm_summaries 2>/dev/null
df -h /root/autodl-tmp | tail -1
echo "SLIM_ARMS_APPLY_DONE"
