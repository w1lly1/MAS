#!/bin/bash
# 清磁盘：按 **run 目录**精确删除老的运行产物，给新批次腾地方。
#
# ## 为什么重写（旧版有坑）
# 旧版是 `ls -t analysis | tail -n +26 | rm -rf`：它按 **mtime 删整个 CVE 目录**，
# 保留"最近 25 个"。问题是 ① 目录 mtime 会随新 run 写入而变，排序不可靠；
# ② 删的是 CVE 级目录，会把**同一 CVE 下所有历史 run（含要用来对照的那次）一起删掉**。
# 那次如果真按 --apply 跑，会静默毁掉对照实验的证据。
#
# ## 现在的口径
#   · 单位是 **run 目录**（`reports/analysis/<CVE>/<run_id>/`），不是 CVE 目录；
#   · **keep 清单里的 run 永不删**（例如正在评测的那几批：t2_runs.txt / held*.txt）；
#   · 只删"最后修改时间早于 N 分钟"的（默认 60），避免动到正在写的 run；
#   · 默认**干跑**，打印将删的清单与可释放空间；`--apply` 才真删。
#
# 用法:
#   bash script.sh                       # 干跑，默认保留最近 60 分钟
#   bash script.sh --keep /root/autodl-tmp/MAS/reports/t2_runs.txt --older-than-min 30
#   bash script.sh --keep ... --apply
#   ↑ keep 路径写成绝对路径最稳（脚本会 cd 到 reports/，但两种基准都会试；
#     **指定的 keep 文件找不到时直接报错退出**，不再静默忽略）
set -u
cd /root/autodl-tmp/MAS/reports || exit 1

APPLY=0
OLDER=60
KEEP_FILES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --older-than-min) OLDER="$2"; shift 2 ;;
    --keep) KEEP_FILES+=("$2"); shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

echo "reports 总占用: $(du -sh . 2>/dev/null | cut -f1)"
echo "磁盘: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4" 可用 ("$5" 已用)"}')"

# 1) 收集 keep 清单（每行 CVE/run_id）
#    ⚠️ 这里曾经有一个**会毁证据的静默失效**：脚本开头 `cd .../reports`，而用法示例传的是
#    `reports/xxx_runs.txt`（相对仓库根）⇒ `[ -f ]` 判否 ⇒ **该 keep 文件被静默跳过**，
#    打印"keep 清单条目: 0"却继续干跑/真删 —— 于是"我指定了要保留"变成了"全都能删"。
#    2026-10-04 干跑时就差点把**有效基线 baseline_fix 一起列进删除清单**（350 个 run、15.9 GB）。
#    现在：① 路径按"当前目录 / 仓库根"两种基准各试一次；② 指定的 keep 文件**找不到就报错退出**；
#    ③ 打印每个 keep 文件贡献了几条，并报告"对不上任何真实 run 目录"的条目（写错名字能被看见）。
KEEP=/tmp/_keep_runs.txt
: > "$KEEP"
REPO_ROOT="$(cd .. && pwd)"
for f in "${KEEP_FILES[@]:-}"; do
  [ -z "$f" ] && continue
  resolved=""
  for cand in "$f" "$REPO_ROOT/$f" "../$f"; do
    if [ -f "$cand" ]; then resolved="$cand"; break; fi
  done
  if [ -z "$resolved" ]; then
    echo "❌ keep 文件不存在: $f（拒绝继续 —— 静默忽略 keep 会导致误删证据）" >&2
    exit 1
  fi
  before=$(wc -l < "$KEEP")
  sed 's#^/*##' "$resolved" | tr -d ' \r' | grep -v '^$' >> "$KEEP"
  after=$(wc -l < "$KEEP")
  echo "  keep 文件 $resolved → 贡献 $((after-before)) 条"
done
sort -u "$KEEP" -o "$KEEP"
echo "keep 清单条目: $(wc -l < "$KEEP")（来自 ${KEEP_FILES[*]:-无}）"
if [ -s "$KEEP" ]; then
  # 校验 keep 条目是否都能对上真实 run 目录（写错/已被删的名字要看得见）
  MISS=0
  while IFS= read -r key; do
    [ -d "analysis/$key" ] || { MISS=$((MISS+1)); [ "$MISS" -le 5 ] && echo "  ⚠️ keep 条目没有对应目录: $key"; }
  done < "$KEEP"
  echo "  keep 条目中找不到对应目录的: $MISS（>0 说明清单有错字或证据已被删）"
fi

# 2) 找出候选：run 目录里带 run_summary.json 或 agents/ 的，且 mtime 早于 OLDER 分钟
NOW=$(date +%s)
DEL=/tmp/_to_delete_runs.txt
: > "$DEL"
TOTAL_MB=0
while IFS= read -r cve_dir; do
  cve=$(basename "$cve_dir")
  while IFS= read -r run_dir; do
    run=$(basename "$run_dir")
    key="$cve/$run"
    if grep -qxF "$key" "$KEEP" 2>/dev/null; then continue; fi
    # 只在"像 run 的目录"里动手
    if [ ! -f "$run_dir/run_summary.json" ] && [ ! -d "$run_dir/agents" ]; then continue; fi
    MT=$(stat -c %Y "$run_dir")
    AGE_MIN=$(( (NOW - MT) / 60 ))
    if [ "$AGE_MIN" -lt "$OLDER" ]; then continue; fi
    MB=$(du -sm "$run_dir" 2>/dev/null | cut -f1)
    echo "$MB $run_dir" >> "$DEL"
    TOTAL_MB=$((TOTAL_MB + MB))
  done < <(find "$cve_dir" -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
done < <(find analysis -maxdepth 1 -mindepth 1 -type d 2>/dev/null)

N=$(wc -l < "$DEL")
echo
echo "候选删除（run 目录）: $N 个，预计释放 ${TOTAL_MB} MB"
echo "最大的 8 个："
sort -rn "$DEL" 2>/dev/null | head -8 | awk '{printf "   %6d MB  %s\n", $1, $2}'

if [ "$APPLY" = "1" ]; then
  while read -r mb path; do rm -rf "$path"; done < "$DEL"
  echo "已删除 $N 个 run 目录。"
  echo "reports 现在: $(du -sh . 2>/dev/null | cut -f1)"
  df -h /root/autodl-tmp | tail -1
else
  echo "（干跑；确认清单没问题再加 --apply）"
fi
