#!/bin/bash
# 把一个臂的**必要数据**打包（日志 + 逐项 CSV + run 列表 + 每个 run 的 r2/run_summary/debug）。
# 不打包 fullLayer 等大块产物 —— 评测只需要 small files，这样每臂几 MB，能直接拉回本地。
#
# 用法: bash _srv_pack_arm.sh <arm_name> <log1> [log2 ...]
set -u
cd /root/autodl-tmp/MAS || exit 1
NAME="${1:?用法: _srv_pack_arm.sh <arm_name> <log...>}"
shift
LOGS="$@"

STAGE="reports/arm_artifacts/$NAME"
rm -rf "$STAGE"; mkdir -p "$STAGE"

# 1) 日志
for lg in $LOGS; do
  [ -f "$lg" ] && cp -f "$lg" "$STAGE/" || echo "  (缺日志 $lg)"
done
# 2) 逐项 CSV（每个批次会覆盖，所以必须在批结束后立刻快照）
[ -f reports/batch_summary.csv ] && cp -f reports/batch_summary.csv "$STAGE/batch_summary.csv"

# 3) run 列表（补跑覆盖失败那次）
./venv/bin/python utils/experiments/make_run_list.py --out "$STAGE/runs.txt" --logs $LOGS > "$STAGE/runs.log" 2>&1 || true
[ -f "$STAGE/runs.txt" ] || : > "$STAGE/runs.txt"
echo "  run 列表条数: $(wc -l < "$STAGE/runs.txt")"

# 4) 每个 run 的小文件
n=0
while IFS= read -r cr; do
  [ -z "$cr" ] && continue
  cve="${cr%%/*}"; run="${cr##*/}"
  dest="$STAGE/runs/$cve/$run"
  mkdir -p "$dest"
  cp -f "reports/analysis/$cve/$run/run_summary.json" "$dest/" 2>/dev/null || true
  cp -f "reports/analysis/$cve/$run/second_pass/consolidated/"*.json "$dest/" 2>/dev/null || true
  find "reports/analysis/$cve/$run" -name '*.jsonl' -size -20M -exec cp -f {} "$dest/" \; 2>/dev/null || true
  n=$((n+1))
done < "$STAGE/runs.txt"
echo "  收集 run 目录: $n 个"

OUT="/root/autodl-tmp/artifacts_$NAME.tgz"
tar czf "$OUT" -C reports/arm_artifacts "$NAME"
echo "  打包: $OUT  $(du -h "$OUT" | cut -f1)"
