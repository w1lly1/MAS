#!/usr/bin/env bash
# 校验"融合判定重算器"在**活 embedder** 基线上是否与生产同源。
#
# 判据：取刚跑完的、开了融合的那一臂，用重算器复算 term，与产物里记录的 `fusion_semantic_term`
#       逐条比（必须 ≤1e-6），并比放行判定条数。
#
# 用法: bash srv_validate_recompute.sh <批次日志> <CVE>
set -u
LOG="${1:?用法: srv_validate_recompute.sh <批次日志> <CVE>}"
CVE="${2:?缺 CVE}"
cd /root/autodl-tmp/MAS || exit 1
# ⚠️ **任何会做嵌入的脚本都必须显式给 HF_HOME**（非交互 shell 不读 ~/.bashrc）——
# 我自己在这里漏过一次，重算器当场退回兜底向量（14 条不一致，最大偏差 0.727）。
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
echo "HF_HOME=$HF_HOME"

echo "=== 1) 等批次结束 ==="
for i in $(seq 1 45); do
  if ! pgrep -f 'venv/bin/python mas[.]py batch' >/dev/null 2>&1; then echo "  已结束（约 $((i*20)) 秒）"; break; fi
  sleep 20
done

echo
echo "=== 2) 日志里 embedder 是否正常 ==="
echo "  '加载失败' 次数: $(grep -c 加载失败 "$LOG" 2>/dev/null || echo 0)（期望 0）"

echo
echo "=== 3) 定位 run 并做重算校验 ==="
D=$(ls -dt "reports/analysis/$CVE"/*/ 2>/dev/null | head -1)
[ -n "$D" ] || { echo "找不到 run"; exit 2; }
printf '%s\n' "${D#reports/analysis/}" | sed 's:/$::' > /tmp/one_run_list.txt
echo "  run: $D"
venv/bin/python -X utf8 utils/experiments/recompute_fusion_decision.py \
  --runs /tmp/one_run_list.txt --lam 1.5 --theta 0.7 --validate \
  --db infrastructure/database/mas.db \
  --dump reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl \
  --json-out reports/recompute_validate.json 2>&1 | tail -20
echo "VALIDATE_RECOMPUTE_DONE"
