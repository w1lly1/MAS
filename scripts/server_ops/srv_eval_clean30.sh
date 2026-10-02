#!/bin/bash
# B6（库外干净层 30 样本）跑完后的收尾：生成 run 清单 → 库外口径评测 → 完整性审计。
#
# 口径要点（见 utils/experiments/eval_held_runs.py 的文档串）：
#   库外样本在库里**没有自己的条目** ⇒ 每一条放行都是误报面，分"同文件撞车"与"跨文件"两类报。
#   本批是**干净层**（与库中无同末两级路径文件）⇒ 预期"同文件撞车"接近 0，主要是跨文件。
set -u
cd /root/autodl-tmp/MAS || exit 1
LOG=/root/autodl-tmp/batch_held_clean30.log
echo "=== 1) 生成 run 清单 ==="
venv/bin/python -X utf8 utils/experiments/make_run_list.py \
  --out reports/held_clean30_runs.txt --logs "$LOG" 2>&1 | tail -5
echo "  清单行数: $(wc -l < reports/held_clean30_runs.txt)"

echo
echo "=== 2) 库外口径评测（--db 必须指线上那份 mas.db，reports/mas_live.db 在服务器上是空文件）==="
venv/bin/python -X utf8 utils/experiments/eval_held_runs.py \
  --arms "干净层30=reports/held_clean30_runs.txt" \
  --db infrastructure/database/mas.db --max-rows 40 2>&1 | tail -60 | tee reports/held_clean30_eval.txt

echo
echo "=== 3) 完整性审计（批处理自己标的 partial 不等于证据不完整，要独立审）==="
venv/bin/python -X utf8 utils/experiments/audit_run_completeness.py \
  --reports-root reports/analysis \
  --batch-summary reports/held_clean30_runs.txt \
  --json-out reports/held_clean30_audit.json 2>&1 | tail -20

echo
echo "=== 4) 磁盘与新产物体积（本批是第一次在生产上带"写盘瘦身"跑）==="
df -h /root/autodl-tmp | tail -1
du -sh reports/analysis 2>/dev/null
echo "EVAL_CLEAN30_DONE"
