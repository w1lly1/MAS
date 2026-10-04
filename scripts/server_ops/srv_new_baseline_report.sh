#!/bin/bash
# **新基线对照报告**：embedder 修复后的检索侧数字，与"死 embedder 时代"逐项对照。
# 用法: bash srv_new_baseline_report.sh [runs 列表，默认 reports/baseline_fix_runs.txt]
set -u
cd /root/autodl-tmp/MAS || exit 1
RUNS="${1:-reports/baseline_fix_runs.txt}"
DUMP=reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl
DB=infrastructure/database/mas.db

echo "############ 1) 召回构成（通道：词元 vs 语义）############"
echo "# 旧数字（死 embedder）：curated 87.9% / weaviate 12.0% / sqlite 0.2%；自有条目 30/30 全靠词元"
venv/bin/python -X utf8 utils/experiments/analyze_recall_channel.py \
  --arms "新基线=$RUNS" --db "$DB" --show-own 2>&1 | tail -40

echo
echo "############ 2) 语义天花板（预检 2 口径）############"
echo "# 旧数字（死 embedder）：语义候选 17518 条，最大相似度 0.625 < τ 0.65，达到 τ 的 0 条"
venv/bin/python -X utf8 utils/experiments/precheck2_semantic_ceiling.py \
  --runs "$RUNS" --db "$DB" --dump "$DUMP" --limit 30 \
  --json-out reports/precheck2_new_baseline.json 2>&1 | tail -30

echo
echo "############ 3) 该臂的生效性（本臂融合是**关**的，产物里不该有融合字段）############"
FIRST=$(head -1 "$RUNS")
echo "  抽查 run: $FIRST"
echo "  含 fusion_score 的文件数: $(grep -rl fusion_score "reports/analysis/$FIRST" 2>/dev/null | wc -l)（期望 0）"

echo
echo "############ 4) 磁盘 ############"
df -h /root/autodl-tmp | tail -1
echo "NEW_BASELINE_REPORT_DONE"
