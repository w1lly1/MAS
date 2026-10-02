#!/bin/bash
# A5b 融合扫描阶段：score = s_lex + λ·s_sem ≥ θ，语义项用 LLM 判定（从缓存读）。
# 纯 CPU、确定性，不需要 GPU。输出落 reports/a5b_fusion.log。
set -u
cd /root/autodl-tmp/MAS || exit 1
mkdir -p reports
CACHE=reports/a5b_llm_verdicts.json
if [ ! -f "$CACHE" ]; then echo "缺 LLM 判定缓存 $CACHE"; exit 1; fi
echo "判定缓存 sha256: $(sha256sum "$CACHE" | cut -c1-16)"
echo "判定条数: $(python3 -c 'import json;print(sum(len(v) for v in json.load(open("reports/a5b_llm_verdicts.json")).values()))')"
nohup venv/bin/python -X utf8 -u utils/experiments/a5b_graded_fusion.py --stage fusion \
  > reports/a5b_fusion.log 2>&1 &
echo "pid=$!"
