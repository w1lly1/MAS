#!/bin/bash
# 在 GPU 上跑 A5b 的 LLM 配对判定阶段（135 条，判定结果增量落盘）。
# 说明：脚本内部会先复算分级 s_lex（读 tests/BigVul 源码 + KB dump），再逐对判定。
# 判定用 Qwen1.5-7B-Chat，GPU 上 float16（由 _load_qwen 自动选设备，MAS_JUDGE_DEVICE 可覆盖）。
set -u
cd /root/autodl-tmp/MAS || exit 1
mkdir -p reports
DUMP=reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl
if [ ! -f "$DUMP" ]; then
  echo "缺 KB dump：$DUMP —— 先把它传上来"; exit 1
fi
if [ -f reports/a5b_llm_verdicts.json ]; then
  echo "已有判定缓存：$(python3 -c 'import json,sys;print(sum(len(v) for v in json.load(open("reports/a5b_llm_verdicts.json")).values()))') 条（会跳过已判定的）"
fi
nohup env MAS_JUDGE_DEVICE=cuda venv/bin/python -X utf8 -u \
  utils/experiments/a5b_graded_fusion.py --stage llm --llm-topk 3 \
  > reports/a5b_llm.log 2>&1 &
echo "pid=$!"
sleep 25
echo "--- 日志前 12 行 ---"
head -12 reports/a5b_llm.log
echo "--- 日志末 3 行 ---"
tail -3 reports/a5b_llm.log
echo "--- 显存 ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
