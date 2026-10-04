#!/bin/bash
# 当前状态快照（只读）：代码版本 / 配置开关 / 知识库指纹 / 磁盘 / 进程。
# 这台机器每次开机容器名都会变，所以"现在到底是什么状态"必须能一条命令打出来。
set -u
cd /root/autodl-tmp/MAS || exit 1

echo "=== 机器 ==="
echo "容器名    : $(hostname)"
echo "磁盘      : $(df -h /root/autodl-tmp | tail -1 | awk '{print $3" used / "$4" avail ("$5")"}')"
echo "GPU       : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
echo
echo "=== 代码 ==="
echo "HEAD          : $(git log --oneline -1)"
echo "被改动的跟踪文件: $(git status --porcelain | grep -v '^??' | wc -l) 个"
echo "配置 sha256[:16]: $(sha256sum infrastructure/config/ai_agent_config.json | cut -c1-16)"
echo
echo "=== 配置开关 ==="
venv/bin/python - <<'PY'
import json
c = json.load(open("infrastructure/config/ai_agent_config.json", encoding="utf-8"))
sp = c["second_pass_analysis_agent"]
for k in ("gate_fusion", "weaviate_top_k", "similarity_threshold",
          "gate_structured_threshold", "gate_anchor_threshold",
          "gate_weak_structure_threshold", "gap_chunk_semantic_lookup",
          "gap_query_source", "query_offset_correction", "dump_query_vectors"):
    print("  %-28s = %s" % (k, sp.get(k)))
print("  artifact_settings            = %s" % {k: v for k, v in (c.get("artifact_settings") or {}).items() if k != "_comment"})
PY
echo
echo "=== 知识库 ==="
echo "mas.db sha256[:16]: $(sha256sum infrastructure/database/mas.db | cut -c1-16)  （停机快照里是 355fb71efda74cfa）"
echo "Weaviate 进程     : $(pgrep -f 'weaviate --host' | wc -l) 个（0 = 没在跑）"
echo "批次进程          : $(pgrep -f 'venv/bin/python mas[.]py batch' | wc -l) 个"
echo
echo "=== 本批/本臂相关文件 ==="
for f in reports/held_clean30_runs.txt reports/held_clean30_eval.txt reports/clean30_candidates_agg.txt; do
  if [ -f "$f" ]; then echo "  有 $f"; else echo "  无 $f"; fi
done
echo "STATE_NOW_DONE"
