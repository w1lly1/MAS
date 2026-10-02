#!/bin/bash
# 切换"向量通道取几条"（weaviate_top_k），供 ③ 第 8 步的独立杠杆 A/B 用。
#
# 用法: bash srv_set_topk.sh show | 5 | 20
#
# 为什么单独做这个杠杆：离线核算（utils/experiments/a5c_universe_restriction.py）显示
# "自己的条目进没进候选池"卡在检索宽度上 —— top_k 从 5 提到 20，own 进池从 26/30 升到 29/30，
# 而**跨文件误报仍是 0**（多检索进来的候选照样要过 F(x) 的三道守卫）。
# 它与门控融合是**两个独立变量**，所以必须单独跑，别和开关混在一起。
set -u
MODE="${1:?用法: srv_set_topk.sh show|5|20}"
cd /root/autodl-tmp/MAS || exit 1

if [ "$MODE" != "show" ]; then
  ./venv/bin/python - "$MODE" <<'PY'
import json, sys
val = int(sys.argv[1])
p = "infrastructure/config/ai_agent_config.json"
cfg = json.load(open(p, encoding="utf-8"))
sp = cfg.setdefault("second_pass_analysis_agent", {})
old = sp.get("weaviate_top_k")
sp["weaviate_top_k"] = val
json.dump(cfg, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("weaviate_top_k: %r -> %r" % (old, val))
PY
fi

echo "--- 现状（做 A/B 前逐项核对）---"
./venv/bin/python - <<'PY'
import json
cfg = json.load(open("infrastructure/config/ai_agent_config.json", encoding="utf-8"))
sp = cfg["second_pass_analysis_agent"]
print("  weaviate_top_k =", sp.get("weaviate_top_k"))
print("  gate_fusion    =", sp.get("gate_fusion"))
print("  gap_chunk_semantic_lookup =", sp.get("gap_chunk_semantic_lookup"))
PY
echo "  配置 sha256[:16] = $(sha256sum infrastructure/config/ai_agent_config.json | cut -c1-16)"
echo "TOPK_SET_DONE"
