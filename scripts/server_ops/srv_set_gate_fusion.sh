#!/bin/bash
# 切换"门控加法融合"开关（JSON 安全改写，幂等），供 ③ 的 A/B 用。
#
# 用法:
#   bash srv_set_gate_fusion.sh off                 # 历史行为（默认）
#   bash srv_set_gate_fusion.sh on                  # 用配置里的 λ/θ
#   bash srv_set_gate_fusion.sh on 0.5 0.7          # 指定 λ/θ
#   bash srv_set_gate_fusion.sh show                # 只打印现状
#
# 为什么要有它：A/B 必须**只改一个变量**，而且要能证明"开关真的生效了"。
# 所以本脚本改完会打印：① 改动前后的值；② 配置 sha256；③ git diff 里这一处改动的原样。
set -u
MODE="${1:?用法: srv_set_gate_fusion.sh on|off|show [lambda] [theta]}"
LAM="${2:-}"
THETA="${3:-}"
cd /root/autodl-tmp/MAS || exit 1

if [ "$MODE" != "show" ]; then
  ./venv/bin/python - "$MODE" "$LAM" "$THETA" <<'PY'
import json, sys
mode, lam, theta = sys.argv[1], sys.argv[2], sys.argv[3]
p = "infrastructure/config/ai_agent_config.json"
cfg = json.load(open(p, encoding="utf-8"))
sp = cfg.setdefault("second_pass_analysis_agent", {})
gf = sp.setdefault("gate_fusion", {})
before = dict(gf)
if mode == "off":
    gf["enabled"] = False
else:
    gf["enabled"] = True
    if lam:
        gf["semantic_weight"] = float(lam)
    if theta:
        gf["admit_threshold"] = float(theta)
json.dump(cfg, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("gate_fusion: %s -> %s" % (before, gf))
PY
fi

echo "--- 现状 ---"
./venv/bin/python - <<'PY'
import json
cfg = json.load(open("infrastructure/config/ai_agent_config.json", encoding="utf-8"))
sp = cfg["second_pass_analysis_agent"]
print("  gate_fusion         =", sp.get("gate_fusion"))
print("  weaviate_top_k      =", sp.get("weaviate_top_k"))
print("  gap_chunk_semantic  =", sp.get("gap_chunk_semantic_lookup"))
print("  theta_s/tau/theta_a/theta_w = %s / %s / %s / %s" % (
    sp.get("gate_structured_threshold"), sp.get("similarity_threshold"),
    sp.get("gate_anchor_threshold"), sp.get("gate_weak_structure_threshold")))
PY
echo "  配置 sha256[:16] = $(sha256sum infrastructure/config/ai_agent_config.json | cut -c1-16)"
echo "--- git diff（这次开关改动的原样，便于写进实验记录）---"
git diff -U1 -- infrastructure/config/ai_agent_config.json | grep -E '^[-+].*(gate_fusion|enabled|semantic_weight|admit_threshold)' | head -12
echo "GATE_FUSION_SET_DONE"
