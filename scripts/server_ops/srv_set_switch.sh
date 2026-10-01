#!/bin/bash
# 切换"代码维"的开关：gap_chunk_semantic_lookup（JSON 安全改写，幂等）
# 用法: bash _srv_set_switch.sh on|off
set -u
MODE="${1:?用法: _srv_set_switch.sh on|off}"
cd /root/autodl-tmp/MAS || exit 1
./venv/bin/python - "$MODE" <<'PY'
import json, sys
mode = sys.argv[1]
want = (mode == "on")
p = "infrastructure/config/ai_agent_config.json"
cfg = json.load(open(p, encoding="utf-8"))
sp = cfg.setdefault("second_pass_analysis_agent", {})
old = sp.get("gap_chunk_semantic_lookup")
sp["gap_chunk_semantic_lookup"] = want
json.dump(cfg, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("gap_chunk_semantic_lookup: %r -> %r" % (old, want))
PY
grep -n '"gap_chunk_semantic_lookup"' infrastructure/config/ai_agent_config.json
