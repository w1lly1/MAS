#!/bin/bash
# B6 收尾的"证据核实"：① 批处理自报的 partial；② 关融合是否真关（配置 + 产物里不该有 fusion 字段）；
# ③ 30 个 run 的完整性口径；④ 把要拉回本地的小文件列出来。
set -u
cd /root/autodl-tmp/MAS || exit 1

echo "=== ① 批处理汇总（partial 计数用只匹配『（记为 partial』的方式，避免假阳性）==="
tail -12 reports/batch_summary.csv 2>/dev/null | cut -c1-200
echo "  本批日志里『记为 partial』出现次数: $(grep -c '（记为 partial' /root/autodl-tmp/batch_held_clean30.log || true)"

echo
echo "=== ② 关融合是否真关 ==="
venv/bin/python - <<'PY'
import json
cfg = json.load(open("infrastructure/config/ai_agent_config.json", encoding="utf-8"))
sp = cfg["second_pass_analysis_agent"]
print("  gate_fusion =", sp.get("gate_fusion"))
PY
echo "  配置文件 sha256[:16] = $(sha256sum infrastructure/config/ai_agent_config.json | cut -c1-16)"
echo "  git 里这一版是否被改过: $(git status --porcelain infrastructure/config/ai_agent_config.json | wc -l) 处（0 = 与仓库一致）"

echo
echo "=== ③ 产物里是否出现过融合字段（关闭时应当为 0）==="
RUN=$(head -1 reports/held_clean30_runs.txt)
echo "  抽查 run: $RUN"
grep -rl 'fusion_score' "reports/analysis/${RUN%/*}" 2>/dev/null | head -3
echo "  该 run 目录下含 fusion_score 的文件数: $(grep -rl 'fusion_score' "reports/analysis/${RUN%/*}" 2>/dev/null | wc -l)"

echo
echo "=== ④ 本批 run 的完成度（run_summary 是否齐全）==="
venv/bin/python - <<'PY'
from pathlib import Path
import json
root = Path("reports/analysis")
lst = Path("reports/held_clean30_runs.txt").read_text(encoding="utf-8").split()
ok = bad = 0
for rel in lst:
    p = root / rel / "run_summary.json"
    if p.is_file():
        ok += 1
    else:
        bad += 1
        print("   缺 run_summary:", rel)
print("  30 条 run 中 run_summary 存在 %d 条，缺失 %d 条" % (ok, bad))
PY

echo
echo "=== ⑤ 待拉回本地的小文件 ==="
ls -la reports/held_clean30_runs.txt reports/held_clean30_eval.txt reports/held_clean30_audit.json reports/batch_summary.csv 2>/dev/null | awk '{print "  "$5"  "$9}'
echo "VERIFY_CLEAN30_DONE"
