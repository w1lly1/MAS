#!/bin/bash
# 核实"放行 0"到底指什么：直接读一个 run 的二次分析产物，数 evidence_hits / new_findings。
# 目的：把 CSV 里的 vulns_detected=3（安全 agent 自己报的）与评测口径的"放行"分开，防止误读。
set -u
cd /root/autodl-tmp/MAS || exit 1
venv/bin/python - <<'PY'
import json, glob, os
runs = open("reports/held_clean30_runs.txt", encoding="utf-8").read().split()
print("run 总数:", len(runs))
tot = {"evidence_hits": 0, "new_findings": 0, "low_confidence": 0, "candidates": 0, "files": 0}
per = []
for rel in runs:
    files = glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*.json"), recursive=True)
    e = {"evidence_hits": 0, "new_findings": 0, "low_confidence": 0, "candidates": 0}
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for k in e:
            v = d.get(k)
            if isinstance(v, list):
                e[k] += len(v)
        tot["files"] += 1
    for k in e:
        tot[k] += e[k]
    per.append((rel.split("/")[0], e))
print("汇总:", tot)
print("非零放行的 run:")
for cve, e in per:
    if e["evidence_hits"] or e["new_findings"]:
        print("   ", cve, e)
print("（若上面为空 ⇒ 本臂 30 个样本确实一条都没放行）")
PY
echo "VERIFY_SEMANTICS_DONE"
