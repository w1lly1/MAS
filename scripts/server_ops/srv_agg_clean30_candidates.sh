#!/bin/bash
# 按**正确层级**（retrieval_evidence[*].candidates）聚合干净层这一臂的候选与判定。
# 要证明的是：放行 0 是"候选产生了但被门控拒掉"，不是"二次分析没跑"。
set -u
cd /root/autodl-tmp/MAS || exit 1
venv/bin/python - <<'PY'
import json, glob, os
from collections import Counter

runs = open("reports/held_clean30_runs.txt", encoding="utf-8").read().split()
dec, rej, ch = Counter(), Counter(), Counter()
n_files = n_ev = 0
per_run = []
for rel in runs:
    files = glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*.json"), recursive=True)
    c_run = 0
    adm_run = 0
    for f in files:
        n_files += 1
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for block in (d.get("retrieval_evidence") or []) + (d.get("gap_retrieval_evidence") or []):
            if not isinstance(block, dict):
                continue
            n_ev += 1
            for c in block.get("candidates") or []:
                c_run += 1
                ch[str(c.get("channel") or "?")] += 1
                gd = str(c.get("gating_decision") or "(空)")
                dec[gd] += 1
                if gd in ("formal_hit", "explanatory_hit"):
                    adm_run += 1
                r = str(c.get("rejection_reason") or "")
                if r:
                    rej[r] += 1
        adm_run += len(d.get("new_findings") or [])
    per_run.append((rel.split("/")[0], c_run, adm_run))

    for x in d.get("issues") or []:
        pass
print("读到的二次分析产物文件数:", n_files, " 证据块数:", n_ev)
print("候选总数:", sum(v for v in dec.values()))
print("判定分布:", dict(dec))
print("通道分布:", dict(ch))
print("拒绝理由（前 8）:", dict(rej.most_common(8)))
print("放行（formal+explanatory+new_findings）合计:", sum(r[2] for r in per_run))
print("候选数为 0 的样本:", [r[0] for r in per_run if r[1] == 0])
PY
echo "AGG_DONE"
