# -*- coding: utf-8 -*-
"""验证 #5 冒烟：10 个缺失型 CVE 的二次校验是否误报(new_findings>0)。"""
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/MAS")
AN = ROOT / "reports" / "analysis"
MAN = ROOT / "reports" / "negative_exp_manifest_400_missing.json"

man = json.loads(MAN.read_text(encoding="utf-8"))
cves = [r["cve"] for r in man.get("rows", [])]

total_nf = 0
fp = []
for cve in cves:
    d = AN / cve
    nf = 0
    if d.is_dir():
        runs = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime)
        if runs:
            run = runs[-1]
            sp = sorted((run / "second_pass" / "consolidated").glob("*.json"))
            if not sp:
                sp = sorted((run / "fullLayer" / "consolidated").glob("*.json"))
            if sp:
                j = json.loads(sp[-1].read_text(encoding="utf-8"))
                nf = len(j.get("new_findings", []))
    total_nf += nf
    if nf > 0:
        fp.append((cve, nf))

print(f"缺失型样本数: {len(cves)}")
print(f"总 new_findings 条数: {total_nf}")
print(f"产生误报(new_findings>0)的样本: {len(fp)}")
for cve, n in fp:
    print(f"  {cve}: {n}")
