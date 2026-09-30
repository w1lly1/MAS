# -*- coding: utf-8 -*-
"""#5 部分结果分析：已完成缺失型 CVE 的误报(new_findings>0)统计。"""
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/MAS")
AN = ROOT / "reports" / "analysis"
MAN = ROOT / "reports" / "negative_exp_manifest_400_missing.json"

man = json.loads(MAN.read_text(encoding="utf-8"))
cves = [r["cve"] for r in man.get("rows", [])]

done = 0
fp = []
nf_dist = {}
for cve in cves:
    d = AN / cve
    if not d.is_dir():
        continue
    done += 1
    nf = 0
    runs = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime)
    if runs:
        run = runs[-1]
        sp = sorted((run / "second_pass" / "consolidated").glob("*.json"))
        if not sp:
            sp = sorted((run / "fullLayer" / "consolidated").glob("*.json"))
        if sp:
            j = json.loads(sp[-1].read_text(encoding="utf-8"))
            nf = len(j.get("new_findings", []))
    nf_dist[nf] = nf_dist.get(nf, 0) + 1
    if nf > 0:
        fp.append((cve, nf))

print(f"已完成缺失型样本: {done}/{len(cves)}")
print(f"误报(new_findings>0): {len(fp)} 个 → 误报率 {len(fp)/done:.1%}")
print(f"new_findings 分布: {dict(sorted(nf_dist.items()))}")
if fp:
    print("误报明细:")
    for cve, n in fp:
        print(f"  {cve}: {n} 条 new_findings")
