# -*- coding: utf-8 -*-
"""seed=2025 批次的部分结果验证：对已完成 CVE 统计二次校验召回/误报。"""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/MAS")
DB = ROOT / "infrastructure" / "database" / "mas.db"
AN = ROOT / "reports" / "analysis"
MANIFEST = ROOT / "reports" / "negative_exp_manifest_400_error.json"

con = sqlite3.connect(str(DB))
cur = con.cursor()
cur.execute("SELECT id, title FROM issue_patterns")
id_by_title = {t: i for i, t in cur.fetchall()}
cur.execute("SELECT id, pattern_id FROM curated_issues")
ci_to_p = {i: p for i, p in cur.fetchall()}
con.close()

man = json.loads(MANIFEST.read_text(encoding="utf-8"))
kb_set = set(man.get("kb_cves", []))
held_set = set(man.get("held_cves", []))

def self_hit(cve):
    d = AN / cve
    runs = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime)
    if not runs:
        return None
    run = runs[-1]
    own = id_by_title.get(cve)
    sp = sorted((run / "second_pass" / "consolidated").glob("*.json"))
    if not sp:
        sp = sorted((run / "fullLayer" / "consolidated").glob("*.json"))
    if not sp:
        return False
    j = json.loads(sp[-1].read_text(encoding="utf-8"))
    nf = j.get("new_findings", [])
    return any(
        ((x.get("evidence") or {}).get("channel") == "curated_issue"
         and ci_to_p.get((x.get("evidence") or {}).get("sqlite_id")) == own)
        or ((x.get("evidence") or {}).get("sqlite_id") == own)
        for x in nf
    )

kb_done = kb_cap = held_done = held_fp = 0
notdone_kb = []
notdone_held = []
for cve in sorted(kb_set):
    if (AN / cve).is_dir():
        kb_done += 1
        if self_hit(cve):
            kb_cap += 1
    else:
        notdone_kb.append(cve)
for cve in sorted(held_set):
    if (AN / cve).is_dir():
        held_done += 1
        if self_hit(cve):
            held_fp += 1
    else:
        notdone_held.append(cve)

print(f"=== seed=2025 部分结果（已完成 CVE）===")
print(f"kb:   完成 {kb_done}/200，召回 {kb_cap} → {kb_cap/kb_done:.1%}  (seed=2024 全程 63.0%)")
print(f"held: 完成 {held_done}/200，误报 {held_fp} → {held_fp/held_done:.1%}  (seed=2024 全程 4.0%)")
print(f"未完成: kb {len(notdone_kb)} 个, held {len(notdone_held)} 个")
