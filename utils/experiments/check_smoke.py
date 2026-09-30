# -*- coding: utf-8 -*-
"""冒烟测试二次校验结果检查：6 kb 是否被召回(self_hit)。"""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/MAS")
DB = ROOT / "infrastructure" / "database" / "mas.db"
AN = ROOT / "reports" / "analysis"

con = sqlite3.connect(str(DB))
cur = con.cursor()
cur.execute("SELECT id, title FROM issue_patterns")
id_by_title = {t: i for i, t in cur.fetchall()}
cur.execute("SELECT id, pattern_id FROM curated_issues")
ci_to_p = {i: p for i, p in cur.fetchall()}
con.close()

KB = ["CVE-2018-13301", "CVE-2015-1465", "CVE-2019-3554",
      "CVE-2016-7914", "CVE-2015-5221", "CVE-2017-6439"]
HELD = ["CVE-2014-9728", "CVE-2014-6229"]

def check(cve):
    d = AN / cve
    if not d.is_dir():
        return f"{cve}: NO_DIR"
    runs = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime)
    if not runs:
        return f"{cve}: NO_RUN"
    run = runs[-1]
    own = id_by_title.get(cve)
    sp = sorted((run / "second_pass" / "consolidated").glob("*.json"))
    if not sp:
        sp = sorted((run / "fullLayer" / "consolidated").glob("*.json"))
    if not sp:
        return f"{cve}: NO_CONSOLIDATED (own={own})"
    j = json.loads(sp[-1].read_text(encoding="utf-8"))
    nf = j.get("new_findings", [])
    self_hit = any(
        ((x.get("evidence") or {}).get("channel") == "curated_issue"
         and ci_to_p.get((x.get("evidence") or {}).get("sqlite_id")) == own)
        or ((x.get("evidence") or {}).get("sqlite_id") == own)
        for x in nf
    )
    return f"{cve}: own={own} n_new_findings={len(nf)} self_hit={self_hit}"

print("=== kb (应 self_hit=True) ===")
for c in KB:
    print(" ", check(c))
print("=== held (应 self_hit=False) ===")
for c in HELD:
    print(" ", check(c))
