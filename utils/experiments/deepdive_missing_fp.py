# -*- coding: utf-8 -*-
"""深挖 #5 的 10 个误报：缺失型 CVE 被错配到了哪个 KB 错误型 pattern。"""
import json
import sqlite3
from pathlib import Path

ROOT = Path("/root/autodl-tmp/MAS")
AN = ROOT / "reports" / "analysis"
DB = ROOT / "infrastructure" / "database" / "mas.db"
MAN = ROOT / "reports" / "negative_exp_manifest_400_missing.json"

con = sqlite3.connect(str(DB))
cur = con.cursor()
cur.execute("SELECT id, title, error_type FROM issue_patterns")
id2title = {i: (t, e) for i, t, e in cur.fetchall()}
con.close()

man = json.loads(MAN.read_text(encoding="utf-8"))
cves = [r["cve"] for r in man.get("rows", [])]

for cve in cves:
    d = AN / cve
    if not d.is_dir():
        continue
    runs = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.stat().st_mtime)
    if not runs:
        continue
    run = runs[-1]
    sp = sorted((run / "second_pass" / "consolidated").glob("*.json"))
    if not sp:
        sp = sorted((run / "fullLayer" / "consolidated").glob("*.json"))
    if not sp:
        continue
    j = json.loads(sp[-1].read_text(encoding="utf-8"))
    nf = j.get("new_findings", [])
    if not nf:
        continue
    print(f"\n=== {cve} (误报 {len(nf)} 条) ===")
    for x in nf:
        ev = x.get("evidence") or {}
        sid = ev.get("sqlite_id")
        ch = ev.get("channel")
        title, etype = id2title.get(sid, ("?", "?"))
        desc = str(x.get("description") or "")[:120]
        print(f"  错配到 KB pattern: sqlite_id={sid} -> {title} (type={etype})")
        print(f"    通道={ch}, 描述={desc}")
