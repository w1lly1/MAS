# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path

ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
DB1 = ROOT / "infrastructure/database/mas.db"
DB2 = ROOT / "reports/mas_seed2025.db"

TEXT_COLS = ["title", "error_type", "severity", "language", "framework",
             "error_description", "problematic_pattern", "solution",
             "file_pattern", "class_pattern"]

def load(db):
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    rows = {r["id"]: dict(r) for r in con.execute("SELECT * FROM issue_patterns")}
    con.close()
    return rows

p1 = load(DB1)
p2 = load(DB2)

# compare text cols
same = 0
diffs = []
for pid in p1:
    a = {c: p1[pid].get(c) for c in TEXT_COLS}
    b = {c: p2[pid].get(c) for c in TEXT_COLS}
    if a == b:
        same += 1
    else:
        diffs.append((pid, [c for c in TEXT_COLS if a[c] != b[c]]))
print("same text cols:", same, "/ 200")
print("diff rows:", len(diffs))
for pid, cols in diffs[:20]:
    print("  id", pid, "differ cols:", cols)

# also check kb_code presence
print("\nkb_code sample db1:", {pid: p1[pid].get('kb_code') for pid in [1,2,3]})
print("kb_code sample db2:", {pid: p2[pid].get('kb_code') for pid in [1,2,3]})

# curated_issues text compare per pattern (first snippet)
def load_cur(db):
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM curated_issues")]
    con.close()
    return rows

c1 = load_cur(DB1)
c2 = load_cur(DB2)
print("\ncurated_issues cols:", list(c1[0].keys()))

# group first code_snippet per pattern_id
def first_snip(rows):
    d = {}
    for r in rows:
        pid = r["pattern_id"]
        if pid not in d:
            d[pid] = (r.get("code_snippet") or "")
    return d

s1 = first_snip(c1)
s2 = first_snip(c2)
same = 0
diffs = []
for pid in s1:
    if s1[pid] == s2[pid]:
        same += 1
    else:
        diffs.append(pid)
print("curated first code_snippet same:", same, "/ 200, differ:", len(diffs))
if diffs:
    print("differ pids (first 10):", diffs[:10])
