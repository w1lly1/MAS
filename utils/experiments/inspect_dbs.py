# -*- coding: utf-8 -*-
import json
import sqlite3
from pathlib import Path

ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
DB1 = ROOT / "infrastructure/database/mas.db"
DB2 = ROOT / "reports/mas_seed2025.db"

def load_issue_patterns(db):
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM issue_patterns")]
    con.close()
    return rows

def load_curated(db):
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM curated_issues")]
    con.close()
    return rows

p1 = load_issue_patterns(DB1)
p2 = load_issue_patterns(DB2)
print("mas.db issue_patterns:", len(p1), " mas_seed2025.db issue_patterns:", len(p2))

# columns
cols = list(p1[0].keys())
print("issue_patterns cols:", cols)

# compare by id
ids1 = [r["id"] for r in p1]
ids2 = [r["id"] for r in p2]
print("same id set:", set(ids1) == set(ids2), " same order:", ids1 == ids2)

# compare full content per id (excluding id ordering)
def content_sig(rows):
    d = {}
    for r in rows:
        d[r["id"]] = {k: r[k] for k in r if k != "id"}
    return d
s1 = content_sig(p1)
s2 = content_sig(p2)
same = 0
diff = []
for pid in s1:
    if pid not in s2:
        diff.append((pid, "missing in db2"))
    elif s1[pid] != s2[pid]:
        diff.append((pid, "content differs"))
    else:
        same += 1
print(f"identical content rows: {same}, diff: {len(diff)}")
for d in diff[:10]:
    print("  diff:", d)

# curated_issues: code_snippet coverage
for db, name in [(DB1, "mas.db"), (DB2, "mas_seed2025.db")]:
    cur = load_curated(db)
    nonempty = [r for r in cur if (r.get("code_snippet") or "").strip()]
    pids = set(r["pattern_id"] for r in nonempty)
    print(f"{name}: curated_issues={len(cur)}, with non-empty code_snippet={len(nonempty)}, distinct pattern_ids={len(pids)}")

# manifest kb list
man = json.loads((ROOT / "reports/negative_exp_manifest_400_error.json").read_text(encoding="utf-8"))
kb = [r["cve"] for r in man["rows"] if r["role"] == "kb"]
held = [r["cve"] for r in man["rows"] if r["role"] == "held"]
print("manifest rows:", len(man["rows"]), " kb:", len(kb), " held:", len(held))
print("manifest top-level keys:", list(man.keys()))
if man["rows"]:
    print("manifest row keys:", list(man["rows"][0].keys()))
