#!/bin/bash
# 查证：否决判据用的 solution 来自哪张表？（curated 通道的候选来自 curated_issues）
set -u
cd /root/autodl-tmp/MAS || exit 1
./venv/bin/python - <<'PY'
import sqlite3

for p in ("infrastructure/database/mas.db", "/root/autodl-tmp/mas_rebuild_candidate.db"):
    print("=" * 90)
    print("库:", p)
    con = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    print("  表:", tables)
    if "curated_issues" in tables:
        cols = [r[1] for r in con.execute("pragma table_info(curated_issues)")]
        print("  curated_issues 列:", cols)
        rows = list(con.execute("select * from curated_issues where id=163"))
        if rows:
            rec = dict(zip(cols, rows[0]))
            for k, v in rec.items():
                s = str(v or "")
                if "Remove incorrect logic" in s or len(s) > 120:
                    print("    %-22s %s" % (k, s[:240].replace("\n", " | ")))
        else:
            print("    （没有 id=163）")
        n = con.execute("select count(*) from curated_issues").fetchone()[0]
        print("  curated_issues 行数:", n)
    if "issue_patterns" in tables:
        cols = [r[1] for r in con.execute("pragma table_info(issue_patterns)")]
        if "solution" in cols:
            r = list(con.execute("select solution from issue_patterns where id=127"))
            if r:
                s = str(r[0][0] or "")
                print("  issue_patterns 127 的 solution 前 200 字:")
                print("     ", s[:200].replace("\n", " | "))
                print("      其中 payload 按单分号切: %d 段；按双分号切: %d 段"
                      % (len([x for x in s.split(";") if x.strip()]),
                         len([x for x in s.split(";;") if x.strip()])))
    con.close()
PY
