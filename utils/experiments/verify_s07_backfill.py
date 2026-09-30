#!/usr/bin/env python
"""核验 S0-7 的真实影响：门控看到的 file_pattern 是否已由 SQLite 回填？

背景：我此前据「Weaviate 800 对象的 file_pattern 属性全为空（0/800）」判定
「跨文件守卫与同文件锚点只能靠从 error_description 散文里抠假文件名」（S0-7）。
但代码里 `_backfill_weaviate_candidate_solution()` 会用 sqlite_id 从 SQLite
回填 solution / error_description / class_pattern / file_pattern。

本脚本用【落盘的候选】实证核验：候选里的 file_pattern 是否等于 SQLite 的真值。
"""
import glob
import json
import os
import sqlite3
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

QOFF2 = {"0fbe60f4-dea6-4cfd-864a-f789fd0194d0", "12dc120e-2c9d-49fe-8c2d-8e2a614c8f26",
         "6c817c11-76ae-41c9-ac66-ad027c9415a8", "76be00c4-52e3-41e8-afc5-afa29b9edf94",
         "8ca34dfe-f407-4a65-abad-38634eff4b31", "b63de181-9078-4c9e-9c34-e51ffe54c425",
         "ce3ceece-684a-433e-bf6f-0021e0bb6a6c", "df87d95a-b194-4694-8574-a77007d26706"}


def main():
    sq = sqlite3.connect(os.path.join(ROOT, "infrastructure", "database", "mas.db"))
    fpm = {str(r[0]): (r[1] or "") for r in
           sq.execute("select id, file_pattern from issue_patterns").fetchall()}
    print("SQLite file_pattern 非空: %d / %d" % (sum(1 for v in fpm.values() if v), len(fpm)))

    stat = Counter()
    mismatch = []
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        if f.split("/")[3] not in QOFF2:
            continue
        d = json.load(open(f, encoding="utf-8"))
        for b in ("retrieval_evidence", "gap_retrieval_evidence"):
            for it in (d.get(b) or []):
                for c in (it.get("candidates") or []):
                    ch = str(c.get("channel"))
                    sid = c.get("sqlite_id")
                    cp = str(c.get("file_pattern") or "").strip()
                    truth = fpm.get(str(sid), "") if sid is not None else ""
                    if ch == "weaviate":
                        if cp and cp == truth:
                            stat["weaviate_候选file_pattern==SQLite真值"] += 1
                        elif cp and truth and cp != truth:
                            stat["weaviate_候选与真值不一致"] += 1
                            if len(mismatch) < 5:
                                mismatch.append((sid, cp, truth))
                        elif cp and not truth:
                            stat["weaviate_候选有值但SQLite无"] += 1
                        elif not cp and truth:
                            stat["weaviate_候选为空但SQLite有值(回填失败)"] += 1
                        else:
                            stat["weaviate_两边都空"] += 1
                    elif ch == "curated_issue":
                        stat["curated_候选(其file_pattern取自curated_issues.file_path)"] += 1
                    else:
                        stat["其它通道"] += 1
    print("\n=== weaviate 候选的 file_pattern 核验（qoff2 运行）===")
    for k, v in stat.most_common():
        print("  %-52s %d" % (k, v))
    if mismatch:
        print("\n  不一致样例 (sid, 候选值, SQLite真值):")
        for m in mismatch:
            print("     ", m)

    # 门控实际用到的 knowledge_base：_knowledge_file_basename 读 candidate['file_pattern']
    tot = stat.get("weaviate_候选file_pattern==SQLite真值", 0) + stat.get("weaviate_候选为空但SQLite有值(回填失败)", 0)
    if tot:
        print("\n=== 结论 ===")
        print("  weaviate 候选中 file_pattern 正确回填比例: %.2f%% (%d/%d)" % (
            100.0 * stat.get("weaviate_候选file_pattern==SQLite真值", 0) / tot,
            stat.get("weaviate_候选file_pattern==SQLite真值", 0), tot))


if __name__ == "__main__":
    sys.exit(main())
