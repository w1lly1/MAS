#!/usr/bin/env python
"""C2 修复后的残余 hubness 画像：跨 9 文件仍共现的 KB 行到底是什么？

目标：判定 residual coexist（A/B 后为 6）是"真正通用型知识条目"还是"仍未被消除的伪 hub"。
同时给出按层的共现分解，以及"被 ≥k 个文件命中"的行数分布（看长尾形状）。
"""
import glob
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

BASE = {"f5fbeb25-8c47-4b11-b3a1-870ceafe10bb", "42efd1e7-b6fb-4874-af25-11531b9779b1",
        "a78ea346-820c-4a87-9b7e-e9a96ad5f098", "66f9ad5f-4407-4ce0-a772-12a9a737ec7c",
        "9a5a2f2d-4b21-4763-ab10-6730bde0bc48", "ba139961-bc84-45e0-8b62-39246b6928fe",
        "902dc158-5c26-4f45-acf6-4d9acd10dfdd", "a1ca5b0b-0c8a-4cdc-bb43-044d0c16cb9e"}
QOFF2 = {"0fbe60f4-dea6-4cfd-864a-f789fd0194d0", "12dc120e-2c9d-49fe-8c2d-8e2a614c8f26",
         "6c817c11-76ae-41c9-ac66-ad027c9415a8", "76be00c4-52e3-41e8-afc5-afa29b9edf94",
         "8ca34dfe-f407-4a65-abad-38634eff4b31", "b63de181-9078-4c9e-9c34-e51ffe54c425",
         "ce3ceece-684a-433e-bf6f-0021e0bb6a6c", "df87d95a-b194-4694-8574-a77007d26706"}
LAYERS = ["semantic", "code_pattern", "solution", "full"]


def collect(runs):
    rowfiles = defaultdict(set)          # sid -> set(files)
    rowfiles_layer = defaultdict(set)    # (layer, sid) -> set(files)
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        run = f.split("/")[3]
        if run not in runs:
            continue
        d = json.load(open(f, encoding="utf-8"))
        ana = os.path.basename(str(d.get("file") or ""))
        for b in ("retrieval_evidence", "gap_retrieval_evidence"):
            for it in (d.get(b) or []):
                for h in (it.get("weaviate_hits") or []):
                    sid = h.get("sqlite_id")
                    L = str(h.get("vector_layer"))
                    rowfiles[sid].add(ana)
                    rowfiles_layer[(L, sid)].add(ana)
    return rowfiles, rowfiles_layer


def profile(name, runs):
    rf, rfl = collect(runs)
    nf = len({x for s in rf.values() for x in s})
    print("\n### %s   （涉及文件 %d，被召回行 %d/200）" % (name, nf, len(rf)))
    print("  按『被几个文件命中』分布:")
    byk = Counter(len(v) for v in rf.values())
    for k in sorted(byk, reverse=True):
        print("     被 %d 个文件命中: %d 行" % (k, byk[k]))
    print("  按层：跨全部 %d 文件共现的行数" % nf)
    for L in LAYERS:
        co = [sid for (l, sid), fs in rfl.items() if l == L and len(fs) >= nf]
        print("     %-14s %d 行  %s" % (L, len(co), sorted(co)[:12]))
    co_all = sorted({sid for sid, fs in rf.items() if len(fs) >= nf})
    return rf, rfl, co_all, nf


def main():
    print("=" * 78)
    rf_b, _, co_b, nf_b = profile("基线（偏移关闭）", BASE)
    rf_q, _, co_q, nf_q = profile("C2 修复后（偏移开启，真实拟合）", QOFF2)

    sq = sqlite3.connect(os.path.join(ROOT, "infrastructure", "database", "mas.db"))
    rows = {r[0]: r for r in sq.execute(
        "select id, error_type, severity, language, framework, file_pattern, class_pattern, "
        "length(error_description), substr(error_description,1,110) from issue_patterns").fetchall()}

    for tag, co in (("基线残余共现", co_b), ("修复后残余共现", co_q)):
        print("\n=== %s 的 KB 行画像 ===" % tag)
        if not co:
            print("  （无）"); continue
        et = Counter(); fw = Counter(); lang = Counter(); ln = []
        for sid in co:
            r = rows.get(sid)
            if not r:
                print("  sid=%s 不在 SQLite（!）" % sid); continue
            et[r[1]] += 1; fw[r[4]] += 1; lang[r[3]] += 1; ln.append(r[7] or 0)
        print("  error_type 分布:", dict(et))
        print("  framework  分布:", dict(fw))
        print("  language   分布:", dict(lang))
        print("  error_description 长度: 中位 %s  最长 %s" % (
            int(np.median(ln)) if ln else 0, max(ln) if ln else 0))
        print("  明细:")
        for sid in co:
            r = rows.get(sid)
            if r:
                print("     sid=%-4s etype=%-20s fw=%-14s file_pattern=%-28s desc=%s" % (
                    sid, r[1], r[4], (r[5] or "")[:28], (r[8] or "").replace("\n", " ")))

    print("\n=== 全库 error_type 分布（对照：判断残余行是否'通用型'）===")
    allc = Counter(r[1] for r in rows.values())
    print("  ", dict(allc.most_common(10)))

    json.dump({"baseline_coexist": co_b, "qoff_coexist": co_q},
              open("/root/autodl-tmp/residual_hub.json", "w"), ensure_ascii=False, indent=1)
    print("\n已写出 /root/autodl-tmp/residual_hub.json")


if __name__ == "__main__":
    sys.exit(main())
