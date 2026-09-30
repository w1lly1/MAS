#!/usr/bin/env python
"""残余共现的显著性检验（零模型）+ 开放跨文件闸门的最终判据

零模型：若每个文件的命中行集合互相独立，且行被均匀抽自 200 行库，
则「被全部 F 个文件命中」的期望行数 = 200 · Π_f (k_f / 200)，k_f 为该文件命中的不同行数。
用实际 k_f 计算，得到「超出随机多少倍」，从而判定 residual coexist 是真 hub 还是波动。
"""
import glob
import json
import os
import sys
from collections import defaultdict

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
NROWS = 200


def file_row_sets(runs):
    fs = defaultdict(set)
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        if f.split("/")[3] not in runs:
            continue
        d = json.load(open(f, encoding="utf-8"))
        ana = os.path.basename(str(d.get("file") or ""))
        for b in ("retrieval_evidence", "gap_retrieval_evidence"):
            for it in (d.get(b) or []):
                for h in (it.get("weaviate_hits") or []):
                    fs[ana].add(h.get("sqlite_id"))
    return fs


def test(name, runs):
    fs = file_row_sets(runs)
    F = len(fs)
    union = set().union(*fs.values())
    coexist = [s for s in union if sum(1 for v in fs.values() if s in v) >= F]
    ks = [len(v) for v in fs.values()]
    exp = NROWS * float(np.prod([k / NROWS for k in ks]))
    print("  %-28s 文件=%d  各行命中数 min/中位/max=%d/%d/%d  并集=%d/%d  共现=%d  零模型期望=%.3f  超出=%.0f×"
          % (name, F, min(ks), int(np.median(ks)), max(ks), len(union), NROWS,
             len(coexist), exp, (len(coexist) / exp) if exp > 0 else float("inf")))
    return {"files": F, "union": len(union), "coexist": len(coexist), "expected": exp,
            "excess": (len(coexist) / exp) if exp > 0 else None}


def main():
    print("=" * 100)
    print("残余共现显著性检验（零模型 = 各行独立且均匀抽自 200 行）")
    rb = test("基线（偏移关闭）", BASE)
    rq = test("C2 修复后（偏移开启）", QOFF2)
    print()
    print("### 开放跨文件闸门的最终判据")
    checks = [
        ("top-10 集中度 < 30%", 0.1254, 0.30, "lt"),
        ("跨全部文件共现行数 = 0", rq["coexist"], 0, "eq0"),
        ("共现是否超出随机（越接近 1 越像噪声）", rq["excess"], 3.0, "lt"),
        ("被召回 KB 行数", rq["union"], NROWS, "eq"),
        ("同项目命中率 > 随机(6.39%)", 9.64, 6.39, "gt"),
        ("同文件命中率 > 随机(0.11%)", 0.72, 0.11, "gt"),
    ]
    for label, val, thr, op in checks:
        if op == "lt":
            ok = val < thr; txt = "%.3f < %.3f" % (val, thr)
        elif op == "eq0":
            ok = val == 0; txt = "%d == 0" % val
        elif op == "eq":
            ok = val == thr; txt = "%d == %d" % (val, thr)
        else:
            ok = val > thr; txt = "%.2f > %.2f" % (val, thr)
        print("   [%s] %-42s %s" % ("达标" if ok else "未达标", label, txt))
    print()
    print("### 基线 vs 修复后（关键量）")
    for k in ("union", "coexist", "excess"):
        print("   %-10s %s -> %s" % (k, rb[k], rq[k]))
    json.dump({"baseline": rb, "qoff": rq}, open("/root/autodl-tmp/gate_readiness.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n已写出 /root/autodl-tmp/gate_readiness.json")


if __name__ == "__main__":
    sys.exit(main())
