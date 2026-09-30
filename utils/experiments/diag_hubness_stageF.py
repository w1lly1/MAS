#!/usr/bin/env python
"""阶段0 · Stage F：验证「甲：离线拟合查询偏移」的跨批次泛化

用两组【不相交的真实查询向量】：
  A 组 = query_vectors_A.jsonl（2 项：CVE-2018-13301 / CVE-2014-6229）
  B 组 = query_vectors_B.jsonl（6 项：其余）
全部在 B 组上评测，三种变体直接可比：
  · 基线        ：不做偏移修正
  · 甲 跨批次    ：偏移在 A 组上拟合，应用到 B 组  ← 这是能否进生产的关键
  · 乙 oracle   ：偏移在 B 组自身拟合（上界，但有批次依赖）

指标：top10 集中度 / N_max / 被召回行数 / perQ_σ /
      同文件% 与 同项目%（KB file_pattern 取 SQLite 真值，含随机期望）/
      跨 B 组全部文件共现的 KB 行数。
"""
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

QA, QB = "/root/autodl-tmp/query_vectors_A.jsonl", "/root/autodl-tmp/query_vectors_B.jsonl"
KB = "/root/autodl-tmp/kb_vectors.json"
LAYERS = ["semantic", "code_pattern", "solution", "full"]
K = 5


def l2(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def skew(x):
    x = np.asarray(x, float)
    s = x.std()
    return 0.0 if s == 0 or x.size < 3 else float(((x - x.mean()) ** 3).mean() / s ** 3)


def norm_path(p):
    s = str(p or "").replace("\\", "/").strip().lower()
    b = os.path.basename(s)
    if "__" in b:
        s = (os.path.dirname(s) + "/" if os.path.dirname(s) else "") + b.replace("__", "/")
    return s


def base_of(p):
    return os.path.basename(norm_path(p))


def proj_of(p):
    n = norm_path(p)
    return n.split("/")[0] if "/" in n else ""


def load(path, need_file=False):
    by = defaultdict(dict)
    if not os.path.exists(path):
        return by, 0
    n = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        key = r["text_sha1"]
        if key in by[r["layer"]]:
            continue
        by[r["layer"]][key] = {"vec": np.asarray(r["vec"], float),
                               "file": r.get("file", "") or ""}
        n += 1
    return by, n


def evaluate(Zq, files, X, kbfp):
    S = Zq @ X.T
    top = np.argsort(-S, axis=1)[:, :K]
    N = Counter()
    rowfiles = defaultdict(set)
    nf = npr = tot = 0
    for qi, row in enumerate(top):
        fam = files[qi]
        for j in set(row.tolist()):
            N[int(j)] += 1
            rowfiles[int(j)].add(fam)
    for qi, row in enumerate(top):
        fb, fp = base_of(files[qi]), proj_of(files[qi])
        for j in row.tolist():
            fpx = kbfp[j]
            if not fpx:
                continue
            tot += 1
            if base_of(fpx) == fb:
                nf += 1
            if proj_of(fpx) and proj_of(fpx) == fp:
                npr += 1
    nv = np.array(list(N.values()), float) if N else np.array([0.0])
    od = np.sort(nv)[::-1]
    allf = len(set(files))
    coexist = sum(1 for s, fs in rowfiles.items() if len(fs) >= allf)
    return {
        "nq": len(Zq), "top10": float(od[:10].sum() / max(1.0, nv.sum())),
        "N_max": float(nv.max()), "N_skew": skew(nv), "rows": len(N),
        "perQ_std": float(S.std(axis=1).mean()), "qx_std": float(S.std()),
        "same_file": 100.0 * nf / max(1, tot), "same_proj": 100.0 * npr / max(1, tot),
        "coexist": coexist, "files": allf,
    }


def main():
    A, na = load(QA)
    B, nb = load(QB)
    print("A 组条目 %d   B 组条目 %d" % (na, nb))
    if nb == 0:
        raise SystemExit("B 组为空，请先跑完 smoke_dump6")

    sq = sqlite3.connect(os.path.join(ROOT, "infrastructure", "database", "mas.db"))
    fpm = {str(r[0]): (r[1] or "") for r in
           sq.execute("select id, file_pattern from issue_patterns").fetchall()}
    objs = json.load(open(KB, encoding="utf-8"))
    idx = defaultdict(list)
    for o in objs:
        p = o.get("properties") or {}
        v = (o.get("vectors") or {}).get("default") or []
        if v:
            idx[p.get("vector_layer")].append(np.asarray(v, float))
            # 索引向量与 kbfp 必须同序
    # 重建带 file_pattern 的索引矩阵
    Xby, kbfp_by = {}, {}
    for L in LAYERS:
        rows, fps = [], []
        for o in objs:
            p = o.get("properties") or {}
            v = (o.get("vectors") or {}).get("default") or []
            if v and p.get("vector_layer") == L:
                rows.append(np.asarray(v, float))
                fps.append(fpm.get(str(p.get("sqlite_id")), ""))
        Xby[L] = l2(np.vstack(rows))
        kbfp_by[L] = fps

    print("\n%-14s %-14s %8s %8s %8s %8s %9s %9s %8s %8s" % (
        "层", "变体", "nq", "top10", "N_max", "命中行", "perQ_σ", "同文件%", "同项目%", "共现"))
    out = {}
    for L in LAYERS:
        if not B.get(L) or not A.get(L):
            continue
        Bq = list(B[L].values())
        Zb = l2(np.asarray([x["vec"] for x in Bq], float))
        files = [x["file"] for x in Bq]
        Aq = l2(np.asarray([x["vec"] for x in A[L].values()], float))
        X = Xby[L]
        kbfp = kbfp_by[L]
        variants = {
            "基线": Zb,
            "甲 跨批次(A拟合)": l2(Zb - Aq.mean(axis=0, keepdims=True)),
            "乙 oracle(B自拟合)": l2(Zb - Zb.mean(axis=0, keepdims=True)),
        }
        out[L] = {}
        for name, Z in variants.items():
            m = evaluate(Z, files, X, kbfp)
            out[L][name] = m
            print("%-14s %-14s %8d %8.3f %8.0f %8d %9.4f %8.2f%% %8.2f%% %8d" % (
                L, name, m["nq"], m["top10"], m["N_max"], m["rows"],
                m["perQ_std"], m["same_file"], m["same_proj"], m["coexist"]))
        print()

    allfp = list(fpm.values())
    fs = sorted({x["file"] for L in B for x in B[L].values() if x["file"]})
    ef = float(np.mean([sum(1 for p in allfp if base_of(p) == base_of(f)) / len(allfp) for f in fs]))
    ep = float(np.mean([sum(1 for p in allfp if proj_of(p) and proj_of(p) == proj_of(f)) / len(allfp) for f in fs]))
    print("随机期望: 同文件 %.2f%%  同项目 %.2f%%" % (100 * ef, 100 * ep))

    print("\n=== 四层平均汇总 ===")
    print("%-22s %8s %8s %9s %9s %9s %8s" % ("变体", "top10", "N_max", "命中行", "perQ_σ", "同文件%", "共现"))
    for name in ("基线", "甲 跨批次(A拟合)", "乙 oracle(B自拟合)"):
        rs = [out[L][name] for L in out if name in out[L]]
        print("%-22s %8.3f %8.0f %9.0f %9.4f %9.2f%% %8d" % (
            name,
            np.mean([r["top10"] for r in rs]), np.mean([r["N_max"] for r in rs]),
            np.mean([r["rows"] for r in rs]), np.mean([r["perQ_std"] for r in rs]),
            np.mean([r["same_file"] for r in rs]), sum(r["coexist"] for r in rs)))
    json.dump(out, open("/root/autodl-tmp/diag_hubness_stageF.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n已写出 /root/autodl-tmp/diag_hubness_stageF.json")


if __name__ == "__main__":
    sys.exit(main())
