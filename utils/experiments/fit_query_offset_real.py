#!/usr/bin/env python
"""C2 修复：用【真实生产查询向量】重拟合 query_offset_transform.json

与 fit_query_offset.py 的区别：后者用记录里的 issue_description（代理文本）拟合，
分布与生产查询（tag-soup）不符，导致端到端 A/B 反向恶化。
本脚本改为读取 agent 落盘的真实查询向量。

拟合语料：/root/autodl-tmp/query_vectors_A.jsonl + query_vectors_B.jsonl
        （A = CVE-2018-13301 / CVE-2014-6229；B = 其余 6 项；两组不相交）

跨批次泛化证据（Stage F）：在 A 上拟合、应用到 B，top10 0.718→0.260、N_max 132→33、
被召回行 60→143/200、同文件精度 0.14%→0.49%（随机 0.07%），达到 oracle 的约 85%。
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

DUMP = ["/root/autodl-tmp/query_vectors_A.jsonl", "/root/autodl-tmp/query_vectors_B.jsonl"]
WH = os.path.join(ROOT, "infrastructure", "embeddings", "whitening_transform.json")
OUT = os.path.join(ROOT, "infrastructure", "embeddings", "query_offset_transform.json")
LAYERS = ["semantic", "code_pattern", "solution", "full"]


def main():
    wh = json.load(open(WH, encoding="utf-8"))
    k_of = {L: np.asarray(wh[L]["W"], float).shape[1] for L in LAYERS}
    vecs = defaultdict(list)
    files = defaultdict(set)
    n = 0
    for p in DUMP:
        if not os.path.exists(p):
            print("  跳过（不存在）:", p)
            continue
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            L = r["layer"]
            v = r["vec"]
            if L not in k_of:
                continue
            vecs[L].append([float(x) for x in v[: k_of[L]]])
            if r.get("file"):
                files[L].add(r["file"])
            n += 1
    print("读取真实查询条目 %d" % n)
    out = {"_meta": {
        "purpose": "C2 修复：移除查询向量共同的偏移方向，消除查询共线导致的 hubness",
        "fit_corpus": "agent 落盘的真实查询向量（query_vectors_A.jsonl + query_vectors_B.jsonl）",
        "fit_source": "utils/experiments/fit_query_offset_real.py",
        "n_entries": n,
        "n_per_layer": {L: len(vecs[L]) for L in LAYERS},
        "formula": "z = L2( (q - mean_index) @ W - offset_layer )",
        "cross_batch_evidence": "Stage F：A 组拟合→B 组应用，top10 0.718→0.260，N_max 132→33，命中行 60→143/200，同文件精度 0.14%→0.49%（随机 0.07%）",
        "caveat": "样本为 8 个 BigVul 文件；正式实验前应在更大查询语料（如全量 400 CVE）上重拟合",
    }}
    for L in LAYERS:
        if not vecs[L]:
            continue
        M = np.asarray(vecs[L], float)
        off = M.mean(axis=0)
        out[L] = {"offset": [float(x) for x in off]}
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        iu = np.triu_indices(Mn.shape[0], 1)
        pre = float((Mn @ Mn.T)[iu].mean())
        Z = M - off
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        post = float((Z @ Z.T)[iu].mean())
        print("  %-14s n=%-4d k=%-3d |offset|=%.4f  查询间余弦 前=%.4f 后=%.4f" % (
            L, len(vecs[L]), k_of[L], float(np.linalg.norm(off)), pre, post))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("已写出:", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    sys.exit(main())
