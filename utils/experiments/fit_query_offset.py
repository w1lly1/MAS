#!/usr/bin/env python
"""拟合「查询偏移方向」产物（C2 修复 · 甲方案）

背景：白化用【索引均值】中心化。查询文本（C 代码块 / 中文 issue 描述）整体远离索引分布
（英文元数据散文），于是 (q − m_index) 的【偏移分量】成为主导；所有查询偏移方向近乎相同，
L2 后全部塌向同一方向 → 查询间余弦 0.36、top10 集中度 0.41、跨文件共现 6。

修法：白化后再减去【查询集自身的均值方向】 mu_layer（k 维），然后 L2。
  z = L2( (q − m_index)·W − mu_layer )

Stage D 已验证「留一文件拟合」与「本 run 全量拟合」效果几乎相同
（top10 0.225 vs 0.223、共现 1 vs 1、命中行 168 vs 167）→ 该方向跨文件稳定，
故可用全部查询拟合生产产物。

产物：infrastructure/embeddings/query_offset_transform.json
  { "<layer>": {"offset": [...k]}, "_meta": {...} }

注意：本产物用 smoke8 的 236 条查询拟合，样本偏小；正式实验前应在更大查询语料上重拟合。
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

WH = os.path.join(ROOT, "infrastructure", "embeddings", "whitening_transform.json")
OUT = os.path.join(ROOT, "infrastructure", "embeddings", "query_offset_transform.json")
LAYERS = ["semantic", "code_pattern", "solution", "full"]


def l2(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def collect_queries():
    txts, files = [], []
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        d = json.load(open(f, encoding="utf-8"))
        fn = os.path.basename(str(d.get("file") or ""))
        for b in ("retrieval_evidence", "gap_retrieval_evidence"):
            for it in (d.get(b) or []):
                t = str(it.get("issue_description") or "").strip()
                if t:
                    txts.append(t)
                    files.append(fn)
    seen, T = set(), []
    for t, fn in zip(txts, files):
        if t not in seen:
            seen.add(t)
            T.append((t, fn))
    return T


def main():
    T = collect_queries()
    print("查询数（去重）:", len(T))
    from infrastructure.embeddings.codebert_embedder import _forward_raw
    Q = l2(np.asarray([_forward_raw(t) for t, _ in T], float))
    wh = json.load(open(WH, encoding="utf-8"))
    out = {"_meta": {
        "purpose": "C2 修复：移除查询集共同的偏移方向，消除查询向量共线导致的 hubness",
        "fit_corpus": "reports/analysis/*/*/second_pass/consolidated/*_r2.json 的 issue_description",
        "n_queries": len(T),
        "n_files": len(set(f for _, f in T)),
        "formula": "z = L2( (q - mean_index) @ W - offset_layer )",
        "caveat": "样本偏小（smoke8 单批次）；正式实验前应在更大查询语料上重拟合",
    }}
    for L in LAYERS:
        W = np.asarray(wh[L]["W"], float)
        m = np.asarray(wh[L]["mean"], float)
        w = (Q - m) @ W
        mu = w.mean(axis=0)
        out[L] = {"offset": [float(x) for x in mu]}
        print("  %-14s k=%d  offset 范数=%.4f  查询间余弦: 前=%.4f 后=%.4f" % (
            L, W.shape[1], float(np.linalg.norm(mu)),
            float((l2(w) @ l2(w).T)[np.triu_indices(len(w), 1)].mean()),
            float((l2(w - mu) @ l2(w - mu).T)[np.triu_indices(len(w), 1)].mean())))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("已写出:", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    sys.exit(main())
