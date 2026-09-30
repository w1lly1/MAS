#!/usr/bin/env python
"""阶段0 · Stage E：用【真实生产查询向量】复核「查询共线」诊断

背景：此前 Stage B/C/D 用的是代理查询文本（记录的 issue_description），
而生产查询是 tag-soup（analysis_type/line_number/sig:<sha256>/file:/snippet:/ext:）。
用代理拟合的偏移在端到端 A/B 中把 hubness 显著恶化（top10 0.508→0.766，
被召回行 138→22），证明代理结论不可用。

本步改用 agent 落盘的真实查询向量（/root/autodl-tmp/query_vectors.jsonl）：
  1. 真实查询间余弦（vs 索引间余弦）—— 判定共线诊断是否成立
  2. 真实查询×索引 的每查询 σ —— 判别力
  3. top-5 hubness（N_max / top10 集中度 / 被召回行数）
  4. 用真实查询集均值去除共同方向，测「可达上界」（oracle 乙），
     以判断值得不值得再拟合一次正确分布的偏移
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

QL = "/root/autodl-tmp/query_vectors.jsonl"
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


def topk_metrics(Zq, Zx):
    S = Zq @ Zx.T
    top = np.argsort(-S, axis=1)[:, :K]
    N = Counter()
    for r in top:
        for j in set(r.tolist()):
            N[int(j)] += 1
    nv = np.array(list(N.values()), float) if N else np.array([0.0])
    od = np.sort(nv)[::-1]
    return {
        "qx_mean": float(S.mean()), "qx_std": float(S.std()),
        "per_q_std": float(S.std(axis=1).mean()),
        "N_max": float(nv.max()), "N_skew": skew(nv),
        "anti": int(len(Zx) - len(N)),
        "rows": len(N),
        "top10": float(od[:10].sum() / max(1.0, nv.sum())),
    }


def main():
    if not os.path.exists(QL):
        raise SystemExit("未找到 %s，请先跑含 dump_query_vectors=true 的运行" % QL)
    raw = [json.loads(l) for l in open(QL, encoding="utf-8") if l.strip()]
    print("落盘查询条目: %d" % len(raw))
    by_layer = defaultdict(list)
    sha_by_layer = defaultdict(list)
    for r in raw:
        by_layer[r["layer"]].append(r["vec"])
        sha_by_layer[r["layer"]].append(r["text_sha1"])
    for L in LAYERS:
        uniq = len(set(sha_by_layer.get(L, [])))
        print("  %-14s 条目=%-6d 唯一查询=%-6d 维度=%d" % (
            L, len(by_layer.get(L, [])), uniq,
            len(by_layer[L][0]) if by_layer.get(L) else 0))

    objs = json.load(open(KB, encoding="utf-8"))
    idx = defaultdict(list)
    for o in objs:
        p = o.get("properties") or {}
        v = (o.get("vectors") or {}).get("default") or []
        if v:
            idx[p.get("vector_layer")].append(np.asarray(v, float))

    print("\n%-14s %-28s %-28s" % ("层", "真实查询间余弦", "索引间余弦"))
    for L in LAYERS:
        if not by_layer.get(L):
            continue
        Q = l2(np.asarray(by_layer[L], float))
        iu = np.triu_indices(Q.shape[0], 1)
        Sqq = Q @ Q.T
        kk = min(len(idx[L][0]), Q.shape[1])
        X = l2(np.vstack(idx[L])[:, :kk])
        Sxx = X @ X.T
        iu2 = np.triu_indices(X.shape[0], 1)
        print("%-14s mean=%.4f σ=%.4f min=%.4f max=%.4f   mean=%.4f σ=%.4f" % (
            L, Sqq[iu].mean(), Sqq[iu].std(), Sqq[iu].min(), Sqq[iu].max(),
            Sxx[iu2].mean(), Sxx[iu2].std()))

    print("\n=== top-5 hubness：真实查询 vs 去掉查询集均值（oracle 上界）===")
    print("%-14s %-22s %8s %8s %8s %8s %8s" % ("层", "变体", "qx_σ", "perQ_σ", "N_max", "top10", "命中行"))
    res = {}
    for L in LAYERS:
        if not by_layer.get(L):
            continue
        Q = l2(np.asarray(by_layer[L], float))
        kk = min(Q.shape[1], len(idx[L][0]))
        X = l2(np.vstack(idx[L])[:, :kk])
        Qk = Q[:, :kk]
        for name, Zq in (("原始", Qk), ("去查询集均值(oracle)", l2(Qk - Qk.mean(axis=0, keepdims=True)))):
            m = topk_metrics(Zq, X)
            res.setdefault(name, {})[L] = m
            print("%-14s %-22s %8.4f %8.4f %8.0f %8.3f %8d" % (
                L, name, m["qx_std"], m["per_q_std"], m["N_max"], m["top10"], m["rows"]))
        print()

    print("=== 汇总（四层平均）===")
    for name in res:
        rs = res[name]
        print("  %-22s top10=%.3f  N_max=%.0f  命中行=%.0f/200  perQ_σ=%.4f" % (
            name,
            np.mean([rs[L]["top10"] for L in rs]),
            np.mean([rs[L]["N_max"] for L in rs]),
            np.mean([rs[L]["rows"] for L in rs]),
            np.mean([rs[L]["per_q_std"] for L in rs])))

    json.dump(res, open("/root/autodl-tmp/diag_hubness_stageE.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n已写出 /root/autodl-tmp/diag_hubness_stageE.json")


if __name__ == "__main__":
    sys.exit(main())
