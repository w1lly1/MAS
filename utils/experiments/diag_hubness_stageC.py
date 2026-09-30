#!/usr/bin/env python
"""阶段0 诊断 · 第4步（C2 修法筛选，离线，无 GPU）

Stage A/B 已证：
  · 索引侧同一变换下 hubness 很低（top10≈0.12、anti-hub 4~12/200、index-index 余弦≈0.00~0.055）
  · 但查询侧 hubness 极高（top10 0.28~0.87、anti-hub 44~162/236）
  · 且【查询间余弦 0.20~0.51】>>【索引间余弦 0.0002~0.055】
=> 查询向量彼此几乎共线（塌向同一方向），所以排名近似与查询无关 → 同一批行被反复命中。

本步比较几种「去共线」修法的效果（复用已编码的查询向量，不再重新编码）：
  A 基线（当前生产）：z = L2((q − m_idx)·W)
  B 白化后再去查询集均值：z = L2(w − mean_w)
  C 查询侧不做索引均值中心化：z = L2(q·W)
  D 原始空间先去查询集均值：z = L2((q − mean_q)·W)
  F C 的 α=0.75 版本
指标：查询间余弦、查询×索引 σ、top10 集中度、N5_max/skew、anti-hub、命中行数、跨9文件共现行数。
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

KB = "/root/autodl-tmp/kb_vectors.json"
WH = os.path.join(ROOT, "infrastructure", "embeddings", "whitening_transform.json")
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


def load_q():
    qs = []
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        d = json.load(open(f, encoding="utf-8"))
        fn = os.path.basename(str(d.get("file") or ""))
        for b in ("retrieval_evidence", "gap_retrieval_evidence"):
            for it in (d.get(b) or []):
                t = str(it.get("issue_description") or "").strip()
                if t:
                    qs.append((t, fn))
    seen, out = {}, []
    for t, fn in qs:
        if t not in seen:
            seen[t] = fn
            out.append((t, fn))
    return out


def metrics(Zq, Zx, qfiles, allfiles):
    S = Zq @ Zx.T
    top = np.argsort(-S, axis=1)[:, :K]
    N = np.zeros(S.shape[1])
    for r in top:
        for j in r:
            N[j] += 1
    perfile = defaultdict(set)
    for (t, fn), r in zip(qfiles, top):
        for j in r:
            perfile[fn].add(int(j))
    co = Counter()
    for fn, s in perfile.items():
        for j in s:
            co[j] += 1
    nf = len(perfile)
    od = np.sort(N)[::-1]
    Sqq = Zq @ Zq.T
    iu = np.triu_indices(Zq.shape[0], 1)
    return {
        "qq_mean": float(Sqq[iu].mean()),
        "qx_std": float(S.std()),
        "qx_mean": float(S.mean()),
        "top10_share": float(od[:10].sum() / max(1.0, N.sum())),
        "Nk_max": float(N.max()), "Nk_skew": skew(N),
        "anti_hubs": int((N == 0).sum()),
        "rows_hit": int((N > 0).sum()),
        "coexist": int(sum(1 for j, c in co.items() if c >= nf)),
        "files": nf,
    }


def main():
    objs = json.load(open(KB, encoding="utf-8"))
    wh = json.load(open(WH, encoding="utf-8"))
    idx = defaultdict(list)
    for o in objs:
        p = o.get("properties") or {}
        v = (o.get("vectors") or {}).get("default") or []
        if v:
            idx[p.get("vector_layer")].append(np.asarray(v, float))
    qraw = np.load("/root/autodl-tmp/_qraw.npy")
    qs = load_q()
    assert qraw.shape[0] == len(qs), (qraw.shape, len(qs))
    qf = [(t, fn) for t, fn in qs]
    files = {fn for _, fn in qs}
    print("查询去重 %d，文件 %d" % (len(qs), len(files)))

    Qn = l2(qraw)
    qmean = Qn.mean(axis=0)
    print("\n%-6s %-14s %8s %8s %8s %8s %7s %6s %7s %9s" % (
        "变体", "层", "QQ余弦", "QX_σ", "top10", "N5_max", "anti", "命中行", "跨文件共现", "N5_skew"))
    res = {}
    for L in LAYERS:
        W = np.asarray(wh[L]["W"], float)
        m = np.asarray(wh[L]["mean"], float)
        lam = 1.0 / np.clip(np.diag(W.T @ W), 1e-12, None)
        V = W @ np.diag(np.sqrt(lam))
        kk = W.shape[1]
        X1 = l2(np.vstack(idx[L])[:, :kk])

        def variant(name, Zq, Zx):
            r = metrics(Zq, Zx, qf, files)
            res.setdefault(name, {})[L] = r
            print("%-6s %-14s %8.4f %8.4f %8.3f %8.0f %7d %6d %9d %9.2f" % (
                name, L, r["qq_mean"], r["qx_std"], r["top10_share"], r["Nk_max"],
                r["anti_hubs"], r["rows_hit"], r["coexist"], r["Nk_skew"]))

        variant("A", l2((Qn - m) @ W), X1)                    # 基线
        w = (Qn - m) @ W
        variant("B", l2(w - w.mean(axis=0, keepdims=True)), X1)   # 白化后去查询均值
        variant("C", l2(Qn @ W), X1)                          # 不做索引均值中心化
        variant("D", l2((Qn - qmean) @ W), X1)                # 原始空间去查询均值
        W75 = V @ np.diag(lam ** (-0.75))
        X75 = l2(X1 * (lam ** 0.25))
        variant("F", l2((Qn - m) @ W75), X75)                 # α=0.75 基线中心化
        print()

    json.dump(res, open("/root/autodl-tmp/diag_hubness_stageC.json", "w"),
              ensure_ascii=False, indent=1)
    print("已写出 /root/autodl-tmp/diag_hubness_stageC.json")

    print("\n=== 汇总：各变体在 4 层上的平均 top10 集中度 / 跨文件共现 / 命中行数 ===")
    for name in sorted(res):
        rs = res[name]
        t10 = np.mean([rs[L]["top10_share"] for L in LAYERS])
        co = sum(rs[L]["coexist"] for L in LAYERS)
        rh = np.mean([rs[L]["rows_hit"] for L in LAYERS])
        qq = np.mean([rs[L]["qq_mean"] for L in LAYERS])
        print("  %-4s 平均top10=%.3f  共现合计=%2d  平均命中行=%.0f/200  平均QQ余弦=%.4f" % (
            name, t10, co, rh, qq))


if __name__ == "__main__":
    sys.exit(main())
