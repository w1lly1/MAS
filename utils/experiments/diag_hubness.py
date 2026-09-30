#!/usr/bin/env python
"""阶段0 诊断 · 第2步（C2 hubness 主诊断，索引侧）

目标：判定 hubness/空间坍缩是否由「全白化 + 秩亏估计」造成，并找出最优 α。

做法：
  1. 从 Weaviate 导出的 800 条 (layer_text, vector) 复原白化基：
       存储的 W 是 768×k；因 V 正交，λ_i = 1/(WᵀW)_ii，V_k = W·diag(√λ)
       于是任意 α 的白化基 W_α = V_k·diag(λ^(-α))
     α=1 ⇒ 全白化（当前生产口径）；α=0 ⇒ 仅投影到主子空间不做缩放。
  2. 用真实 layer_text 重算「原始（未白化）向量」——白化不可逆，必须重算。
  3. 对 α ∈ {0, 0.25, 0.5, 0.75, 1.0} 计算每层 200×200 余弦相似度矩阵，输出：
       - 非对角相似度 mean/σ/min/max   （判别力：σ 越大越好）
       - hubness：N_5 的 max / 偏度 / anti-hub 数 / top-10 集中度
       - 可分性 AUC（同 error_type 对 vs 异 error_type 对；同 framework 同理）
  4. 校验：α=1 重算的向量应与 Weaviate 存储向量一致（证明复原正确）。

只读，不改向量库。产物：/root/autodl-tmp/diag_hubness_stageA.json
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

# 以脚本方式运行时，sys.path[0] 是脚本目录，需补上仓库根才能 import infrastructure.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

KB_PATH = "/root/autodl-tmp/kb_vectors.json"
WH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "infrastructure", "embeddings", "whitening_transform.json",
)
OUT = "/root/autodl-tmp/diag_hubness_stageA.json"
LAYERS = ["semantic", "code_pattern", "solution", "full"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
K_HUB = 5


def l2norm(M, axis=1):
    n = np.linalg.norm(M, axis=axis, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def skewness(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    m, s = x.mean(), x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / s ** 3)


def auc(pos, neg):
    """P(pos > neg) + 0.5·P(pos == neg)，用秩统计。"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1.0
    # 处理并列：平均秩
    order = allv.argsort()
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    rp = r[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def evaluate(X, groups, k=K_HUB):
    """X: (n,d) 已 L2 归一。groups: dict[str, list[Any]] 用于可分性。"""
    n = X.shape[0]
    S = X @ X.T
    iu = np.triu_indices(n, 1)
    off = S[iu]
    Sd = S.copy()
    np.fill_diagonal(Sd, -np.inf)
    nn = np.argsort(-Sd, axis=1)[:, :k]
    N = np.zeros(n, dtype=float)
    for row in nn:
        for j in row:
            N[j] += 1.0
    order_desc = np.sort(N)[::-1]
    top10 = float(order_desc[:10].sum() / max(1.0, N.sum()))
    res = {
        "off_mean": float(off.mean()),
        "off_std": float(off.std()),
        "off_min": float(off.min()),
        "off_max": float(off.max()),
        "off_p05": float(np.percentile(off, 5)),
        "off_p50": float(np.percentile(off, 50)),
        "off_p95": float(np.percentile(off, 95)),
        "Nk_max": float(N.max()),
        "Nk_std": float(N.std()),
        "Nk_skew": skewness(N),
        "anti_hubs": int((N == 0).sum()),
        "top10_share": top10,
    }
    for gname, gvals in groups.items():
        g = np.asarray(gvals)
        same = g[iu[0]] == g[iu[1]]
        pos, neg = off[same], off[~same]
        res[f"auc_{gname}"] = auc(pos, neg)
        res[f"npos_{gname}"] = int(pos.size)
    return res


def main():
    print("=== 1) 载入数据 ===")
    objs = json.load(open(KB_PATH, encoding="utf-8"))
    wh = json.load(open(WH_PATH, encoding="utf-8"))
    print("  对象数:", len(objs), " 白化层:", sorted(wh.keys()))
    for L in LAYERS:
        e = wh.get(L) or {}
        W = np.asarray(e.get("W") or [], dtype=np.float64)
        m = np.asarray(e.get("mean") or [], dtype=np.float64)
        print("    %-14s mean=%d  W.shape=%s" % (L, m.size, W.shape))

    by_layer = defaultdict(list)
    for o in objs:
        p = o.get("properties") or {}
        vec = (o.get("vectors") or {}).get("default") or []
        if not vec:
            continue
        by_layer[p.get("vector_layer")].append({
            "sqlite_id": p.get("sqlite_id"),
            "layer_text": p.get("layer_text") or "",
            "file_pattern": p.get("file_pattern") or "",
            "error_type": p.get("error_type") or "",
            "framework": p.get("framework") or "",
            "language": p.get("language") or "",
            "stored": np.asarray(vec, dtype=np.float64),
        })
    for L in LAYERS:
        print("    %-14s 对象=%d" % (L, len(by_layer[L])))

    print("\n=== 2) 重算原始（未白化）向量（CPU，800 条）===")
    from infrastructure.embeddings.codebert_embedder import _forward_raw
    cache = {}
    done = 0
    for L in LAYERS:
        for r in by_layer[L]:
            t = r["layer_text"]
            if t not in cache:
                cache[t] = _forward_raw(t)
            r["raw"] = np.asarray(cache[t] or [0.0] * 768, dtype=np.float64)
            done += 1
            if done % 200 == 0:
                print("    已编码 %d / 800" % done)
    print("    唯一文本数:", len(cache))

    print("\n=== 3) 复原白化基并校验 α=1 ===")
    bases = {}
    for L in LAYERS:
        e = wh[L]
        W = np.asarray(e["W"], dtype=np.float64)          # (768, k)
        m = np.asarray(e["mean"], dtype=np.float64)       # (768,)
        G = W.T @ W
        lam = 1.0 / np.clip(np.diag(G), 1e-12, None)      # (k,)
        V = W @ np.diag(np.sqrt(lam))                     # (768, k) 近似正交
        colnorm = np.linalg.norm(V, axis=0)
        bases[L] = {"mean": m, "V": V, "lam": lam, "W": W}
        raw = np.vstack([r["raw"] for r in by_layer[L]])
        Rn = l2norm(raw)
        z = l2norm((Rn - m) @ W)
        stored = np.vstack([r["stored"] for r in by_layer[L]])
        # 存储向量是 768 维零填充；比较前 k 维
        k = W.shape[1]
        d = np.abs(np.abs(z[:, :k]) - np.abs(stored[:, :k])).max()
        cos = float(np.mean(np.sum(z[:, :k] * stored[:, :k], axis=1)))
        print("    %-14s k=%-4d V列范数[%.4f,%.4f]  α=1 最大分量误差=%.2e 余弦=%.6f"
              % (L, k, colnorm.min(), colnorm.max(), d, cos))
        bases[L]["k"] = k

    print("\n=== 4) α 扫描：hubness 与可分性 ===")
    report = {}
    hdr = ("层/α", "off_σ", "off_mean", "N5_max", "N5_skew", "anti", "top10", "AUC_etype", "AUC_fw")
    print("  %-22s %8s %9s %8s %9s %6s %7s %11s %8s" % hdr)
    for L in LAYERS:
        raw = np.vstack([r["raw"] for r in by_layer[L]])
        Rn = l2norm(raw)
        groups = {
            "etype": [r["error_type"] for r in by_layer[L]],
            "fw": [r["framework"] for r in by_layer[L]],
        }
        report[L] = {}
        k = bases[L]["k"]
        for a in ALPHAS:
            W_a = bases[L]["V"] @ np.diag(bases[L]["lam"] ** (-a))
            Z = l2norm((Rn - bases[L]["mean"]) @ W_a)
            r = evaluate(Z, groups)
            report[L]["alpha=%.2f" % a] = r
            print("  %-22s %8.4f %9.4f %8.1f %9.3f %6d %7.3f %11.4f %8.4f" % (
                "%s a=%.2f" % (L, a), r["off_std"], r["off_mean"], r["Nk_max"],
                r["Nk_skew"], r["anti_hubs"], r["top10_share"],
                r.get("auc_etype", float("nan")), r.get("auc_fw", float("nan"))))
        # 基线：完全不用白化（α 任意都含「投影」，故另算纯原始向量）
        Z0 = Rn
        r0 = evaluate(Z0, groups)
        report[L]["raw_no_whitening"] = r0
        print("  %-22s %8.4f %9.4f %8.1f %9.3f %6d %7.3f %11.4f %8.4f" % (
            "%s 原始(无白化)" % L, r0["off_std"], r0["off_mean"], r0["Nk_max"],
            r0["Nk_skew"], r0["anti_hubs"], r0["top10_share"],
            r0.get("auc_etype", float("nan")), r0.get("auc_fw", float("nan"))))
        print()

    report["_meta"] = {
        "n_per_layer": {L: len(by_layer[L]) for L in LAYERS},
        "k": {L: int(bases[L]["k"]) for L in LAYERS},
        "alphas": ALPHAS,
        "k_hub": K_HUB,
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已写出:", OUT)


if __name__ == "__main__":
    sys.exit(main())
