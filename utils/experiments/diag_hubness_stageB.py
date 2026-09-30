#!/usr/bin/env python
"""阶段0 诊断 · 第3步（C1/C2 查询侧决定性测量）

Stage A 结论：索引侧白化后非对角相似度均值 ≈0.001~0.055、σ≈0.15（区分良好），
而未白化时均值高达 0.95（严重坍缩）。但生产环境「查询×索引」的 σ 只有 0.028。
=> 症结不在索引侧几何，而在【查询向量与索引不在同一分布】。

本步用真实记录的查询文本（从冒烟产物 retrieval_evidence / gap_retrieval_evidence
的 issue_description 取）离线复算，测：
  1. 查询-查询 平均余弦（坍缩检验：若查询都塌向同一方向，会显著 >0）
  2. 查询-索引 相似度 mean/σ（看是否复现生产的 0.028）
  3. top-5 hubness：N_5 分布 / 偏度 / anti-hub / top-10 集中度
  4. 跨文件共现行数（9 个文件都被同一 KB 行命中）—— 本步的核心目标指标（目标=0）
并对 α ∈ {0.25, 0.5, 1.0} 与 CSLS 各自评估，找出真正有效的修法。

只读。产物：/root/autodl-tmp/diag_hubness_stageB.json
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

KB_PATH = "/root/autodl-tmp/kb_vectors.json"
WH_PATH = os.path.join(ROOT, "infrastructure", "embeddings", "whitening_transform.json")
OUT = "/root/autodl-tmp/diag_hubness_stageB.json"
LAYERS = ["semantic", "code_pattern", "solution", "full"]
ALPHAS = [0.25, 0.5, 1.0]
K_TOPK = 5
CSLS_K = 10


def l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def skew(x):
    x = np.asarray(x, float)
    s = x.std()
    return 0.0 if s == 0 or x.size < 3 else float(((x - x.mean()) ** 3).mean() / s ** 3)


def load_queries():
    """收集本次冒烟 r2 全部真实查询文本 + 其所属文件/通道。"""
    qs = []
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        d = json.load(open(f, encoding="utf-8"))
        fn = os.path.basename(str(d.get("file") or ""))
        for bucket, ch in (("retrieval_evidence", "validation"), ("gap_retrieval_evidence", "gap")):
            for it in (d.get(bucket) or []):
                txt = str(it.get("issue_description") or "").strip()
                if txt:
                    qs.append({"text": txt, "file": fn, "channel": ch})
    return qs


def main():
    objs = json.load(open(KB_PATH, encoding="utf-8"))
    wh = json.load(open(WH_PATH, encoding="utf-8"))

    idx = defaultdict(list)
    for o in objs:
        p = o.get("properties") or {}
        v = (o.get("vectors") or {}).get("default") or []
        if v:
            idx[p.get("vector_layer")].append({
                "sid": p.get("sqlite_id"), "text": p.get("layer_text") or "",
                "stored": np.asarray(v, float)})

    bases = {}
    for L in LAYERS:
        W = np.asarray(wh[L]["W"], float)
        m = np.asarray(wh[L]["mean"], float)
        lam = 1.0 / np.clip(np.diag(W.T @ W), 1e-12, None)
        bases[L] = {"W": W, "mean": m, "V": W @ np.diag(np.sqrt(lam)), "lam": lam}

    queries = load_queries()
    uniq = {}
    for q in queries:
        uniq.setdefault(q["text"], q["file"])
    print("查询实例 %d，去重后 %d" % (len(queries), len(uniq)))
    by_ch = Counter(q["channel"] for q in queries)
    print("  通道分布:", dict(by_ch))
    files = sorted({q["file"] for q in queries})
    print("  文件数:", len(files))

    print("\n=== 编码查询文本（CPU）===")
    from infrastructure.embeddings.codebert_embedder import _forward_raw
    qtexts = list(uniq.keys())
    Qraw = []
    for i, t in enumerate(qtexts):
        Qraw.append(_forward_raw(t))
        if (i + 1) % 100 == 0:
            print("  已编码 %d / %d" % (i + 1, len(qtexts)))
    Qraw = np.asarray(Qraw, float)
    print("  完成:", Qraw.shape)
    json.dump({"qtexts": qtexts}, open("/root/autodl-tmp/_qtmp.json", "w"), ensure_ascii=False)
    np.save("/root/autodl-tmp/_qraw.npy", Qraw)

    # 查询-查询 坍缩检验（用全部查询，α 无关：原始向量即可说明是否同向）
    Qn_raw = l2norm(Qraw)
    Sqq_raw = Qn_raw @ Qn_raw.T
    iu = np.triu_indices(Qn_raw.shape[0], 1)
    print("\n=== A) 查询-查询 平均余弦（坍缩检验）===")
    print("  原始(未白化)查询间: mean=%.4f σ=%.4f  [min=%.4f max=%.4f]" % (
        Sqq_raw[iu].mean(), Sqq_raw[iu].std(), Sqq_raw[iu].min(), Sqq_raw[iu].max()))
    for L in LAYERS:
        Zq = l2norm((Qn_raw - bases[L]["mean"]) @ bases[L]["W"])
        S = Zq @ Zq.T
        print("  %-14s α=1 查询间: mean=%.4f σ=%.4f" % (L, S[iu].mean(), S[iu].std()))

    report = {}
    for a in ALPHAS:
        Zq = {}
        Zx = {}
        for L in LAYERS:
            Wa = bases[L]["V"] @ np.diag(bases[L]["lam"] ** (-a))
            Zq[L] = l2norm((Qn_raw - bases[L]["mean"]) @ Wa)
            X = l2norm(np.vstack([r["stored"][:bases[L]["W"].shape[1]] for r in idx[L]]))
            # stored 是布局在 k 维上的白化向量；重建同 α 的索引向量需重算原始，
            # 这里用 Stage A 已证「stored == α=1 结果」，故 α=1 直接用 stored，
            # 其它 α 由 W 变换关系换算：z_α = z_1 · diag(λ^(1-α)) 后重新 L2
            lam = bases[L]["lam"]
            Za = X * (lam ** (1.0 - a))
            Zx[L] = l2norm(Za)
        # CSLS 半径（索引侧，预计算）
        radius = {}
        for L in LAYERS:
            Sxx = Zx[L] @ Zx[L].T
            np.fill_diagonal(Sxx, -np.inf)
            kk = min(CSLS_K, Sxx.shape[0] - 1)
            part = np.partition(Sxx, -kk, axis=1)[:, -kk:]
            radius[L] = part.mean(axis=1)
        print("\n=== B) α=%.2f 查询-索引 ===" % a)
        rep = {"alpha": a}
        for L in LAYERS:
            S = Zq[L] @ Zx[L].T                      # (nq, 200)
            mx = S.max(axis=1)
            top = np.argsort(-S, axis=1)[:, :K_TOPK]
            N = np.zeros(S.shape[1])
            for row in top:
                for j in row:
                    N[j] += 1
            # 每文件命中集合 -> 跨文件共现
            perfile = defaultdict(set)
            qi = 0
            for t in qtexts:
                for j in top[qi]:
                    perfile[uniq[t]].add(int(j))
                qi += 1
            co = Counter()
            for fn, s in perfile.items():
                for j in s:
                    co[j] += 1
            allf = len(perfile)
            coexist = sum(1 for j, c in co.items() if c >= allf)
            od = np.sort(N)[::-1]
            rep[L] = {
                "q_x_mean": float(S.mean()), "q_x_std": float(S.std()),
                "per_query_std_mean": float(S.std(axis=1).mean()),
                "top1_mean": float(mx.mean()),
                "Nk_max": float(N.max()), "Nk_skew": skew(N),
                "anti_hubs": int((N == 0).sum()),
                "top10_share": float(od[:10].sum() / max(1.0, N.sum())),
                "distinct_rows_hit": int((N > 0).sum()),
                "crossfile_coexist": int(coexist),
                "files": allf,
            }
            r = rep[L]
            print("  %-14s 查询×索引 mean=%.4f σ=%.4f | 每查询σ̄=%.4f | top1=%.4f | N5_max=%.0f skew=%.2f anti=%d top10=%.3f | 命中行=%d/%d | 跨%d文件共现=%d"
                  % (L, r["q_x_mean"], r["q_x_std"], r["per_query_std_mean"], r["top1_mean"],
                     r["Nk_max"], r["Nk_skew"], r["anti_hubs"], r["top10_share"],
                     r["distinct_rows_hit"], S.shape[1], allf, r["crossfile_coexist"]))
        # CSLS
        print("  --- 叠加 CSLS（α=%.2f）---" % a)
        rep_csls = {}
        for L in LAYERS:
            S = Zq[L] @ Zx[L].T
            rq = np.sort(S, axis=1)[:, -CSLS_K:].mean(axis=1, keepdims=True)
            Sc = 2.0 * S - rq - radius[L][None, :]
            top = np.argsort(-Sc, axis=1)[:, :K_TOPK]
            N = np.zeros(Sc.shape[1])
            for row in top:
                for j in row:
                    N[j] += 1
            perfile = defaultdict(set)
            for t, row in zip(qtexts, top):
                for j in row:
                    perfile[uniq[t]].add(int(j))
            co = Counter()
            for fn, s in perfile.items():
                for j in s:
                    co[j] += 1
            allf = len(perfile)
            coexist = sum(1 for j, c in co.items() if c >= allf)
            od = np.sort(N)[::-1]
            rep_csls[L] = {"Nk_max": float(N.max()), "Nk_skew": skew(N),
                           "anti_hubs": int((N == 0).sum()),
                           "top10_share": float(od[:10].sum() / max(1.0, N.sum())),
                           "distinct_rows_hit": int((N > 0).sum()),
                           "crossfile_coexist": int(coexist), "files": allf}
            r = rep_csls[L]
            print("  %-14s CSLS: N5_max=%.0f skew=%.2f anti=%d top10=%.3f | 命中行=%d | 跨%d文件共现=%d"
                  % (L, r["Nk_max"], r["Nk_skew"], r["anti_hubs"], r["top10_share"],
                     r["distinct_rows_hit"], allf, r["crossfile_coexist"]))
        rep["csls"] = rep_csls
        report["alpha=%.2f" % a] = rep

    report["_meta"] = {"n_queries": len(qtexts), "files": len(files),
                       "channels": dict(by_ch), "k_topk": K_TOPK, "csls_k": CSLS_K}
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写出:", OUT)


if __name__ == "__main__":
    sys.exit(main())
