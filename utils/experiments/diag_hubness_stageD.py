#!/usr/bin/env python
"""阶段0 诊断 · 第5步（去查询均值：精度检验 + 甲/乙 方案判别）

Stage C 已证：白化后再去掉【查询集自身均值】能把查询间余弦 0.364→0.035、
top10 集中度 0.414→0.223、跨文件共现 6→1。但那只证明"排名被摊开了"，
**没有证明精度上升**（也可能只是把噪声摊匀）。

本步回答两件事：
  ① 精度检验：修复后 top-5 里"同文件 / 同项目(顶层目录)"的占比是否上升？
     并给出「随机命中」的期望占比作对照，避免把先验占比当成精度。
  ② 甲/乙 判别：查询均值若按【本 run 全部查询】拟合（乙），结果依赖批次组成；
     若按【离线拟合的全局偏移】（甲），是否同样有效？
     用留一文件法（leave-one-file-out）模拟"拟合集与评测集分离"，检验甲的泛化性。

复用已编码的 /root/autodl-tmp/_qraw.npy，不再编码。0 GPU。
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

KB = "/root/autodl-tmp/kb_vectors.json"
WH = os.path.join(ROOT, "infrastructure", "embeddings", "whitening_transform.json")
LAYERS = ["semantic", "code_pattern", "solution", "full"]
K = 5


def l2(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def norm_path(p):
    """把数据集的 a__b__c.c 与 KB 的 a/b/c.c 统一为 a/b/c.c"""
    s = str(p or "").replace("\\", "/").strip().lower()
    base = os.path.basename(s)
    if "__" in base:
        s = (os.path.dirname(s) + "/" if os.path.dirname(s) else "") + base.replace("__", "/")
    return s


def base_of(p):
    return os.path.basename(norm_path(p))


def proj_of(p):
    n = norm_path(p)
    return n.split("/")[0] if "/" in n else ""


def skew(x):
    x = np.asarray(x, float)
    s = x.std()
    return 0.0 if s == 0 or x.size < 3 else float(((x - x.mean()) ** 3).mean() / s ** 3)


def load_queries():
    qs = []
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        d = json.load(open(f, encoding="utf-8"))
        fn = os.path.basename(str(d.get("file") or ""))
        for b in ("retrieval_evidence", "gap_retrieval_evidence"):
            for it in (d.get(b) or []):
                t = str(it.get("issue_description") or "").strip()
                if t:
                    qs.append((t, fn))
    seen, out = set(), []
    for t, fn in qs:
        if t not in seen:
            seen.add(t)
            out.append((t, fn))
    return out


def score_block(Zq, Zx, qfiles, kbfile, kbproj):
    """返回逐命中统计"""
    S = Zq @ Zx.T
    top = np.argsort(-S, axis=1)[:, :K]
    nfile = nproj = tot = 0
    for (t, fn), row in zip(qfiles, top):
        fb, fp = base_of(fn), proj_of(fn)
        for j in row:
            tot += 1
            if kbproj[j] and kbproj[j] == fp and fp:
                nproj += 1
            if kbfile[j] and kbfile[j] == fb and fb:
                nfile += 1
    N = np.zeros(S.shape[1])
    for row in top:
        for j in row:
            N[j] += 1
    perfile = defaultdict(set)
    for (t, fn), row in zip(qfiles, top):
        for j in row:
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
        "hits": tot,
        "same_file": nfile / max(1, tot),
        "same_proj": nproj / max(1, tot),
        "top10_share": float(od[:10].sum() / max(1.0, N.sum())),
        "Nk_max": float(N.max()), "Nk_skew": skew(N),
        "anti_hubs": int((N == 0).sum()),
        "rows_hit": int((N > 0).sum()),
        "coexist": int(sum(1 for j, c in co.items() if c >= nf)),
        "files": nf,
        "qq_mean": float(Sqq[iu].mean()) if Zq.shape[0] > 1 else 0.0,
    }


def main():
    objs = json.load(open(KB, encoding="utf-8"))
    wh = json.load(open(WH, encoding="utf-8"))
    # 关键：迁移过来的 Weaviate KB 的 file_pattern/class_pattern 属性为空（0/800），
    # 必须回落到 SQLite 取真实值；否则精度度量的随机基线都会是 0。
    sq = sqlite3.connect(os.path.join(ROOT, "infrastructure", "database", "mas.db"))
    fp_by_sid = {str(r[0]): (r[1] or "") for r in
                 sq.execute("select id, file_pattern from issue_patterns").fetchall()}
    n_wv = sum(1 for o in objs if (o["properties"].get("file_pattern") or "").strip())
    print("Weaviate file_pattern 非空 %d/800；改用 SQLite 的 %d 条" % (n_wv, len(fp_by_sid)))
    idx = defaultdict(list)
    kbfile, kbproj = {}, {}
    for o in objs:
        p = o.get("properties") or {}
        v = (o.get("vectors") or {}).get("default") or []
        L = p.get("vector_layer")
        if not v:
            continue
        idx[L].append(np.asarray(v, float))
        real_fp = fp_by_sid.get(str(p.get("sqlite_id")), "") or (p.get("file_pattern") or "")
        kbfile[L] = kbfile.get(L, []) + [base_of(real_fp)]
        kbproj[L] = kbproj.get(L, []) + [proj_of(real_fp)]
    qraw = np.load("/root/autodl-tmp/_qraw.npy")
    qs = load_queries()
    assert qraw.shape[0] == len(qs), (qraw.shape, len(qs))
    Qn = l2(qraw)
    qfile = np.array([fn for _, fn in qs])
    files = sorted(set(qfile.tolist()))
    print("查询 %d，文件 %d" % (len(qs), len(files)))

    # 随机基线：若命中在 200 行上均匀随机，同文件/同项目占比的期望
    print("\n=== 随机基线（均匀随机的期望占比）===")
    for L in LAYERS:
        fs = [base_of(f) for f in files]
        pr = [proj_of(f) for f in files]
        ef = np.mean([sum(1 for b in kbfile[L] if b == b0) / 200.0 for b0 in fs])
        ep = np.mean([(sum(1 for q in kbproj[L] if q == p0) or 0) / 200.0 for p0 in pr])
        print("  %-14s 期望同文件=%.4f  期望同项目=%.4f" % (L, ef, ep))

    results = defaultdict(lambda: defaultdict(dict))
    for L in LAYERS:
        W = np.asarray(wh[L]["W"], float)
        m = np.asarray(wh[L]["mean"], float)
        kk = W.shape[1]
        X1 = l2(np.vstack(idx[L])[:, :kk])
        w_all = (Qn - m) @ W
        Zbase = l2(w_all)
        # A 基线
        results["A 基线(当前生产)"][L] = score_block(Zbase, X1, list(zip([q[0] for q in qs], qfile)),
                                                    kbfile[L], kbproj[L])
        # 乙：用本 run 全部查询拟合均值
        mu_run = w_all.mean(axis=0, keepdims=True)
        results["乙 本run均值(批次依赖)"][L] = score_block(
            l2(w_all - mu_run), X1, list(zip([q[0] for q in qs], qfile)), kbfile[L], kbproj[L])
        # 甲：留一文件法（拟合集排除被评测文件）
        rows, lbls = [], []
        for f in files:
            tr = qfile != f
            te = qfile == f
            if tr.sum() == 0 or te.sum() == 0:
                continue
            mu_ho = w_all[tr].mean(axis=0, keepdims=True)
            rows.append(l2(w_all[te] - mu_ho))
            lbls.extend(qfile[te].tolist())
        Zho = np.vstack(rows)
        results["甲 留一拟合全局偏移"][L] = score_block(
            Zho, X1, list(zip([""] * len(lbls), lbls)), kbfile[L], kbproj[L])
        # 甲2：留一 + α=1 之外不额外处理（对照：只在查询集上做，不碰索引）

    print("\n=== 精度与 hubness（按层）===")
    print("%-24s %-14s %8s %9s %9s %8s %8s %7s %7s" % (
        "方案", "层", "同文件%", "同项目%", "top10", "N5_max", "anti", "命中行", "共现"))
    for name in ("A 基线(当前生产)", "乙 本run均值(批次依赖)", "甲 留一拟合全局偏移"):
        for L in LAYERS:
            r = results[name][L]
            print("%-24s %-14s %7.2f%% %8.2f%% %9.3f %8.0f %8d %7d %7d" % (
                name, L, 100 * r["same_file"], 100 * r["same_proj"],
                r["top10_share"], r["Nk_max"], r["anti_hubs"], r["rows_hit"], r["coexist"]))
        print()

    print("=== 四层汇总（等权平均）===")
    print("%-24s %9s %9s %9s %8s %8s %7s" % ("方案", "同文件%", "同项目%", "top10", "anti", "命中行", "共现合计"))
    for name in results:
        rs = results[name]
        print("%-24s %8.2f%% %8.2f%% %9.3f %8.0f %8.0f %7d" % (
            name,
            100 * np.mean([rs[L]["same_file"] for L in LAYERS]),
            100 * np.mean([rs[L]["same_proj"] for L in LAYERS]),
            np.mean([rs[L]["top10_share"] for L in LAYERS]),
            np.mean([rs[L]["anti_hubs"] for L in LAYERS]),
            np.mean([rs[L]["rows_hit"] for L in LAYERS]),
            sum(rs[L]["coexist"] for L in LAYERS)))

    out = {n: {L: results[n][L] for L in LAYERS} for n in results}
    json.dump(out, open("/root/autodl-tmp/diag_hubness_stageD.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n已写出 /root/autodl-tmp/diag_hubness_stageD.json")


if __name__ == "__main__":
    sys.exit(main())
