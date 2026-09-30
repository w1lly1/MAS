# -*- coding: utf-8 -*-
"""逐视图 recall@1 复现 + 白化消融（语义通道逐视图首位命中率，论文表5/表6）。

协议（= 论文/whiten_distilbert.py 在其原始运行时的实际行为，见下方说明）：
  - 库侧：kb 组 issue_patterns 的四层文本（semantic/code_pattern/solution/full）的 distilbert 嵌入；
  - 查询侧：kb 条目对应的 before 错误代码片段（curated_issues.code_snippet，MAX 每 pattern_id）；
  - 检索：n 条目上穷举余弦，recall@1 = 正确条目排第 1 的比例；
  - 白化：对每层库向量做 PCA(95% 方差) 白化，查询用同一层变换（per-layer 一致）。

关于 whiten_distilbert.py 的已知缺陷（重要）：
  该脚本 21:42 首次运行时尚无 whitening_transform.json，故其 embed_text() 实际返回"未白化"向量；
  因此它的"白化前"= 真·未白化，它的"白化后"= 真·单次 PCA 白化（库与查询同层变换一致），
  论文表 5/6 的 10.5/12.0/65.5/56.0 即取自那次"白化后"。
  但随后 whiten_prepare.py 生成了 whitening_transform.json，使 embed_text() 默认带白化；
  再跑 whiten_distilbert.py 时，其 query 用 layer="full" 嵌入、库用各自 layer 嵌入，造成
  "查询用 full 变换、库用本层变换"的跨层失配（见本文本 whitened_query_full 模式），
  以及"白化前=一次白化 / 白化后=两次白化"的标签错乱。故本脚本不直接采用该脚本现数字。

三种嵌入模式：
  - raw（白化关）       : get_embedder()._forward_raw(text) —— mean pooling(去 CLS)+L2，768 维，未白化。
  - whitened（白化开）  : embed_text(text, layer) —— _forward_raw 后 (v-mean)@W 再 L2，零填充回 768；
                          库=embed_text(lt[L], L)，查询=embed_text(snip, L)（同层一致）。
  - whitened_query_full : 复刻 whiten_distilbert.py 现 bug——库=embed_text(lt[L], L)，
                          查询=embed_text(snip, "full")（跨层失配），仅作证据留存。

运行（MAS 根目录，离线 CPU）：
    python utils/experiments/view_recall1_whitening_ablation.py --db mas.db
    python utils/experiments/view_recall1_whitening_ablation.py --db mas_seed2025.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from infrastructure.embeddings.codebert_embedder import (  # noqa: E402
    _apply_whitening,
    embed_text,
    get_embedder,
)

LAYERS = ["semantic", "code_pattern", "solution", "full"]


def build_layer_texts(p):
    return {
        "semantic": "\n".join([
            f"[error_type] {p.get('error_type') or ''}",
            f"[severity] {p.get('severity') or ''}",
            f"[language] {p.get('language') or ''}",
            f"[framework] {p.get('framework') or ''}",
            f"[description] {p.get('error_description') or ''}",
        ]),
        "code_pattern": "\n".join([
            f"[problematic_pattern] {p.get('problematic_pattern') or ''}",
            f"[file_pattern] {p.get('file_pattern') or ''}",
            f"[class_pattern] {p.get('class_pattern') or ''}",
            f"[language] {p.get('language') or ''}",
        ]),
        "solution": "\n".join([
            f"[solution] {p.get('solution') or ''}",
            f"[error_description] {p.get('error_description') or ''}",
            f"[severity] {p.get('severity') or ''}",
        ]),
        "full": "\n".join([
            f"[error_type] {p.get('error_type') or ''}",
            f"[severity] {p.get('severity') or ''}",
            f"[language] {p.get('language') or ''}",
            f"[framework] {p.get('framework') or ''}",
            f"[description] {p.get('error_description') or ''}",
            f"[pattern] {p.get('problematic_pattern') or ''}",
            f"[solution] {p.get('solution') or ''}",
            f"[file_pattern] {p.get('file_pattern') or ''}",
            f"[class_pattern] {p.get('class_pattern') or ''}",
        ]),
    }


def load_data(db_path: Path):
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    pats = [dict(r) for r in con.execute(
        "SELECT id, title, error_type, severity, language, framework, error_description, "
        "problematic_pattern, solution, file_pattern, class_pattern FROM issue_patterns"
    )]
    snips = {pid: (s or "") for pid, s in con.execute(
        "SELECT pattern_id, MAX(code_snippet) FROM curated_issues "
        "WHERE code_snippet IS NOT NULL GROUP BY pattern_id"
    )}
    con.close()
    pats = [p for p in pats if (snips.get(p["id"]) or "").strip()]
    return pats, snips


def evaluate(lib_vecs, q_vecs, ids):
    """穷举余弦。lib_vecs/q_vecs 均需 L2 归一。返回 (recall, correct_sims, wrong_sims)。"""
    n = len(ids)
    recall = 0
    correct = []
    wrong = []
    for i in range(n):
        sims = sorted([(float(np.dot(q_vecs[i], lib_vecs[j])), ids[j]) for j in range(n)], reverse=True)
        if sims[0][1] == ids[i]:
            recall += 1
        correct.append(float(np.dot(q_vecs[i], lib_vecs[i])))
        wrong.extend([float(np.dot(q_vecs[i], lib_vecs[j])) for j in range(n) if j != i])
    return recall, correct, wrong


def l2(v):
    nrm = np.linalg.norm(v)
    return v / nrm if nrm else v


def pca_whiten_params(X):
    """PCA(95% 方差) 白化参数（与 whiten_prepare.py / whiten_distilbert.py 同式，改用 eigh 更稳）。

    cov 为对称半正定矩阵，用 eigh（按特征值降序）等价于 svd，但数值上更稳定，
    避免 mas_seed2025.db 上 svd 不收敛的问题。
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = Xc.T @ Xc / max(X.shape[0] - 1, 1)
    S, U = np.linalg.eigh(cov)          # 升序
    order = np.argsort(S)[::-1]         # 降序
    S = np.clip(S[order], 0.0, None)
    U = U[:, order]
    cum = S.cumsum() / S.sum()
    k = int((cum <= 0.95).sum()) + 1
    k = max(1, min(k, len(S)))
    W = U[:, :k] / np.sqrt(S[:k] + 1e-8)
    return mean, W, k


def apply_wht(X, mean, W):
    Y = X - mean
    Y = Y @ W
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    return Y / np.where(nrm == 0, 1.0, nrm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mas.db",
                    help="mas.db（seed=2024，whiten_distilbert.py 所用）或 mas_seed2025.db（seed=2025）")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import os
    import torch
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    if args.db == "mas.db":
        db_path = ROOT / "infrastructure" / "database" / "mas.db"
    elif args.db == "mas_seed2025.db":
        db_path = ROOT / "reports" / "mas_seed2025.db"
    else:
        db_path = Path(args.db)

    pats, snips = load_data(db_path)
    n = len(pats)
    ids = [p["id"] for p in pats]
    print(f"数据库: {db_path}")
    print(f"有效 kb 条目（有非空 code_snippet）: {n}\n", flush=True)

    emb = get_embedder()

    # 原始前向（白化关），按文本缓存
    raw_cache = {}
    def get_raw(text):
        if text not in raw_cache:
            v = emb._forward_raw(text)
            if v is None:
                raise RuntimeError("distilbert 未就绪")
            raw_cache[text] = np.array(v, dtype=np.float64)
        return raw_cache[text]

    # 库四层文本 + 查询文本
    lib_texts = {L: [] for L in LAYERS}
    q_texts = []
    for p in pats:
        lt = build_layer_texts(p)
        for L in LAYERS:
            lib_texts[L].append(lt[L])
        q_texts.append(snips[p["id"]])

    # 原始向量
    lib_raw = {L: np.array([get_raw(t) for t in lib_texts[L]]) for L in LAYERS}  # (n,768)
    q_raw = np.array([get_raw(t) for t in q_texts])                              # (n,768)

    # 白化开：库=embed_text(lt[L], L)，查询=embed_text(snip, L)（同层一致）
    # 通过 _apply_whitening(_forward_raw(t), layer) 实现（与 embed_text 字节级等价，先做一次一致性校验）。
    check_txt = q_texts[0]
    assert np.allclose(np.array(embed_text(check_txt, "full")),
                       np.array(_apply_whitening(get_raw(check_txt).tolist(), "full")), atol=1e-9)
    print("已确认：embed_text(t, layer) ≡ _forward_raw(t) 后 _apply_whitening(·, layer)\n", flush=True)

    lib_wh = {}
    q_wh = {}
    q_wh_full = np.array([_apply_whitening(get_raw(t).tolist(), "full") for t in q_texts])  # 跨层失配对照
    for L in LAYERS:
        lib_wh[L] = np.array([_apply_whitening(get_raw(t).tolist(), L) for t in lib_texts[L]])
        q_wh[L] = np.array([_apply_whitening(get_raw(t).tolist(), L) for t in q_texts])

    # 白化开（fresh）：在当前库自身 raw 向量上算 PCA(95%) 白化，库/查询同层一致（干净的同数据消融）
    lib_wh_fresh = {}
    q_wh_fresh = {}
    for L in LAYERS:
        mean, W, k = pca_whiten_params(lib_raw[L])
        lib_wh_fresh[L] = apply_wht(lib_raw[L], mean, W)
        q_wh_fresh[L] = apply_wht(q_raw, mean, W)

    # 各模式的 recall@1 + 分离度
    results = {}
    for L in LAYERS:
        results[L] = {}
        r, cor, wrg = evaluate(lib_raw[L], q_raw, ids)
        results[L]["raw"] = {"recall": r, "n": n, "cor_med": statistics.median(cor), "wrong_med": statistics.median(wrg)}
        r, cor, wrg = evaluate(lib_wh[L], q_wh[L], ids)
        results[L]["whitened_saved"] = {"recall": r, "n": n, "cor_med": statistics.median(cor), "wrong_med": statistics.median(wrg)}
        r, cor, wrg = evaluate(lib_wh_fresh[L], q_wh_fresh[L], ids)
        results[L]["whitened_fresh"] = {"recall": r, "n": n, "cor_med": statistics.median(cor), "wrong_med": statistics.median(wrg)}
        r, cor, wrg = evaluate(lib_wh[L], q_wh_full, ids)
        results[L]["query_full"] = {"recall": r, "n": n, "cor_med": statistics.median(cor), "wrong_med": statistics.median(wrg)}

    # 打印
    print("=" * 110)
    print(f"逐视图 recall@1（n={n}）")
    print("=" * 110)
    print(f"{'layer':<14}{'raw(白化关)':>18}{'whitened_saved':>22}{'whitened_fresh':>22}{'query_full(复刻bug)':>24}")
    for L in LAYERS:
        rw = results[L]["raw"]
        ws = results[L]["whitened_saved"]
        wf = results[L]["whitened_fresh"]
        qf = results[L]["query_full"]
        s_rw = f"{rw['recall']}/{rw['n']}={rw['recall']/rw['n']:.1%}"
        s_ws = f"{ws['recall']}/{ws['n']}={ws['recall']/ws['n']:.1%}"
        s_wf = f"{wf['recall']}/{wf['n']}={wf['recall']/wf['n']:.1%}"
        s_qf = f"{qf['recall']}/{qf['n']}={qf['recall']/qf['n']:.1%}"
        print(f"{L:<14}{s_rw:>18}{s_ws:>22}{s_wf:>22}{s_qf:>24}")

    print("\n分离度 gap = 正确中位 − 错误中位（越大越可分）")
    print(f"{'layer':<14}{'raw gap':>14}{'saved gap':>16}{'fresh gap':>16}")
    for L in LAYERS:
        rw = results[L]["raw"]
        ws = results[L]["whitened_saved"]
        wf = results[L]["whitened_fresh"]
        print(f"{L:<14}{rw['cor_med']-rw['wrong_med']:>14.4f}"
              f"{ws['cor_med']-ws['wrong_med']:>16.4f}"
              f"{wf['cor_med']-wf['wrong_med']:>16.4f}")

    print("\n详细中位数（正确 / 错误，余弦）")
    for L in LAYERS:
        for mode in ["raw", "whitened_saved", "whitened_fresh"]:
            r = results[L][mode]
            print(f"  {L:<14} {mode:<16} 正确中位={r['cor_med']:.4f}  错误中位={r['wrong_med']:.4f}")

    out = ROOT / "reports" / f"view_recall1_whitening_{args.db}.json"
    out.write_text(json.dumps({"db": str(db_path), "n": n,
                               "results": {L: results[L] for L in LAYERS}},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
