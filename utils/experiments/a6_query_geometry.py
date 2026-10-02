#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A6：用**真实查询向量**复测 C2 修复的几何指标（纯离线，不需要 GPU/向量库）。

## 为什么是"复测"而不是"重定 σ 判据"

原先定的验收门槛"相似度波动 σ 要显著变大"**已经作废**（《01》第六节的说明）：
C2 修复的原理是"让**不同查询**的排序互不相同"，而不是"让单个查询内部更分明"，
所以 σ 基本不变是**正常的**。真正该看的几何量是这四个：

| 指标 | 含义 | 方向 |
|---|---|---|
| **查询间余弦 QQ** | 不同查询向量彼此有多像（"全都塌向同一方向"的量化） | 越低越好（预筛目标 < 0.15） |
| **top-10 集中度** | 命中集中在少数几条知识上（"万能邻居"） | 越低越好（目标 < 0.30） |
| **N_max / anti-hub** | 被最多查询命中的那条 / 一次都没被命中的条数 | N_max 越低、anti-hub 越少越好 |
| **被召回行数** | 200 条里有几条进过 top-5 | 越高越好（目标 200/200） |

本脚本对每个查询向量文件（按层）算这四个，**并让数据自己标注来源**：
C2 修复前查询间余弦实测约 0.36、修复后约 0.035 —— 谁是谁一眼就能对上。

用法：
    python -X utf8 utils/experiments/a6_query_geometry.py
    python -X utf8 utils/experiments/a6_query_geometry.py --queries reports/query_vectors/*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAYERS = ["semantic", "code_pattern", "solution", "full"]
TOPK = 5
QQ_SAMPLE = 400          # 查询间余弦最多用这么多条（两两对数 ~8 万，够稳且秒级）


def load_kb(dump: Path) -> dict:
    out: dict[str, list] = {L: [] for L in LAYERS}
    ids: dict[str, list] = {L: [] for L in LAYERS}
    for line in dump.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        L = str(r.get("vector_layer") or "").strip().lower()
        v = r.get("_vector") or r.get("vector")
        if L in out and v:
            out[L].append(v)
            ids[L].append(int(r.get("sqlite_id") or 0))
    return {L: (np.asarray(out[L], dtype=np.float32), ids[L]) for L in LAYERS if out[L]}


def load_queries(path: Path) -> dict:
    q: dict[str, list] = {L: [] for L in LAYERS}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        L = str(r.get("layer") or "").strip().lower()
        v = r.get("vec")
        if L in q and v:
            q[L].append(v)
    return {L: np.asarray(v, dtype=np.float32) for L, v in q.items() if v}


def l2(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def metrics(q: np.ndarray, kb: np.ndarray, ids: list) -> dict:
    q, kb = l2(q), l2(kb)
    S = q @ kb.T                                    # (nq, nkb) 余弦
    order = np.argsort(-S, axis=1)[:, :TOPK]        # 每个查询的 top-5
    nk = np.zeros(kb.shape[0], dtype=int)
    for row in order:
        nk[row] += 1
    od = np.sort(nk)[::-1]
    res = {
        "nq": int(q.shape[0]),
        "nkb": int(kb.shape[0]),
        "qx_std": float(S.std()),
        "per_q_std": float(S.std(axis=1).mean()),
        "qx_mean": float(S.mean()),
        "N_max": int(od[0]) if od.size else 0,
        "top10_share": float(od[:10].sum() / max(1.0, nk.sum())),
        "rows_hit": int((nk > 0).sum()),
        "anti_hub": int((nk == 0).sum()),
    }
    if q.shape[0] >= 3:
        sub = q[:QQ_SAMPLE]
        C = sub @ sub.T
        iu = np.triu_indices(sub.shape[0], k=1)
        res["qq_mean"] = float(C[iu].mean())
        res["qq_p90"] = float(np.percentile(C[iu], 90))
        res["qq_n"] = int(sub.shape[0])
    else:
        res["qq_mean"] = None
        res["qq_p90"] = None
        res["qq_n"] = int(q.shape[0])
    # 同一查询的 top-1 是否总落在同一条上（"万能邻居"的最直观形态）
    top1 = order[:, 0]
    if top1.size:
        vals, cnt = np.unique(top1, return_counts=True)
        res["top1_share"] = float(cnt.max() / top1.size)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path,
                    default=ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl")
    ap.add_argument("--queries", nargs="+",
                    default=sorted(glob.glob(str(ROOT / "reports/query_vectors/*.jsonl"))))
    ap.add_argument("--json-out", type=Path, default=ROOT / "reports/a6_query_geometry.json")
    args = ap.parse_args()

    kb = load_kb(args.kb)
    print("知识库向量：%s → %s" % (args.kb.name,
                                  {L: len(v[1]) for L, v in kb.items()}))
    allres = {}
    print("\n%-26s %-13s %6s %9s %9s %8s %8s %8s %8s %8s"
          % ("文件", "层", "nq", "QQ均值", "QQ_p90", "σ(全)", "N_max", "top10", "命中行", "anti"))
    for qs in args.queries:
        p = Path(qs)
        qs_by_layer = load_queries(p)
        allres[p.name] = {}
        for L in LAYERS:
            if L not in qs_by_layer or L not in kb:
                continue
            m = metrics(qs_by_layer[L], kb[L][0], kb[L][1])
            allres[p.name][L] = m
            print("%-26s %-13s %6d %9s %9s %8.4f %8d %8.3f %8d %8d"
                  % (p.name, L, m["nq"],
                     ("%.4f" % m["qq_mean"]) if m["qq_mean"] is not None else "-",
                     ("%.4f" % m["qq_p90"]) if m["qq_p90"] is not None else "-",
                     m["qx_std"], m["N_max"], m["top10_share"], m["rows_hit"], m["anti_hub"]))

    # 跨层平均（同一文件的整体印象）
    print("\n%-26s %9s %9s %9s %9s %9s %9s"
          % ("文件（四层平均）", "QQ均值", "σ(全)", "N_max", "top10", "命中行", "anti"))
    for name, per_layer in allres.items():
        if not per_layer:
            continue
        def avg(k, only_num=True):
            vals = [v[k] for v in per_layer.values() if v.get(k) is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        print("%-26s %9.4f %9.4f %9.1f %9.3f %9.1f %9.1f"
              % (name, avg("qq_mean"), avg("qx_std"), avg("N_max"),
                 avg("top10_share"), avg("rows_hit"), avg("anti_hub")))

    print("\n判据（《01》第六节）：QQ < 0.15（预筛用）｜top-10 集中度 < 0.30｜被召回行 200/200")
    args.json_out.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
    print("明细已落盘: %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
