# -*- coding: utf-8 -*-
"""① 编码器对比：distilbert（现用） vs codebert（候选），在**同一套离线协议**下比。

## 为什么做这个

《01》问题 9 结尾自己列出的下一步之一就是"换编码器（通用英文模型 → 代码模型）"。
本机 `model_cache` 里 **codebert-base 与 distilbert 都在**，实测都能离线出 768 维向量，
所以这件事**不需要 GPU**。

更实际的意义：**知识库重建（第 3 项）要不要连编码器一起换？**
重建是唯一动权威数据的一步，不该返工 —— 先把数字拿到手再动手。

## 现状（先看清楚，免得白做）

`infrastructure/embeddings/codebert_embedder.py` 的 `embed()` 注释写着
"**回滚**：code_pattern 层不再用 codebert …… 四层统一走 distilbert"，
但该文件**开头的文档字符串仍写着"code_pattern 层：codebert-base"** —— 文档与代码不一致。
那次回滚只给了"代码精确匹配由 is_subseq 承担"这个理由，**没有任何实测数字**，
所以这次不是在重复已测过的事。

## 口径（先说清楚，否则数字不可比）

* **索引文本**：用生产自己的 `_build_enhanced_issue_pattern_text()` 现算。两种索引：
  `cur` = 不带 `llm_semantic`（线上现状）；`sem` = 带 `llm_semantic`（重建后的样子）。
* **查询文本**：用生产自己的 `_build_query_text()`；两种查询，**都按线上做法"每个分片各查一次再取并集"**：
  `code` = 各分片原始代码（占线上查询量约 80%）；`llm` = 各分片语义描述。
* **相似度**：向量 L2 归一化后点积（= 余弦）。线上是余弦距离，二者单调等价
  （界面展示时用 `1 - 距离/2`）。
* **四条臂**：`D+live`（distilbert + 线上白化 = **生产参考线**）、`D+raw`（不白化）、
  `C+raw`（codebert 不白化）、`C+fit`（codebert + 在同样索引文本上现拟合白化，
  复用仓库自己的 `pca_whiten_params`）。
* **偏移**：四臂**都不做** C2 查询偏移（否则偏移与编码器耦合，说不清功劳归谁）。
* **指标**：自己那条知识进 top-5 的层数（8 样本 × 4 层 = 32）；另报首位集中度（hubness 代理）。
* **先写死的判定**：只有 codebert 在"**sem 索引 + code 查询**"这条主导线上
  比参考线**至少好 20%（且 ≥ +6/32）**，并且四个格子都不比参考线差超过 1 个命中，
  才认为"值得换"；否则**不换**（换掉要连带作废白化基、C2 偏移与全部旧基线）。
* **样本量诚实声明**：每格 n=32，1~2 个命中的差别属于噪声；随机猜的期望约 0.8/32。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from utils.experiments.experiment_llm_semantic_alignment import LAYERS  # noqa: E402
from utils.experiments.view_recall1_whitening_ablation import pca_whiten_params  # noqa: E402

WH_PATH = ROOT / "infrastructure/embeddings/whitening_transform.json"
ARMS = ("D+live", "D+raw", "C+raw", "C+fit")


class Encoders:
    """两个编码器按需加载；mean pooling（去 CLS）+ L2，与生产 `_forward_raw` 同式。

    带**磁盘缓存**：一次实验要编 3000 多条文本（其中 codebert 很慢），
    调试时重跑不该再等十分钟。键 = 编码器名 + 文本 sha1。
    """

    def __init__(self, threads: int = 8, cache_path=None):
        import torch
        torch.set_num_threads(threads)
        self._cache: dict = {}
        self._models: dict = {}
        self._disk_path = Path(cache_path) if cache_path else None
        self._disk: dict = {}
        if self._disk_path and self._disk_path.exists():
            try:
                import numpy as np
                with np.load(self._disk_path, allow_pickle=False) as z:
                    for k, v in zip(z["keys"], z["vecs"]):
                        self._disk[str(k)] = v
                print("    [缓存] 载入 %d 条已编码向量 (%s)" % (len(self._disk), self._disk_path.name))
            except Exception as e:
                print("    [缓存] 载入失败（忽略）:", e)
                self._disk = {}

    def save_disk(self) -> None:
        if not self._disk_path or not self._disk:
            return
        import numpy as np
        keys = list(self._disk)
        self._disk_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._disk_path, keys=np.array(keys),
                            vecs=np.array([self._disk[k] for k in keys]))
        print("    [缓存] 写出 %d 条 -> %s" % (len(keys), self._disk_path))

    def _load(self, which: str):
        if which not in self._models:
            from transformers import AutoModel, AutoTokenizer
            name = {"D": "distilbert-base-uncased", "C": "microsoft/codebert-base"}[which]
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            mdl = AutoModel.from_pretrained(name, local_files_only=True)
            mdl.eval()
            self._models[which] = (tok, mdl)
        return self._models[which]

    def encode(self, which: str, text: str):
        import hashlib
        key = (which, text)
        if key in self._cache:
            return self._cache[key]
        disk_key = "%s:%s" % (which, hashlib.sha1((text or "").encode("utf-8")).hexdigest())
        if disk_key in self._disk:
            self._cache[key] = self._disk[disk_key]
            return self._disk[disk_key]
        import numpy as np
        import torch
        tok, mdl = self._load(which)
        inp = tok(text or "", return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = mdl(**inp)
        vec = out.last_hidden_state[:, 1:, :].mean(dim=1).squeeze(0).numpy()
        n = float(np.linalg.norm(vec))
        vec = vec / n if n else vec
        self._cache[key] = vec
        self._disk[disk_key] = vec
        return vec

    def encode_many(self, which: str, texts, label: str = ""):
        import numpy as np
        if not texts:
            return np.zeros((0, 768))
        out = []
        for i, t in enumerate(texts, 1):
            out.append(self.encode(which, t))
            if label and i % 100 == 0:
                print("    编码 %s: %d/%d" % (label, i, len(texts)), flush=True)
        return np.array(out)


def whiten(X, mean, W):
    import numpy as np
    Y = (X - mean) @ W
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    return Y / np.where(n == 0, 1.0, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--index-semantic", type=Path, default=ROOT / "reports/ls_index_v2.json")
    ap.add_argument("--query-semantic", type=Path, default=ROOT / "reports/ls_query_v2.json")
    ap.add_argument("--chunks", type=Path, default=ROOT / "reports/smoke_chunks.json")
    ap.add_argument("--all-chunks", type=Path, default=ROOT / "reports/smoke_allchunks.json")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/encoder_swap.json")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import numpy as np

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    from infrastructure.database.weaviate.service import WeaviateVectorService

    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    agent.query_offset_correction = False
    svc = WeaviateVectorService()

    idx_sem = json.loads(args.index_semantic.read_text(encoding="utf-8"))
    qry_all = json.loads(args.query_semantic.read_text(encoding="utf-8"))
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    allchunks = json.loads(args.all_chunks.read_text(encoding="utf-8"))

    vuln_key = {}
    for cve, cs in allchunks.items():
        target = str(chunks.get(cve, {}).get("code") or "")[:200]
        for i, c in enumerate(cs):
            if target and str(c.get("text") or "")[:200] == target:
                vuln_key[cve] = "%s@%d" % (cve, i)
                break
        vuln_key.setdefault(cve, "%s@0" % cve)

    con = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    kb, by_cve = {}, {}
    for row in con.execute(
        "select id, title, error_type, severity, language, framework, error_description, "
        "problematic_pattern, solution, file_pattern, class_pattern from issue_patterns"
    ):
        sid, title = int(row[0]), (row[1] or "").strip().upper()
        rec = dict(zip(["error_type", "severity", "language", "framework", "error_description",
                        "problematic_pattern", "solution", "file_pattern", "class_pattern"], row[2:]))
        rec["cve"] = title
        rec["llm_semantic"] = idx_sem.get(title, "")
        kb[sid] = rec
        by_cve[title] = (sid, rec)
    con.close()
    print("条目 %d；查询样本 %d 个（分片 %d 条）" % (len(kb), len(chunks), len(allchunks)))

    # ---- 索引文本 ---------------------------------------------------------- #
    texts = {"cur": {L: {} for L in LAYERS}, "sem": {L: {} for L in LAYERS}}
    for sid, rec in kb.items():
        for variant, sem in (("cur", ""), ("sem", rec["llm_semantic"])):
            props = dict(rec, sqlite_id=sid, status="active", llm_semantic=sem)
            for L in LAYERS:
                texts[variant][L][sid] = svc._build_enhanced_issue_pattern_text(props, L)
    print("  带 llm_semantic 的条目：%d" % sum(1 for r in kb.values() if r["llm_semantic"]))

    # ---- 查询文本 ---------------------------------------------------------- #
    def code_query_texts(cve):
        fp = kb[by_cve[cve][0]]["file_pattern"]
        out = []
        for c in allchunks.get(cve, []):
            iss = {"source": "source_code_chunk",
                   "description": "source_code_chunk L%s-%s: %s" % (c.get("start"), c.get("end"), c.get("text")),
                   "code_snippet": c.get("text"), "file": fp, "severity": "medium"}
            out.append(agent._build_query_text(iss, fp))
        return out

    def llm_query_texts(cve):
        fp = kb[by_cve[cve][0]]["file_pattern"]
        keys = [k for k in qry_all if k.startswith(cve + "@")] or [vuln_key.get(cve)]
        out = []
        for k in keys:
            d = qry_all.get(k, "")
            if d:
                iss = {"source": "source_code_chunk", "description": d, "code_snippet": "",
                       "file": fp, "severity": "medium"}
                out.append(agent._build_query_text(iss, fp))
        return out

    queries = {"code": {c: code_query_texts(c) for c in chunks},
               "llm": {c: llm_query_texts(c) for c in chunks}}
    print("  查询条数：code %d，llm %d"
          % (sum(len(v) for v in queries["code"].values()), sum(len(v) for v in queries["llm"].values())))

    # ---- 编码与索引 -------------------------------------------------------- #
    enc = Encoders(args.threads, cache_path=ROOT / "reports" / "encoder_swap_cache.npz")
    live = {L: (np.array(v["mean"]), np.array(v["W"]))
            for L, v in json.loads(WH_PATH.read_text(encoding="utf-8")).items()}

    index_vectors: dict = {}
    fitted: dict = {}
    for variant in ("cur", "sem"):
        for L in LAYERS:
            sids = list(texts[variant][L])
            tlist = [texts[variant][L][s] for s in sids]
            Xd = enc.encode_many("D", tlist, label="D/%s/%s" % (variant, L))
            Xc = enc.encode_many("C", tlist, label="C/%s/%s" % (variant, L))
            index_vectors[(variant, "D+raw", L)] = {s: Xd[i] for i, s in enumerate(sids)}
            index_vectors[(variant, "D+live", L)] = {
                s: whiten(Xd, live[L][0], live[L][1])[i] for i, s in enumerate(sids)}
            index_vectors[(variant, "C+raw", L)] = {s: Xc[i] for i, s in enumerate(sids)}
            if L not in fitted:                      # 白化参数在 cur 索引文本上拟合一次
                m2, W2, k2 = pca_whiten_params(Xc)
                fitted[L] = (m2, W2, k2)
            index_vectors[(variant, "C+fit", L)] = {
                s: whiten(Xc, fitted[L][0], fitted[L][1])[i] for i, s in enumerate(sids)}

    print("  白化保留维数 k：codebert 现拟合 %s；线上 distilbert %s"
          % ({L: fitted[L][2] for L in LAYERS}, {L: live[L][1].shape[1] for L in LAYERS}))

    # 矩阵化：把每格索引堆成矩阵，排名用矩阵乘法（用 Python 循环点积会慢几十倍）
    INDEX = {}
    for key, d in index_vectors.items():
        sids = list(d)
        INDEX[key] = (sids, np.array([d[s] for s in sids]), {s: i for i, s in enumerate(sids)})

    def rank_and_top(q, own, key):
        sids, M, pos = INDEX[key]
        sims = M @ q
        i = pos[own]
        return 1 + int((sims > sims[i]).sum()), sids[int(sims.argmax())]

    def query_vec(arm: str, layer: str, text: str):
        which = "D" if arm.startswith("D") else "C"
        raw = enc.encode(which, text)
        if arm.endswith("raw"):
            return raw
        mean, W = (live[layer] if arm == "D+live" else fitted[layer][:2])
        y = (raw - mean) @ W
        n = float(np.linalg.norm(y))
        return y / n if n else y

    # ---- 主测量 ------------------------------------------------------------ #
    results = {}
    for variant in ("cur", "sem"):
        for arm in ARMS:
            for qmode in ("code", "llm"):
                hit5, tot = 0, 0
                top1 = Counter()
                rows = []
                for cve in chunks:
                    own = by_cve[cve][0]
                    per = {}
                    for L in LAYERS:
                        best, best_top1 = None, None
                        for txt in queries[qmode].get(cve, []):
                            r, t1 = rank_and_top(query_vec(arm, L, txt), own, (variant, arm, L))
                            if best is None or r < best:
                                best, best_top1 = r, t1
                        per[L] = best
                        tot += 1
                        hit5 += int(best is not None and best <= 5)
                        if best_top1 is not None:
                            top1[best_top1] += 1
                    rows.append({"cve": cve, "layers": per})
                modal = top1.most_common(1)[0][1] / max(1, sum(top1.values())) if top1 else 0.0
                results["%s|%s|%s" % (variant, arm, qmode)] = {
                    "top5": hit5, "total": tot, "hub_share": modal, "rows": rows}

    # ---- 自检 / 负控 ------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("自检 / 负控")
    print("=" * 100)
    for arm in ("D+live", "C+fit"):
        sid = next(iter(kb))
        q = INDEX[("cur", arm, "semantic")][1][INDEX[("cur", arm, "semantic")][2][sid]]
        print("  自检（用条目自己的层文本当查询，应排第 1）%-7s rank=%d"
              % (arm, rank_and_top(q, sid, ("cur", arm, "semantic"))[0]))
    rnd = np.random.default_rng(0)
    hitr = 0
    # 注意：白化后的索引向量在 k 维空间（k≈87），随机向量必须同维，否则矩阵乘法直接报维度错
    ref_key = ("cur", "D+live", "semantic")
    ref_dim = INDEX[ref_key][1].shape[1]
    for _ in range(300):
        q = rnd.normal(size=ref_dim)
        q = q / np.linalg.norm(q)
        hitr += int(rank_and_top(q, next(iter(kb)), ref_key)[0] <= 5)
    print("  负控（随机向量，维度 %d）命中 top-5 比例：%.1f%%（理论 ≈ 5/200 = 2.5%%）"
          % (ref_dim, 100 * hitr / 300))

    # ---- 结果 -------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("结果：自己那条知识进 top-5 的层数（8 样本 × 4 层 = 32）")
    print("=" * 100)
    print("  %-5s %-6s | %-22s | %-22s" % ("索引", "查询", "distilbert", "codebert"))
    print("  %-5s %-6s | %-10s %-11s | %-10s %-11s" % ("", "", "参考(D+live)", "不白化(D+raw)", "不白化(C+raw)", "现拟合(C+fit)"))
    for variant in ("cur", "sem"):
        for qmode in ("code", "llm"):
            g = lambda arm: results["%s|%s|%s" % (variant, arm, qmode)]
            print("  %-5s %-6s | %-10s %-11s | %-10s %-11s"
                  % (variant, qmode,
                     "%d/%d" % (g("D+live")["top5"], g("D+live")["total"]),
                     "%d/%d" % (g("D+raw")["top5"], g("D+raw")["total"]),
                     "%d/%d" % (g("C+raw")["top5"], g("C+raw")["total"]),
                     "%d/%d" % (g("C+fit")["top5"], g("C+fit")["total"])))

    print("\n  首位集中度（同一格内 top-1 落在同一条目上的最大占比，越低越不像「万能邻居」）：")
    for variant in ("cur", "sem"):
        for qmode in ("code", "llm"):
            print("    %-5s %-6s  D+live %.2f  C+fit %.2f"
                  % (variant, qmode, results["%s|D+live|%s" % (variant, qmode)]["hub_share"],
                     results["%s|C+fit|%s" % (variant, qmode)]["hub_share"]))

    print("\n  逐样本（sem 索引 / code 查询，显示每层的名次，越小越好）：")
    rd = {r["cve"]: r for r in results["sem|D+live|code"]["rows"]}
    rc = {r["cve"]: r for r in results["sem|C+fit|code"]["rows"]}
    for cve in chunks:
        f = lambda r: " ".join("%s%s" % (L[:4], r["layers"][L]) for L in LAYERS)
        print("    %-16s D: %-24s C: %-24s" % (cve, f(rd[cve]), f(rc[cve])))

    # ---- 判定 -------------------------------------------------------------- #
    base = results["sem|D+live|code"]["top5"]
    cand = results["sem|C+fit|code"]["top5"]
    need = base + max(6, int(round(base * 0.2)))
    cells = [("%s|%s" % (v, q), results["%s|D+live|%s" % (v, q)]["top5"],
              results["%s|C+fit|%s" % (v, q)]["top5"])
             for v in ("cur", "sem") for q in ("code", "llm")]
    print("\n" + "=" * 100)
    print("按先写死的判定：主导线 = sem 索引 + code 查询；基准 %d/32，需要 ≥ %d，且四格都不差超过 1" % (base, need))
    print("=" * 100)
    print("  主导线：codebert(C+fit) %d  vs  distilbert(D+live) %d  ->  %s"
          % (cand, base, "达标" if cand >= need else "未达标"))
    ok_all = True
    for name, b, c in cells:
        if c - b < -1:
            ok_all = False
        print("    %-12s 参考 %2d  候选 %2d  (%+d)" % (name, b, c, c - b))
    verdict = "值得换" if (cand >= need and ok_all) else "不值得换（保持 distilbert）"
    print("\n  结论：**%s**" % verdict)
    print("  口径提醒：n=32，1~2 个命中的差别属噪声；随机期望 ≈ 0.8/32；四臂都不带 C2 偏移。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    enc.save_disk()
    args.out.write_text(json.dumps({
        "cells": {k: {"top5": v["top5"], "total": v["total"], "hub_share": v["hub_share"]}
                  for k, v in results.items()},
        "codebert_k": {L: fitted[L][2] for L in LAYERS},
        "distilbert_live_k": {L: int(live[L][1].shape[1]) for L in LAYERS},
        "verdict": verdict,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写出: %s" % args.out)


if __name__ == "__main__":
    main()
