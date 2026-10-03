#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**A8 预筛（B2 的"正路"）**：LLM 语义描述 ↔ 库里 `llm_semantic` 的**散文对散文**匹配。

## 为什么单独测这一路（与 A7 的区别）

* **A7** 测的是"代码块 ↔ 代码块"：把库里的错误代码/修复代码切成块当索引。
  结果：**区分度反而更差**（逐块 top-1 指向 own 只有 21%，比 A4 的 30.3% 低；
  中位数有 6 条别的知识的块级相似度 ≥ own，最坏 139 条）⇒ "代码块撞车"很严重。
* **A8** 测的是**设计文档里写的那条路**（`models.py` 里 `llm_semantic` 的注释就是它的契约）：
  > 大模型对该代码的**语义理解**（这段代码在做什么 + 可能出什么问题）……
  > 是**唯一能和"分析时的查询"用同一套规则对接的文本**。
  也就是：**索引侧** = 库条目由 LLM 写的语义描述；**查询侧** = 分析时 LLM 对当前代码写的语义描述；
  两边**同一语域、同一语言**（英文散文），走 `semantic` 层。

## 数据来源（都能本地拿到，不需要 GPU）

* 查询侧：运行产物 `reports/analysis/<CVE>/<run>/second_pass/**/*_r2.json` 里
  `issues[].llm_semantic`（**实测确实存下来了**，不是重新生成）。
* 索引侧：`issue_patterns.llm_semantic`（重建后 **193/200 条非空**）。
* 嵌入：索引侧 `_default_embed(text, "semantic")`、查询侧 `_query_embed(text, "semantic")`
  —— **全是生产实现**，不重新拟合任何基。

## 指标（与 A4/A7 同口径，便于直接比）

own 条目的名次 / 相对分 z / 有多少别的条目分数 ≥ own（撞车代价），
以及**三个一直救不回的样本**（`CVE-2002-2443` / `CVE-2017-17053` / `CVE-2018-6057`）单独列出。

用法：
    python -X utf8 utils/experiments/precheck6_llm_prose_match.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

DB = ROOT / "reports/mas_rebuild_candidate_v2.db"
CACHE = ROOT / "reports/a8_prose_embeddings.json"
CHRONIC = {"CVE-2002-2443", "CVE-2017-17053", "CVE-2018-6057"}


def cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def harvest_queries(runs_file: Path, limit: int):
    """从运行产物里取每个样本的 **issue 字典**（含 `llm_semantic`），供**生产格式化器**构造查询文本。

    为什么不能直接拿 `llm_semantic` 当查询：生产的查询文本是
    `_build_query_text()` 拼出来的（`error_type: <family> | <描述> | file:… | sig:…`），
    而 C2 的**查询偏移正是在那种格式上拟合**的；直接嵌原始描述等于换了输入格式，不公平。
    """
    out = []
    for line in runs_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cve = line.split("/")[0]
        issues = []
        for f in glob.glob(os.path.join(str(ROOT / "reports/analysis"), line,
                                        "second_pass", "**", "*_r2.json"), recursive=True):
            try:
                j = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception:
                continue
            for it in (j.get("issues") or []):
                if isinstance(it, dict) and len(str(it.get("llm_semantic") or "").strip()) >= 30:
                    it = dict(it)
                    it["_run_file"] = j.get("file") or ""
                    issues.append(it)
        if issues:
            out.append((cve, issues))
        if limit and len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "reports/arm1_runs.txt")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--layer", default="semantic")
    ap.add_argument("--index", choices=("dump", "reembed"), default="dump",
                    help="索引侧向量来源：dump=产库里真实存的层向量（生产口径，默认）；"
                         "reembed=自己嵌 llm_semantic（**不是**生产口径，仅供对照）")
    ap.add_argument("--dump", type=Path,
                    default=ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl")
    ap.add_argument("--report", type=int, default=10)
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import (
        AIDrivenSecondPassAnalysisAgent,
    )
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    con = sqlite3.connect(str(args.db))
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT id, title, llm_semantic FROM issue_patterns").fetchall()
    con.close()
    id_by_title = {(r["title"] or "").strip().upper(): int(r["id"]) for r in rows}

    # ---------- 索引侧：**优先用产库里真实存的向量**（生产口径），否则退回"自己嵌 llm_semantic" ----------
    kv = {}
    if args.index == "dump":
        dump = args.dump
        for line in dump.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get("vector_layer") or "").lower() != args.layer:
                continue
            v = r.get("_vector") or r.get("vector")
            if v:
                kv[int(r["sqlite_id"])] = [float(x) for x in v]
        print("索引侧：来自**产库 dump** 的 %s 层向量 %d 条（生产口径：层文本含 error_type/描述/llm_semantic）"
              % (args.layer, len(kv)))
    else:
        cache0 = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
        kv = cache0.setdefault("kb_reembed", {})
        kb = [(int(r["id"]), str(r["llm_semantic"] or "").strip()) for r in rows]
        kb = [(i, t) for i, t in kb if len(t) >= 30]
        print("索引侧：自己重嵌 llm_semantic 一段文字（**不是生产口径**）%d 条" % len(kb))
        for i, (sid, t) in enumerate(kb):
            if str(sid) not in kv:
                kv[str(sid)] = agent._default_embed(t, args.layer)
            if i % 50 == 0:
                CACHE.write_text(json.dumps(cache0), encoding="utf-8")
        CACHE.write_text(json.dumps(cache0), encoding="utf-8")
        kv = {int(k): v for k, v in kv.items()}

    queries = harvest_queries(args.runs, args.limit)
    print("查询侧：%d 个样本拿到了 llm_semantic（合计 %d 个 issue）"
          % (len(queries), sum(len(t) for _, t in queries)))

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    qv = cache.setdefault("q_fmt", {})
    per_sample = []
    for cve, issues in queries:
        own = id_by_title.get(cve.upper())
        if own is None:
            continue
        vecs = []
        for k, it in enumerate(issues):
            key = "%s|%d" % (cve, k)
            if key not in qv:
                # **生产口径**：用 `_build_query_text` 拼文本，再走查询侧嵌入（含 C2 偏移）
                text = agent._build_query_text(
                    it, it.get("_run_file") or it.get("file") or "",
                    str(it.get("description") or ""), str(it.get("source") or ""))
                qv[key] = agent._query_embed(text, args.layer)
            vecs.append(qv[key])
        all_sims, best_by_entry = [], {}
        for q in vecs:
            for sid, v in kv.items():
                s = cos(q, v)
                all_sims.append(s)
                if s > best_by_entry.get(int(sid), -1.0):
                    best_by_entry[int(sid)] = s
        if own not in best_by_entry:
            continue
        own_sim = best_by_entry[own]
        order = sorted(best_by_entry.items(), key=lambda x: -x[1])
        rank = [s for s, _ in order].index(own) + 1
        mu = sum(all_sims) / len(all_sims)
        sd = (sum((x - mu) ** 2 for x in all_sims) / max(1, len(all_sims) - 1)) ** 0.5
        per_sample.append({
            "cve": cve, "own": own, "own_sim": own_sim, "rank": rank,
            "z": (own_sim - mu) / sd if sd > 1e-9 else 0.0, "n_q": len(vecs),
            "strong_other": sum(1 for s, v in best_by_entry.items() if s != own and v >= own_sim),
            "over_070": sum(1 for _, v in best_by_entry.items() if v >= 0.70),
        })
    CACHE.write_text(json.dumps(cache), encoding="utf-8")

    n = len(per_sample)
    print("\n" + "=" * 100)
    print("散文↔散文（索引侧 llm_semantic ↔ 查询侧 LLM 语义描述，semantic 层）")
    print("=" * 100)
    print("%-16s %9s %6s %6s %9s %10s %10s" %
          ("样本", "own 相似度", "名次", "查询段数", "z", "别的条目≥own", "≥0.70 的条目"))
    for r in sorted(per_sample, key=lambda x: -x["z"])[: args.report]:
        mark = "  ← 一直救不回" if r["cve"] in CHRONIC else ""
        print("%-16s %9.4f %6d %6d %9.2f %10d %10d%s"
              % (r["cve"], r["own_sim"], r["rank"], r["n_q"], r["z"],
                 r["strong_other"], r["over_070"], mark))

    top1 = sum(1 for r in per_sample if r["rank"] == 1)
    top5 = sum(1 for r in per_sample if r["rank"] <= 5)
    print("\n--- 汇总（%d 个样本）---" % n)
    print("  own top-1 : %d/%d（%.0f%%）    own top-5 : %d/%d（%.0f%%）"
          % (top1, n, 100.0 * top1 / max(1, n), top5, n, 100.0 * top5 / max(1, n)))
    print("  z ≥ 2     : %d/%d      z ≥ 3 : %d/%d"
          % (sum(1 for r in per_sample if r["z"] >= 2), n,
             sum(1 for r in per_sample if r["z"] >= 3), n))
    print("  别的条目 ≥ own（均值 %.1f，最大 %d）｜绝对分 ≥0.70 的条目（均值 %.1f）"
          % (sum(r["strong_other"] for r in per_sample) / max(1, n),
             max(r["strong_other"] for r in per_sample),
             sum(r["over_070"] for r in per_sample) / max(1, n)))
    print("\n--- 三个「一直救不回」的样本 ---")
    for r in per_sample:
        if r["cve"] in CHRONIC:
            print("  %-16s own=%.4f  名次 %3d  z %.2f  别的条目≥own %d"
                  % (r["cve"], r["own_sim"], r["rank"], r["z"], r["strong_other"]))

    out = ROOT / "reports/a8_prose_match.json"
    out.write_text(json.dumps(per_sample, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已落盘: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
