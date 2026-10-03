#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**A7 预筛（B2/D6）**：把**知识库侧也切成块**之后，语义通道到底强多少？

## 与 A4 的区别（一句话）

* **A4 已经做的**：把**样本文件**切成块当查询，去比**条目级**的库向量 → 证明"块级查询"能把分抬起来
  （z≥2 从 11/30 提到 23/30），但**库侧还是一整条压成一个均值向量**（预检 2 已证那表示不了 0.98% 的补丁）。
* **D6 要做的**：**库侧也按块建索引** —— 让"样本里的某一段代码"去比"库里的**某一段**代码"，
  而不是比"整条知识的平均话题"。

## 本脚本回答的三个问题（都在本地、用生产实现、不碰 GPU）

1. **块级索引后，"自己的条目"能不能被更准地找到**：own 条目在 200 条里的名次（top-1 / top-5）。
2. **分数抬起来了吗**：own 的相对分 z（相对"所有块"的分布）分布；z≥2 有几个样本。
3. **代价是什么**：**跨条目高分**有多少（块级匹配天然更容易撞车，A4-3 已证跨样本语义克隆真实存在）。

## 口径与近似（**必须写在结论旁边**）

* 用的是**代码↔代码**这一路：查询 = 样本的代码块（生产切块器），索引 = 库条目的**代码块**
  （`Remove incorrect logic` / `Ensure corrected path` / `problematic_pattern` 三段）。
  「LLM 语义描述 ↔ `llm_semantic`」那一路（prose↔prose）**本脚本没测**——
  它需要样本侧每个 issue 的 `llm_semantic`（在运行产物里，不在这份缓存里）。
* 嵌入一律走**生产实现**：索引侧 `_default_embed`（编码器 + 按层白化，不减查询偏移），
  查询侧 `_query_embed`（再减 C2 查询偏移）。**不重新拟合任何基**（重新拟合会作废既有基线）。
* 查询块来自 A5 缓存（`reports/a5_items_cache.json`，30 个库内样本 + 15 个库外）。

用法：
    python -X utf8 utils/experiments/precheck5_block_index.py
    python -X utf8 utils/experiments/precheck5_block_index.py --limit 30 --report 12
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

ITEMS = ROOT / "reports/a5_items_cache.json"
DB = ROOT / "reports/mas_rebuild_candidate_v2.db"
CACHE = ROOT / "reports/a7_block_embeddings.json"
ERR_MARK = "Remove incorrect logic:"
FIX_MARK = "Ensure corrected path:"


def kb_blocks(solution: str, pattern: str):
    """把一条知识的文本切成"代码块"（按知识条目自己的自然分段，不机械切窗口）。

    返回 [(block_kind, text), ...]：
      · `err`  = "Remove incorrect logic:" 那段（错误代码 = 针的来源）
      · `fix`  = "Ensure corrected path:" 那段（修复代码）
      · `pat`  = `problematic_pattern`（典型易错写法）
    """
    out = []
    sol = solution or ""
    for kind, mark in (("err", ERR_MARK), ("fix", FIX_MARK)):
        i = sol.find(mark)
        if i < 0:
            continue
        seg = sol[i + len(mark):]
        for cut in (";;", ERR_MARK, FIX_MARK):
            j = seg.find(cut)
            if j >= 0:
                seg = seg[:j]
        seg = seg.strip()
        if len(seg) >= 20:
            out.append((kind, seg))
    pat = (pattern or "").strip()
    if len(pat) >= 20:
        out.append(("pat", pat))
    return out


def cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=ITEMS)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--limit", type=int, default=30, help="只算前 N 个正样本（省时间）")
    ap.add_argument("--report", type=int, default=10, help="逐样本打印前 N 行")
    ap.add_argument("--layer", default="solution",
                    help="代码块走哪一层的白化（默认 solution —— 它的语料就是修复前后代码）")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import (
        AIDrivenSecondPassAnalysisAgent,
    )
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    # ---------- 1) 库侧：把每条知识切块 + 索引侧嵌入 ----------
    con = sqlite3.connect(str(args.db))
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT id, title, solution, problematic_pattern FROM issue_patterns").fetchall()
    con.close()
    blocks = []          # [(sid, kind, text)]
    for r in rows:
        for kind, text in kb_blocks(r["solution"], r["problematic_pattern"]):
            blocks.append((int(r["id"]), kind, text))
    print("库侧块数：%d（来自 %d 条知识，平均 %.1f 块/条）"
          % (len(blocks), len(rows), len(blocks) / max(1, len(rows))))

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    kb_vec = cache.setdefault("kb_blocks", {})
    for i, (sid, kind, text) in enumerate(blocks):
        key = "%d|%s" % (sid, kind)
        if key not in kb_vec:
            kb_vec[key] = agent._default_embed(text, args.layer)
        if i % 100 == 0:
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print("索引侧向量已就绪（缓存 %s）" % CACHE.name)

    kb_by_entry = {}
    for (sid, kind, text) in blocks:
        kb_by_entry.setdefault(sid, []).append(kb_vec["%d|%s" % (sid, kind)])

    # ---------- 2) 查询侧：样本的代码块 ----------
    items = json.loads(args.items.read_text(encoding="utf-8"))
    pos = [it for it in items if it.get("kind") == "pos"][: args.limit]
    print("样本（正）：%d 个，查询块合计 %d"
          % (len(pos), sum(len(it.get("chunks") or []) for it in pos)))

    qv = cache.setdefault("query_blocks", {})
    # 库条目的文件 basename（用于"只看同文件"这一档 —— 那才是门控实际会看到的范围）
    con = sqlite3.connect(str(args.db))
    con.row_factory = sqlite3.Row
    file_by_id = {int(r["id"]): (r["fp"] or "") for r in
                  con.execute("SELECT id, file_pattern AS fp FROM issue_patterns")}
    con.close()

    def base_of(p: str) -> str:
        p = str(p or "").replace("\\", "/").strip().lower()
        b = p.rsplit("/", 1)[-1]
        return b.replace("__", "/").rsplit("/", 1)[-1]

    per_sample = []
    block_argmax_own = 0
    block_total = 0
    for it in pos:
        cve = it["cve"]
        own = it.get("own")
        chunk_vecs = []
        for k, ch in enumerate(it.get("chunks") or []):
            text = ch.get("text") or ""
            if len(text.strip()) < 40:
                continue
            key = "%s|%d" % (cve, k)
            if key not in qv:
                qv[key] = agent._query_embed(text, args.layer)
            chunk_vecs.append(qv[key])
        if not chunk_vecs or own is None:
            continue
        all_sims = []            # 该样本所有(块×块)的相似度（用于 z 的分布）
        best_by_entry = {}
        for q in chunk_vecs:
            best_sid, best_v = None, -1.0
            for sid, vecs in kb_by_entry.items():
                best = max(cos(q, v) for v in vecs)
                all_sims.append(best)
                if best > best_by_entry.get(sid, -1.0):
                    best_by_entry[sid] = best
                if best > best_v:
                    best_sid, best_v = sid, best
            # 逐块 top-1 是否指向 own 条目（与 A4-2 同一口径，便于对照）
            block_total += 1
            block_argmax_own += int(best_sid == int(own))
        own_sim = best_by_entry.get(int(own))
        if own_sim is None:
            continue
        ranked = sorted(best_by_entry.items(), key=lambda kv: -kv[1])
        order = [s for s, _ in ranked]
        rank = order.index(int(own)) + 1
        # "只看同文件"：门控的跨文件守卫之后，own 只需在同文件条目里竞争
        own_base = base_of(file_by_id.get(int(own), ""))
        same = [s for s in order if base_of(file_by_id.get(s, "")) == own_base]
        rank_same = same.index(int(own)) + 1 if int(own) in same else -1
        mu = sum(all_sims) / len(all_sims)
        sd = (sum((x - mu) ** 2 for x in all_sims) / max(1, len(all_sims) - 1)) ** 0.5
        z = (own_sim - mu) / sd if sd > 1e-9 else 0.0
        strong_other = sum(1 for s, v in best_by_entry.items() if int(s) != int(own) and v >= own_sim)
        per_sample.append({"cve": cve, "own": int(own), "own_sim": own_sim, "rank": rank,
                           "rank_same_file": rank_same, "n_same_file": len(same),
                           "z": z, "mu": mu, "sd": sd, "n_blocks": len(blocks),
                           "strong_other": strong_other,
                           "over_070": sum(1 for _, v in best_by_entry.items() if v >= 0.70)})
    CACHE.write_text(json.dumps(cache), encoding="utf-8")

    # ---------- 3) 汇总 ----------
    n = len(per_sample)
    print("\n" + "=" * 104)
    print("块级索引（库侧按块） vs 条目级（A4 的对照）：own 条目的名次与相对分")
    print("=" * 104)
    print("%-16s %9s %7s %10s %10s %9s %10s %10s" %
          ("样本", "own 相似度", "名次", "同文件内名次", "同文件条目数", "z",
           "≥own 的别的条目", "≥0.70 的条目"))
    for r in sorted(per_sample, key=lambda x: x["rank_same_file"] if x["rank_same_file"] > 0 else 999)[: args.report]:
        print("%-16s %9.4f %7d %10s %10d %9.2f %10d %10d"
              % (r["cve"], r["own_sim"], r["rank"],
                 ("%d/%d" % (r["rank_same_file"], r["n_same_file"])) if r["rank_same_file"] > 0 else "不在同文件池",
                 r["n_same_file"], r["z"], r["strong_other"], r["over_070"]))

    top1 = sum(1 for r in per_sample if r["rank"] == 1)
    top5 = sum(1 for r in per_sample if r["rank"] <= 5)
    z2 = sum(1 for r in per_sample if r["z"] >= 2)
    z3 = sum(1 for r in per_sample if r["z"] >= 3)
    same1 = sum(1 for r in per_sample if r["rank_same_file"] == 1)
    print("\n--- 汇总（%d 个样本）---" % n)
    print("  逐块 top-1 指向 own  : %d/%d（%.0f%%）   ← A4-2 同口径是 30.3%%"
          % (block_argmax_own, block_total, 100.0 * block_argmax_own / max(1, block_total)))
    print("  own 条目 top-1（全库）: %d/%d（%.0f%%）" % (top1, n, 100.0 * top1 / max(1, n)))
    print("  own 条目 top-5（全库）: %d/%d（%.0f%%）" % (top5, n, 100.0 * top5 / max(1, n)))
    print("  **own 在同文件池内第 1**: %d/%d（%.0f%%）   ← 门控实际看到的范围（跨文件已被守卫拦掉）"
          % (same1, n, 100.0 * same1 / max(1, n)))
    print("  own 相对分 z ≥ 2      : %d/%d   ← A4 条目级下是 23/30" % (z2, n))
    print("  own 相对分 z ≥ 3      : %d/%d" % (z3, n))
    avg_other = sum(r["strong_other"] for r in per_sample) / max(1, n)
    print("  **代价**：平均每个样本有 %.1f 条**别的知识**块级相似度 ≥ own（块级撞车）" % avg_other)
    print("  绝对分 ≥0.70 的条目数（均值）: %.1f" % (sum(r["over_070"] for r in per_sample) / max(1, n)))

    out = ROOT / "reports/a7_block_index.json"
    out.write_text(json.dumps(per_sample, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已落盘: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
