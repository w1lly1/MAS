#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 A5b 的融合扫描**限制到"生产真正会看到的候选集合"**上重算一遍（离线，纯统计）。

## 为什么必须先做这一步

A5b/A5 的离线回测是在**全库 200 条**上算分数的：`admit ⇔ s_lex + λ·s_sem ≥ θ`，
候选池 = 200 条。于是"负放行 52 条 / 跨文件 39 条"这类数字，是**在 200 条里**数出来的。

但生产里 `second_pass_analysis_agent.weaviate_top_k = 5`：
向量通道每层只取前 5，四条层合并后通常只有 5~15 个 sqlite_id 进入门控；
其余通道（curated_issue / sqlite）是**词法命中**才进来。

⇒ 生产的候选池 = {词法命中(针)的条目} ∪ {向量 top-K}，
  **不是**全库。离线表里的收益因此可能**不可兑现**：门控再会算分，没检索到的条目也进不来。

本脚本就是用已有缓存把这个差距量出来：同一套 (λ, θ) 在两种候选池上各算一次。

    U_full  = 全库 200 条                       （= 《02》§16/§23 那张表的口径）
    U_prod  = {lexical 命中} ∪ {sims 前 K 名}    （= 生产口径的代理，K 可调）

只读缓存，不需要模型/向量库。用法：
    python -X utf8 utils/experiments/a5c_universe_restriction.py
    python -X utf8 utils/experiments/a5c_universe_restriction.py --topk 5 10 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from utils.experiments.a5b_graded_fusion import (  # noqa: E402
    W_BASE, add_vector_sem, grade_item, load_kb_rows,
)
from utils.experiments.a5_semantic_path_backtest import basename_of  # noqa: E402

CACHE_ITEMS = ROOT / "reports/a5_items_cache.json"
CACHE_LLM = ROOT / "reports/a5b_llm_verdicts.json"
DUMP = ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl"


def universe(item, topk: int | None, include_own: bool) -> set:
    """生产口径的候选池代理：词法(针)命中 ∪ sims 前 topk 名；topk=None → 全库。"""
    sids = {int(s) for s in item["sims"]}
    if topk is None:
        return set(sids)
    ranked = sorted(item["sims"].items(), key=lambda kv: -kv[1])[:topk]
    u = {int(s) for s, _ in ranked}
    u |= {int(s) for s, hit in (item.get("lexical") or {}).items() if hit}
    if include_own and item.get("own"):
        u.add(int(item["own"]))
    return u


def evaluate(items, graded, sem, lam, theta, topk, include_own, kb_rows):
    pos = [i for i in items if i["kind"] == "pos"]
    neg = [i for i in items if i["kind"] == "neg"]
    own_in_u = own_adm = 0
    for i in pos:
        if not i.get("own"):
            continue
        u = universe(i, topk, include_own)
        own = int(i["own"])
        if own in u:
            own_in_u += 1
            g = graded[i["cve"]].get(own)
            if g is None:
                # 分级表按 sims 全量算，故一定存在
                g = graded[i["cve"]][str(own)]
            s_sem = float((sem.get(i["cve"]) or {}).get(str(own), 0.0) or 0.0)
            if g["s_lex"] + lam * s_sem >= theta:
                own_adm += 1
    tot = same = cross = 0
    for i in neg:
        u = universe(i, topk, include_own)
        fbase = basename_of(i["file"])
        for sid, g in graded[i["cve"]].items():
            sid = int(sid)
            if sid not in u:
                continue
            s_sem = float((sem.get(i["cve"]) or {}).get(str(sid), 0.0) or 0.0)
            if g["s_lex"] + lam * s_sem >= theta:
                tot += 1
                kbase = basename_of((kb_rows.get(sid) or {}).get("file_pattern", ""))
                if kbase and fbase and kbase == fbase:
                    same += 1
                else:
                    cross += 1
    return {"pos_n": len(pos), "own_in_u": own_in_u, "own_adm": own_adm,
            "neg": tot, "same": same, "cross": cross}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--no-force-own", action="store_true",
                    help="不把『自己的条目』硬塞进候选池 —— 这才是生产里最诚实的口径"
                         "（生产必须先检索到它；本脚本默认的 include_own=True 是乐观假设）")
    ap.add_argument("--items", type=Path, default=CACHE_ITEMS)
    ap.add_argument("--verdicts", type=Path, default=CACHE_LLM)
    ap.add_argument("--dump", type=Path, default=DUMP)
    args = ap.parse_args()

    items = json.loads(args.items.read_text(encoding="utf-8"))
    for it in items:
        it["sims"] = {int(k): v for k, v in (it.get("sims") or {}).items()}
    kb_rows = load_kb_rows(args.dump)
    items = add_vector_sem(items)                       # s_sem_vec（z/4，封顶 1）
    verd = json.loads(args.verdicts.read_text(encoding="utf-8"))
    for it in items:
        it["s_sem_llm"] = {str(s): v for s, v in (verd.get(it["cve"]) or {}).items()}

    from core.agents.ai_driven_second_pass_analysis_agent import (
        AIDrivenSecondPassAnalysisAgent,
    )
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    graded = {it["cve"]: grade_item(agent, it, kb_rows) for it in items}

    print("样本 %d（正 %d / 负 %d）" % (len(items), sum(1 for i in items if i["kind"] == "pos"),
                                        sum(1 for i in items if i["kind"] == "neg")))
    lex_cov = sum(1 for i in items if i["kind"] == "pos"
                  and i.get("own") and (i.get("lexical") or {}).get(str(i["own"])) is not None
                  and (i.get("lexical") or {}).get(str(int(i["own"]))) )
    print("正样本里『自己的条目被针命中(词法)』= %d/%d" % (lex_cov, sum(1 for i in items if i["kind"] == "pos")))
    print("⇒ 生产里这类样本的 own 会经 curated_issue/sqlite 词法通道进入候选池\n")

    for sem_name, sem_key in (("向量 z 归一化", "s_sem_vec"), ("LLM 配对判定", "s_sem_llm")):
        print("=" * 96)
        print("语义项 = %s" % sem_name)
        print("=" * 96)
        for lam, theta in ((0.0, 0.65), (0.5, 0.65), (0.5, 0.70), (0.5, 0.90), (1.0, 1.00), (1.0, 1.20)):
            row = []
            for topk, label in [(None, "全库200")] + [(k, "top%-2d" % k) for k in args.topk]:
                r = evaluate(items, graded, {i["cve"]: i[sem_key] for i in items},
                             lam, theta, topk, not args.no_force_own, kb_rows)
                row.append("%s: own %2d/%2d(池内%2d) 负%3d(跨%3d)"
                           % (label, r["own_adm"], r["pos_n"], r["own_in_u"], r["neg"], r["cross"]))
            print("  λ=%.1f θ=%.2f  %s" % (lam, theta, " | ".join(row)))
        print()

    print("=" * 96)
    print("口径说明")
    print("=" * 96)
    print("  · 「全库200」= 离线回测口径（《02》§16/§23 那张表），生产**取不到**这么多候选；")
    print("  · 「topK」= 生产口径代理：{针命中的条目} ∪ {相似度前 K 名}（weaviate_top_k=5，四层合并）；")
    print("  · own 一律计入候选池（对生产是**乐观**假设：生产还要先检索到它）；")
    print("  · 权重表 W_BASE = %s" % W_BASE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
