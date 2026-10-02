#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""③ 之前的**离线设计研究**：LLM 配对判定怎么用才安全（纯统计，不需要 GPU/向量库）。

## 为什么要有这一步

A5b/A5c 已经证明两件事：

* LLM 判定比向量相对分**好**（λ=0.5、θ=0.70 → 正命中 **28/30**、跨文件 0；向量项 25/30；纯词元 22/30）；
* 但**判定器特异性不够**：负样本 **27%** 被判 `same_defect`，且 135 条里 `unrelated` 一次都没出现。
  于是 λ≥1.0 时跨文件误报从 7 炸到 39 —— 一条"满分的错误标签"就能单独越过门限。

⇒ 直接按 λ=0.5、θ=0.70 上线是**拿召回换误报**，而且是在 15 个负样本上验的。
  在花 GPU 之前，先把"**怎么用这个判定器才安全**"用现有缓存算清楚。

## 四种用法（veto = 附加否决条件）

| 模式 | 含义 | 动机 |
|---|---|---|
| `none` | 语义项无条件参与（= A5b 原口径，对照用） | 基线 |
| `has_anchor` | **只有带证据锚点**（针/同文件/类名在码/函数名在码）的候选才允许吃到语义加分 | 只用一条 LLM 标签就放行，等于绕开锚点 |
| `same_file` | **只有同文件**的候选才允许吃到语义加分 | 用户明确"跨文件闸门不开放"：语义只能在同文件内补位 |
| `needle_or_same` | 针命中或同文件之一 | 折中：既承认针（跨文件但强证据），也承认同文件 |

另外比较一个"分级"变体：`strict` = 只认 `same_defect`（把 `related` 记 0）。

## 输出

每个 (模式, λ, θ) 一行：正命中 / 负放行（同文件、跨文件）。
最后给出"跨文件 = 0 且正命中最高"的前几名 —— 这就是可以拿去 ③ 的候选配置。

用法：
    python -X utf8 utils/experiments/a5c_llm_veto_design.py
    python -X utf8 utils/experiments/a5c_llm_veto_design.py --topk 5     # 按生产候选池口径
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

from utils.experiments.a5_semantic_path_backtest import basename_of  # noqa: E402
from utils.experiments.a5b_graded_fusion import grade_item, load_kb_rows  # noqa: E402
from utils.experiments.a5c_universe_restriction import universe  # noqa: E402

ITEMS = ROOT / "reports/a5_items_cache.json"
VERDICTS = ROOT / "reports/a5b_llm_verdicts.json"
DUMP = ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl"

ANCHOR_FIELDS = {"needle", "file_identity", "class_in_code", "func_in_code"}


def semantic_term(fields, verdict, mode: str, strict: bool) -> float:
    """该候选的语义项（含否决条件）。verdict ∈ {1.0, 0.5, 0.0}。"""
    if strict and verdict < 1.0:
        return 0.0
    fs = set(fields or [])
    if mode == "none":
        pass
    elif mode == "has_anchor":
        if not (fs & ANCHOR_FIELDS):
            return 0.0
    elif mode == "same_file":
        if "file_identity" not in fs:
            return 0.0
    elif mode == "needle_or_same":
        if not (fs & {"needle", "file_identity"}):
            return 0.0
    else:
        raise ValueError(mode)
    return float(verdict)


def sem_value(item, sid, verd, source: str) -> float:
    """取该候选的语义原始分：LLM 判定（1/0.5/0）或向量相对分（z/4）。"""
    if source == "llm":
        return float((verd.get(item["cve"]) or {}).get(str(sid), 0.0) or 0.0)
    return float((item.get("s_sem_vec") or {}).get(str(sid), 0.0) or 0.0)


def evaluate(items, graded, verd, lam, theta, mode, strict, topk, kb_rows, source="llm"):
    pos = [i for i in items if i["kind"] == "pos"]
    neg = [i for i in items if i["kind"] == "neg"]
    own_adm = own_n = 0
    for i in pos:
        if not i.get("own"):
            continue
        own = int(i["own"])
        own_n += 1
        u = universe(i, topk, include_own=True)
        if own not in u:
            continue
        g = graded[i["cve"]].get(own) or graded[i["cve"]].get(str(own))
        if not g:
            continue
        v = sem_value(i, own, verd, source)
        if g["s_lex"] + lam * semantic_term(g["fields"], v, mode, strict) >= theta:
            own_adm += 1
    tot = same = cross = 0
    for i in neg:
        fbase = basename_of(i["file"])
        u = universe(i, topk, include_own=True)
        for sid, g in graded[i["cve"]].items():
            sid = int(sid)
            if sid not in u:
                continue
            v = sem_value(i, sid, verd, source)
            if g["s_lex"] + lam * semantic_term(g["fields"], v, mode, strict) >= theta:
                tot += 1
                kbase = basename_of((kb_rows.get(sid) or {}).get("file_pattern", ""))
                if kbase and fbase and kbase == fbase:
                    same += 1
                else:
                    cross += 1
    return {"own": own_adm, "own_n": own_n, "neg": tot, "same": same, "cross": cross}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=ITEMS)
    ap.add_argument("--verdicts", type=Path, default=VERDICTS)
    ap.add_argument("--dump", type=Path, default=DUMP)
    ap.add_argument("--topk", type=int, default=5,
                    help="生产候选池口径（{针命中} ∪ {相似度前 K}）；0 或负数 = 全库")
    ap.add_argument("--sem", choices=("llm", "vector", "both"), default="both",
                    help="语义项来源：llm=配对判定；vector=向量相对分（z/4）")
    args = ap.parse_args()

    items = json.loads(args.items.read_text(encoding="utf-8"))
    for it in items:
        it["sims"] = {int(k): v for k, v in (it.get("sims") or {}).items()}
    kb_rows = load_kb_rows(args.dump)
    verd = json.loads(args.verdicts.read_text(encoding="utf-8"))
    # 向量相对分（z/4，封顶 1）：与 A5b 的 s_sem_vec 完全同一套算法
    from utils.experiments.a5b_graded_fusion import add_vector_sem
    items = add_vector_sem(items)

    from core.agents.ai_driven_second_pass_analysis_agent import (
        AIDrivenSecondPassAnalysisAgent,
    )
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    graded = {it["cve"]: grade_item(agent, it, kb_rows) for it in items}
    topk = None if args.topk <= 0 else args.topk

    print("样本 %d（正 %d / 负 %d），候选池口径 = %s，判定缓存 %d 条"
          % (len(items), sum(1 for i in items if i["kind"] == "pos"),
             sum(1 for i in items if i["kind"] == "neg"),
             "全库" if topk is None else ("{针} ∪ {前 %d}" % topk),
             sum(len(v) for v in verd.values())))

    rows = []
    sources = ("llm", "vector") if args.sem == "both" else (args.sem,)
    for source in sources:
        print("\n" + "=" * 92)
        print("语义项来源 = %s" % ("LLM 配对判定" if source == "llm" else "向量相对分 z/4"))
        print("=" * 92)
        print("%-14s %-5s %10s %8s %8s %8s" % ("模式", "严格", "(λ,θ)", "正命中", "同文件", "跨文件"))
        for mode in ("none", "has_anchor", "same_file", "needle_or_same"):
            for strict in (False, True):
                for lam in (0.3, 0.5, 0.7, 1.0, 1.5):
                    for theta in (0.65, 0.7, 0.8, 0.9, 1.0, 1.1):
                        r = evaluate(items, graded, verd, lam, theta, mode, strict, topk,
                                     kb_rows, source)
                        rows.append({"source": source, "mode": mode, "strict": strict,
                                     "lam": lam, "theta": theta, **r})
        sub = sorted([r for r in rows if r["source"] == source],
                     key=lambda r: (-r["cross"], -r["own"], r["neg"]))
        for r in sub[:10]:
            print("%-14s %-5s %10s %6d/%-2d %8d %8d"
                  % (r["mode"], "是" if r["strict"] else "否",
                     "λ=%.1f,θ=%.2f" % (r["lam"], r["theta"]),
                     r["own"], r["own_n"], r["same"], r["cross"]))
        print("  --- 判据：跨文件 = 0 且正命中最高 ---")
        best = [r for r in sub if r["cross"] == 0]
        best.sort(key=lambda r: (-r["own"], r["neg"]))
        for r in best[:6]:
            print("    正 %2d/%-2d  负放行 %3d（同文件 %2d / 跨文件 %d）  %s%s  λ=%.1f θ=%.2f"
                  % (r["own"], r["own_n"], r["neg"], r["same"], r["cross"],
                     r["mode"], "（只认 same）" if r["strict"] else "", r["lam"], r["theta"]))

    out = ROOT / "reports/a5c_llm_veto_design.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已落盘: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
