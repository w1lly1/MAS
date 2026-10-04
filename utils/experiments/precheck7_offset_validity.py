#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**C2 查询偏移还有效吗？** —— 在"活的 embedder"下重测（本地，纯离线）。

## 为什么要重测（背景）

C2 修复（`z = L2((q − mean)@W − offset_layer)`）当初的**拟合语料是服务器 dump 的"真实查询向量"**，
而今天查明：**那些批次里 embedder 是死的**（模型没加载 → 退回"校验和兜底向量" ⇒ 查询向量几乎是常量）。
于是：

* "拟合在 A、应用到 B，top10 0.718→0.260、命中行 60→143/200"这些**跨批泛化证据，是在常量向量上算的** ⇒ 无效；
* 更要紧的是：**把一个在常量向量上拟合出来的偏移，减到真实查询向量上，可能是"帮倒忙"**。

## 本脚本怎么判

同一批样本、同一套库向量（dump 里真实存的、已白化的），只改**查询侧**：

* **A 组（现状）**：`_query_embed(text, 层)` = 白化 **再减偏移**；
* **B 组（去掉 C2）**：`_default_embed(text, 层)` = **只白化不减偏移**。

比 own 条目的名次 / 相对分 z / "有多少别的条目分数 ≥ own"，以及**绝对相似度上限**。

用法：
    python -X utf8 utils/experiments/precheck7_offset_validity.py --limit 30
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

DUMP = ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl"
DB = ROOT / "reports/mas_rebuild_candidate_v2.db"
LAYERS = ("semantic", "code_pattern", "solution", "full")


def cos(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def load_kb():
    rows = {}
    for line in DUMP.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        L = str(r.get("vector_layer") or "").lower()
        v = r.get("_vector") or r.get("vector")
        if L in LAYERS and v:
            rows.setdefault(L, []).append((int(r["sqlite_id"]), [float(x) for x in v]))
    return rows


def harvest(runs_file: Path, limit: int):
    out = []
    for line in runs_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cve = line.split("/")[0]
        issues = []
        for f in glob.glob(str(ROOT / "reports/analysis" / line / "second_pass" / "**" / "*_r2.json"),
                           recursive=True):
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


def evaluate(agent, kb, id_by_title, samples, use_offset: bool):
    per = []
    for cve, issues in samples:
        own = id_by_title.get(cve.upper())
        if own is None:
            continue
        vecs = []
        for it in issues:
            text = agent._build_query_text(it, it.get("_run_file") or "", 
                                           str(it.get("description") or ""),
                                           str(it.get("source") or ""))
            for L in LAYERS:
                vecs.append((L, (agent._query_embed(text, L) if use_offset
                                 else agent._default_embed(text, L))))
        all_sims, best = [], {}
        for L, q in vecs:
            for sid, v in kb.get(L, []):
                s = (1.0 + cos(q, v)) / 2.0
                all_sims.append(s)
                if s > best.get(sid, -1.0):
                    best[sid] = s
        if own not in best:
            continue
        order = sorted(best.items(), key=lambda kv: -kv[1])
        rank = [s for s, _ in order].index(own) + 1
        mu = sum(all_sims) / len(all_sims)
        sd = (sum((x - mu) ** 2 for x in all_sims) / max(1, len(all_sims) - 1)) ** 0.5
        per.append({"cve": cve, "own": best[own], "rank": rank,
                    "z": (best[own] - mu) / sd if sd > 1e-9 else 0.0,
                    "others_ge": sum(1 for s, v in best.items() if s != own and v >= best[own]),
                    "max_sim": max(best.values())})
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "reports/arm1_runs.txt")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import (
        AIDrivenSecondPassAnalysisAgent,
    )
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    kb = load_kb()
    con = sqlite3.connect("file:%s?mode=ro" % DB.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    con.close()

    samples = harvest(args.runs, args.limit)
    print("样本（有 llm_semantic 的）: %d" % len(samples))
    res = {}
    for name, flag in (("A 现状（白化 − 偏移）", True), ("B 去掉 C2（只白化）", False)):
        rows = evaluate(agent, kb, id_by_title, samples, flag)
        res[name] = rows
        ranks = [r["rank"] for r in rows]
        zs = [r["z"] for r in rows]
        print("\n--- %s（%d 个样本）---" % (name, len(rows)))
        print("  own top-1 : %d    top-5 : %d    z≥2 : %d    z 中位 %.2f"
              % (sum(1 for x in ranks if x == 1), sum(1 for x in ranks if x <= 5),
                 sum(1 for z in zs if z >= 2),
                 sorted(zs)[len(zs) // 2] if zs else 0))
        print("  own 相似度 中位 %.4f   最高 %.4f   别的条目≥own 中位 %.0f"
              % (sorted(r["own"] for r in rows)[len(rows) // 2],
                 max(r["max_sim"] for r in rows),
                 sorted(r["others_ge"] for r in rows)[len(rows) // 2]))

    a, b = res["A 现状（白化 − 偏移）"], res["B 去掉 C2（只白化）"]
    print("\n--- 逐样本对照（A 现状 vs B 去掉 C2）---")
    print("%-16s %14s %14s %10s %10s" % ("样本", "A名次/z", "B名次/z", "A own", "B own"))
    for x, y in list(zip(a, b))[:12]:
        print("%-16s %7d/%6.2f %7d/%6.2f %10.4f %10.4f"
              % (x["cve"], x["rank"], x["z"], y["rank"], y["z"], x["own"], y["own"]))
    better = sum(1 for x, y in zip(a, b) if y["z"] > x["z"])
    print("\nB（去掉 C2）相对分更高的样本数: %d / %d" % (better, len(a)))
    print("判读：若 B 明显更好 ⇒ **C2 偏移在真实向量上是帮倒忙**（它当初是拿常量向量拟合的）；")
    print("      若两者接近 ⇒ 该偏移中性；若 A 明显更好 ⇒ C2 仍然有效（需进一步复核拟合语料）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
