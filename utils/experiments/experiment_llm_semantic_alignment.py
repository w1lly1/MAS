# -*- coding: utf-8 -*-
"""决定性验证：把真实 LLM 生成的语义文本装进索引/查询两侧，检索排名到底变不变？

## 这次测的是"真东西"，不是假设值

前面的规则对齐实验用的是**该 CVE 自己的英文摘要**当"如果 LLM 完全说对"的上界（有循环论证成分）。
这次用的是**真的跑了 Qwen1.5-7B** 产出的文本：

* 索引侧：`reports/llm_semantic.json`（200 条里成功 188 条），由**漏洞窗口**的代码生成；
* 查询侧：`reports/llm_semantic_query8.json`（8/8 成功），由**流水线自己切出来的分片**生成，
  而且**没有告诉模型弱点家族**（让它自己判断）——
  两边输入不同、措辞不同，**不是同一段文本**，所以不构成循环论证。

## 两个变量，做一个 2×2

    index ∈ {旧文本, 新文本（+llm_semantic）}
    query ∈ {代码分片（现状）, LLM 语义描述（拟改）}

指标：**自己那条知识**在 200 条里的排名 / 是否进 top-5（线上每层只取 5 条）。

## 还要单独看一件事：C2 偏移要不要重拟

`_query_embed` 会减掉一个**在旧查询分布上拟合**的偏移（C2 修复）。换上全新写法的查询文本后，
那个偏移就是**外分布**的了。所以这里同时给出"带偏移/不带偏移"两组数字——
如果带偏移明显更差，说明换查询文本这件事**必须连带重拟偏移**，不能只改文本。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

DUMP = ROOT / "utils/experiments/weaviate_kb_dump.jsonl"
LAYERS = ("semantic", "code_pattern", "solution", "full")
CHUNKS = ROOT / "reports/smoke_chunks.json"


def load_dump() -> dict:
    out = {L: {} for L in LAYERS}
    with open(DUMP, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            L = str(r.get("vector_layer") or "")
            if L in out and isinstance(r.get("_vector"), list) and r["_vector"]:
                out[L][int(r["sqlite_id"])] = [float(x) for x in r["_vector"]]
    return out


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def rank_of(q, own_id: int, index: dict):
    own = index.get(own_id)
    if own is None:
        return None, None
    s_own = dot(q, own)
    others = [dot(q, v) for sid, v in index.items() if sid != own_id]
    return 1 + sum(1 for s in others if s > s_own), s_own


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--semantic", type=Path, default=ROOT / "reports/llm_semantic.json")
    ap.add_argument("--query-semantic", type=Path, default=ROOT / "reports/llm_semantic_query8.json")
    ap.add_argument("--chunks", type=Path, default=CHUNKS)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/llm_semantic_alignment.json")
    args = ap.parse_args()

    idx_sem = json.loads(args.semantic.read_text(encoding="utf-8"))
    qry_sem = json.loads(args.query_semantic.read_text(encoding="utf-8"))
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    print("索引侧语义文本 %d 条；查询侧描述 %d 条；分片 %d 条"
          % (len(idx_sem), len(qry_sem), len(chunks)))

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    from infrastructure.database.weaviate.service import WeaviateVectorService
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    svc = WeaviateVectorService()

    con = sqlite3.connect(str(args.db))
    kb = {int(i): {"cve": (t or "").strip().upper(), "error_type": et or "",
                   "severity": sev or "", "language": lang or "", "framework": fw or "",
                   "error_description": ed or "", "problematic_pattern": pp or "",
                   "solution": sol or "", "file_pattern": fp or "", "class_pattern": cp or "",
                   "llm_semantic": idx_sem.get((t or "").strip().upper(), "")}
          for i, t, et, sev, lang, fw, ed, pp, sol, fp, cp in con.execute(
              "select id, title, error_type, severity, language, framework, error_description, "
              "problematic_pattern, solution, file_pattern, class_pattern from issue_patterns")}
    con.close()
    by_cve = {v["cve"]: (k, v) for k, v in kb.items()}

    dump = load_dump()

    # ---- 索引侧：新文本重编码 semantic / full（其余两层不受影响，沿用 dump 的向量）----
    print("正在用新文本重编码 semantic / full 两层的索引向量（本地 CPU）...")
    index_old = {"semantic": dict(dump["semantic"]), "code_pattern": dict(dump["code_pattern"]),
                 "solution": dict(dump["solution"]), "full": dict(dump["full"])}
    index_new = {L: dict(index_old[L]) for L in LAYERS}
    n_replaced = 0
    for sid, rec in kb.items():
        props = dict(rec, sqlite_id=sid, status="active")
        for L in ("semantic", "full"):
            text = svc._build_enhanced_issue_pattern_text(props, L)
            vec = agent._default_embed(text, L)      # 索引侧 = 白化后（不减偏移）
            index_new[L][sid] = vec
        if rec["llm_semantic"]:
            n_replaced += 1
    print("  重编码完成；其中 %d 条带 llm_semantic" % n_replaced)

    # ---- 忠实性自检：用旧文本重编码，应与 dump 里的向量几乎一致 ----
    chk_cve = next(iter(chunks))
    chk_id, chk_rec = by_cve[chk_cve]
    cos = []
    for L in ("semantic", "full"):
        props = dict(chk_rec, sqlite_id=chk_id, status="active", llm_semantic="")
        v = agent._default_embed(svc._build_enhanced_issue_pattern_text(props, L), L)
        dv = index_old[L][chk_id]
        cos.append(dot(v, dv))
    print("  [自检] 旧文本重编码 vs dump 向量的余弦: semantic=%.4f full=%.4f "
          "（接近 1 说明复现口径一致）" % (cos[0], cos[1]))

    # ---- 查询侧：两种查法 × 两种索引 ----
    def q_code(layer, cve):
        c = chunks[cve]
        iss = {"source": "source_code_chunk",
               "description": "source_code_chunk L%s-%s: %s" % (c.get("start"), c.get("end"), c["code"]),
               "code_snippet": c["code"], "file": kb[by_cve[cve][0]]["file_pattern"], "severity": "medium"}
        return agent._build_query_text(iss, kb[by_cve[cve][0]]["file_pattern"])

    def q_llm(layer, cve):
        iss = {"source": "source_code_chunk",
               "description": qry_sem.get(cve, ""),
               "code_snippet": "", "file": kb[by_cve[cve][0]]["file_pattern"], "severity": "medium"}
        return agent._build_query_text(iss, kb[by_cve[cve][0]]["file_pattern"])

    results = {}
    for off in (True, False):
        agent.query_offset_correction = off
        tag = "带C2偏移" if off else "不带偏移"
        for qname, qfn in (("代码分片(现状)", q_code), ("LLM语义(拟改)", q_llm)):
            for iname, index in (("旧索引", index_old), ("新索引", index_new)):
                hit5 = 0
                tot = 0
                rows = []
                for cve in chunks:
                    own, _ = by_cve[cve]
                    per = {}
                    for L in LAYERS:
                        text = qfn(L, cve)
                        q = agent._query_embed(text, L)
                        r, s = rank_of(q, own, index[L])
                        per[L] = {"rank": r, "sim": round(s, 4) if s else None}
                        tot += 1
                        hit5 += int(r is not None and r <= 5)
                    rows.append({"cve": cve, "layers": per})
                results[f"{tag}|{qname}|{iname}"] = {
                    "top5": hit5, "total": tot, "rows": rows}

    print("\n" + "=" * 104)
    print("结果：2×2（各 8 样本 × 4 层 = 32 次查询）-  数字=进 top-5 的层数")
    print("=" * 104)
    print("  %-14s %-16s %-12s %-12s" % ("偏移", "查询", "旧索引", "新索引(+llm_semantic)"))
    for off, tag in ((True, "带C2偏移"), (False, "不带偏移")):
        for qname in ("代码分片(现状)", "LLM语义(拟改)"):
            a = results[f"{tag}|{qname}|旧索引"]
            b = results[f"{tag}|{qname}|新索引"]
            print("  %-14s %-16s %-12s %-12s" % (
                tag, qname, "%d/%d" % (a["top5"], a["total"]), "%d/%d" % (b["top5"], b["total"])))

    print("\n  逐样本（不带偏移、新索引）:")
    print("  %-16s %-26s %-26s" % ("CVE", "代码分片(现状)", "LLM语义(拟改)"))
    base = {r["cve"]: r for r in results["不带偏移|代码分片(现状)|新索引"]["rows"]}
    new = {r["cve"]: r for r in results["不带偏移|LLM语义(拟改)|新索引"]["rows"]}
    for cve in chunks:
        def fmt(r):
            return " ".join("%s%s" % (L[:4], r["layers"][L]["rank"]) for L in LAYERS)
        print("  %-16s %-26s %-26s" % (cve, fmt(base[cve]), fmt(new[cve])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写出: %s" % args.out)
    print("\n  怎么读：① 新索引 vs 旧索引 → 装入 llm_semantic 有没有用；")
    print("          ② LLM语义 vs 代码分片 → 换查询写法有没有用；")
    print("          ③ 带偏移 vs 不带偏移 → 换查询文本后 C2 偏移是不是得重拟。")


if __name__ == "__main__":
    main()
