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
    ap.add_argument("--semantic", type=Path, default=ROOT / "reports/ls_index_v2.json",
                    help="索引侧的 llm_semantic（v2：带家族选择）")
    ap.add_argument("--families", type=Path, default=ROOT / "reports/ls_index_fam_v2.json",
                    help="索引侧模型选定的家族（用于统计与库分类的一致率）")
    ap.add_argument("--query-semantic", type=Path, default=ROOT / "reports/ls_query_v2.json",
                    help="查询侧描述，键为 `CVE@分片序号`")
    ap.add_argument("--query-families", type=Path, default=ROOT / "reports/ls_query_fam_v2.json")
    ap.add_argument("--all-chunks", type=Path, default=ROOT / "reports/smoke_allchunks.json",
                    help="每个样本的全部分片（做聚合实验要用）")
    ap.add_argument("--chunks", type=Path, default=CHUNKS)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/llm_semantic_alignment.json")
    args = ap.parse_args()

    idx_sem = json.loads(args.semantic.read_text(encoding="utf-8"))
    qry_sem_all = json.loads(args.query_semantic.read_text(encoding="utf-8"))
    idx_fam = json.loads(args.families.read_text(encoding="utf-8")) if args.families.exists() else {}
    qry_fam_all = (json.loads(args.query_families.read_text(encoding="utf-8"))
                   if args.query_families.exists() else {})
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    allchunks = (json.loads(args.all_chunks.read_text(encoding="utf-8"))
                 if args.all_chunks.exists() else {})

    # 把"漏洞分片"在"全部分片"里的序号找出来：查询侧主实验用漏洞那一分片的描述，
    # 聚合实验用全部。两边文本来自同一次抽取，取前 200 字符比对即可。
    vuln_key = {}
    for cve, cs in allchunks.items():
        target = str(chunks.get(cve, {}).get("code") or "")[:200]
        for i, c in enumerate(cs):
            if target and str(c.get("text") or "")[:200] == target:
                vuln_key[cve] = "%s@%d" % (cve, i)
                break
        vuln_key.setdefault(cve, "%s@0" % cve)
    qry_sem = {cve: qry_sem_all.get(k, "") for cve, k in vuln_key.items()}
    qry_fam = {cve: qry_fam_all.get(k, "") for cve, k in vuln_key.items()}

    print("索引侧语义文本 %d 条；查询侧描述 %d 条（漏洞分片定位到 %d/%d）；分片 %d 条"
          % (len(idx_sem), len(qry_sem_all), len(vuln_key), len(chunks), len(chunks)))

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
    # 约定：**所有查询构造函数都返回"查询向量"**（内部自己调 _query_embed）。
    # 第一版让 q_code/q_llm 返回文本、由调用方再编码，结果聚合类的函数返回向量、
    # 文本类的返回字符串，混在一个循环里直接崩（字符串被当成向量去点乘）。
    def _code_query_text(cve, code, start=None, end=None):
        """按**线上真实的模板**拼代码查询（description 带分片头，并且填 `snippet:` 位）。

        这一点必须严格：现状基线就是"拿每个分片的原始代码去查"。
        如果基线少填了 `snippet:` 这一位，就会把现状测得偏低、把改动效果测得偏高 ——
        任何"新方案更好"的结论都会变得不可信。
        """
        iss = {"source": "source_code_chunk",
               "description": "source_code_chunk L%s-%s: %s" % (start, end, code),
               "code_snippet": code, "file": kb[by_cve[cve][0]]["file_pattern"],
               "severity": "medium"}
        return agent._build_query_text(iss, kb[by_cve[cve][0]]["file_pattern"])

    def q_code(layer, cve):
        c = chunks[cve]
        return agent._query_embed(
            _code_query_text(cve, c["code"], c.get("start"), c.get("end")), layer)

    def _llm_query_text(cve, fam="", desc=None, key=None):
        """LLM 语义查询文本。

        `fam` 非空时把模型**自己选定的弱点家族**作为 `error_type:` 拼进去 ——
        索引层文本里有 `[error_type] <家族>` 这一行，查询侧带同一套分类编号，
        两侧分类空间才对得上（本轮改动②的目的）。
        """
        d = desc if desc is not None else qry_sem.get(cve, "")
        if fam:
            d = "error_type: %s\n%s" % (fam, d)
        iss = {"source": "source_code_chunk", "description": d, "code_snippet": "",
               "file": kb[by_cve[cve][0]]["file_pattern"], "severity": "medium"}
        return agent._build_query_text(iss, kb[by_cve[cve][0]]["file_pattern"])

    def q_llm(layer, cve, with_family: bool = True):
        return agent._query_embed(
            _llm_query_text(cve, qry_fam.get(cve, "") if with_family else ""), layer)

    def _chunk_keys(cve):
        return [k for k in qry_sem_all if k.startswith(cve + "@")]

    def q_llm_mean(layer, cve):
        """聚合③：同一文件**所有分片**的描述向量取平均。"""
        keys = _chunk_keys(cve)
        if not keys:
            return q_llm(layer, cve)
        vecs = [agent._query_embed(
            _llm_query_text(cve, qry_fam_all.get(k, ""), qry_sem_all[k]), layer) for k in keys]
        n = len(vecs)
        mean = [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]
        nrm = sum(x * x for x in mean) ** 0.5
        return mean if nrm <= 0 else [x / nrm for x in mean]

    def q_llm_best(layer, cve):
        """聚合上界：取各分片描述里"对自己条目排名最好"的那个（线上做不到，仅标定）。"""
        own = by_cve[cve][0]
        best = None
        for k in _chunk_keys(cve):
            q = agent._query_embed(
                _llm_query_text(cve, qry_fam_all.get(k, ""), qry_sem_all[k]), layer)
            r, _s = rank_of(q, own, index_new[layer])
            if r is not None and (best is None or r < best[0]):
                best = (r, q)
        return best[1] if best else q_llm(layer, cve)

    results = {}
    for off in (True, False):
        agent.query_offset_correction = off
        tag = "带C2偏移" if off else "不带偏移"
        for qname, qfn in (("代码分片(现状)", q_code), ("LLM语义(拟改)", q_llm),
                          ("LLM语义+家族", lambda L, c: q_llm(L, c, True)),
                          ("LLM各分片平均", q_llm_mean),
                          ("LLM取最好分片(上界)", q_llm_best)):
            for iname, index in (("旧索引", index_old), ("新索引", index_new)):
                hit5 = 0
                tot = 0
                rows = []
                for cve in chunks:
                    own = by_cve[cve][0]
                    per = {}
                    for L in LAYERS:
                        q = qfn(L, cve)
                        r, s = rank_of(q, own, index[L])
                        per[L] = {"rank": r, "sim": round(s, 4) if s else None}
                        tot += 1
                        hit5 += int(r is not None and r <= 5)
                    rows.append({"cve": cve, "layers": per})
                results[f"{tag}|{qname}|{iname}"] = {
                    "top5": hit5, "total": tot, "rows": rows}

        # 并集口径：**这才是线上真实发生的事** —— 每个分片各自查一次，命中结果再合并。
        # 所以"只要有任一分片的描述把自己的条目查进 top-5"就算命中，而不是取平均。
        #
        # 注意：**现状那一行也必须用并集**。线上是拿"每个分片的原始代码"去查的，
        # 不是只查漏洞那一段；只测单个分片会把现状测得偏低、把改动效果测得偏高。
        for iname, index in (("旧索引", index_old), ("新索引", index_new)):
            for qmode in ("code", "llm"):
                hit5 = 0
                tot = 0
                rows = []
                for cve in chunks:
                    own = by_cve[cve][0]
                    c = chunks[cve]
                    per = {}
                    for L in LAYERS:
                        best = None
                        if qmode == "code":
                            # 现状：**每个分片**都用原始代码、按线上模板查一遍
                            cands = [_code_query_text(cve, c["text"], c.get("start"), c.get("end"))
                                     for c in allchunks.get(cve, [])] or [
                                     _code_query_text(cve, c["code"], c.get("start"), c.get("end"))]
                        else:
                            cands = [_llm_query_text(cve, qry_fam_all.get(k, ""), qry_sem_all.get(k, ""))
                                     for k in (_chunk_keys(cve) or [vuln_key.get(cve)])]
                        for txt in cands:
                            if not txt:
                                continue
                            q = agent._query_embed(txt, L)
                            r, _s = rank_of(q, own, index[L])
                            if r is not None and (best is None or r < best):
                                best = r
                        per[L] = {"rank": best}
                        tot += 1
                        hit5 += int(best is not None and best <= 5)
                    rows.append({"cve": cve, "layers": per})
                results[f"{tag}|{'代码各分片取并集' if qmode == 'code' else 'LLM各分片取并集'}|{iname}"] = {
                    "top5": hit5, "total": tot, "rows": rows}

    print("\n" + "=" * 104)
    print("结果（各 8 样本 × 4 层 = 32 次查询）-  数字=进 top-5 的层数")
    print("=" * 104)
    print("  %-10s %-22s %-12s %-12s" % ("偏移", "查询", "旧索引", "新索引(+llm_semantic)"))
    for off, tag in ((True, "带C2偏移"), (False, "不带偏移")):
        for qname in ("代码分片(现状)", "代码各分片取并集", "LLM语义(拟改)", "LLM语义+家族",
                      "LLM各分片平均", "LLM各分片取并集", "LLM取最好分片(上界)"):
            a = results[f"{tag}|{qname}|旧索引"]
            b = results[f"{tag}|{qname}|新索引"]
            print("  %-10s %-22s %-12s %-12s" % (
                tag, qname, "%d/%d" % (a["top5"], a["total"]), "%d/%d" % (b["top5"], b["total"])))

    print("\n  家族一致性:")
    if idx_fam:
        with_fam = [c for _id, rec in kb.items() if idx_fam.get(rec["cve"])]
        agree = sum(1 for _id, rec in kb.items()
                    if idx_fam.get(rec["cve"]) and idx_fam[rec["cve"]] == rec["error_type"])
        print("    索引侧（**已告诉它库里的分类**，主要反映遵从度）: %d/%d = %.1f%%"
              % (agree, len(with_fam), 100 * agree / max(1, len(with_fam))))
    if qry_fam:
        # 这一项才是**有信息量**的：查询侧没告诉它答案，是模型自己从 7 类里选的
        rowsq = [(cve, qry_fam.get(cve, ""), kb[by_cve[cve][0]]["error_type"]) for cve in chunks]
        got = [r for r in rowsq if r[1]]
        agree = sum(1 for _c, a, b in got if a == b)
        print("    查询侧（**没告诉它答案，自己选的**，这才是有信息量的那个）: %d/%d = %.1f%%"
              % (agree, len(got), 100 * agree / max(1, len(got))))
        for cve, a, b in rowsq:
            flag = "✓" if a == b else "✗"
            print("      %-16s 模型选=%-20s 库里=%-20s %s" % (cve, a or "(空)", b, flag))

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
