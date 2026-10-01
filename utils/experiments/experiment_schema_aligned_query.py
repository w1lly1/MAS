# -*- coding: utf-8 -*-
"""实验：**规则对齐**到底值多少 —— 把查询文本换成与索引侧同构的模板，向量能不能对上？

## 用户提出的假设（本轮的核心问题）

> 首轮 LLM 不该只输出"一句英文说这段代码在干嘛"，而应输出**每个将送进知识库的分片在表达什么功能**；
> 而且这个输出必须与**知识库构建时的注入规则一致**，否则向量依旧很难近似。

这个假设是对的还是不对的，不该靠讨论，可以**直接测**。而且不需要 GPU：本项目用的编码器
（`distilbert-base-uncased`）在本地有缓存，知识库 800 个向量（200 条 × 4 层）也已经导出成
`weaviate_kb_dump.jsonl`（含 `layer_text` 与 `_vector`）。

## 索引侧的"规则"是什么（先把它读清楚，这是对照的基准）

见 `infrastructure/database/weaviate/service.py` 与 `utils/bigvul_ingest/rules.py`：

| 层 | 索引文本模板 |
|---|---|
| semantic | `[error_type] … [severity] … [language] … [framework] … [description] <CVE 英文摘要>` |
| code_pattern | `[problematic_pattern] <按 error_type 选的一句英文模式句> Evidence: <CVE 摘要> … [file_pattern] … [class_pattern] …` |
| solution | `[solution] Remove incorrect logic: <代码>. Ensure corrected path: <代码>` |
| full | 上面全部拼接 |

要点：**索引侧的散文不是 LLM 写的，而是"7 类 error_type → 固定英文模式句"的规则模板**，
配上 CVE 英文摘要。也就是说，两侧要能对上，查询侧必须说**同一套分类体系 + 同一种句式 + 同一种语言**。

## 本实验测什么

对冒烟那 8 个样本（各自条目都在库里、id 已知），用**同一套生产代码**（`_query_embed`，
含白化与 C2 偏移修正）编码 4 种查询文本，然后在 200 条知识里看**自己那条的排名**：

    V1_chunk       现状：原始代码分片按真实模板拼（≈线上 gap 通道的查询）
    V2_schema_gt   平凡天花板：直接用正确条目自己的 layer_text（rank=1 是必然的，仅用于标定）
    V3_prose_only  只给散文：`[description] <CVE 英文摘要>` —— 不含 error_type/文件名/类名
    V4_prose_para  只给**改写过的**散文：把摘要里的 CVE 号、版本号、文件名等"抄答案的线索"删掉，
                   再随机丢掉 40% 的词 —— 模拟"LLM 用自己的话描述同一件事"，
                   这一项才是对用户假设的**严格检验**（不抄原文还能不能对上）
    V5_code_only   可实现：只用代码本身能给出的字段（文件名/扩展名/原始代码），按索引侧模板拼

指标：自己那条的**排名**、**相似度**、是否进 **top-5**（= 线上的取数上限）、
以及**对无关条目的平均相似度**（越低说明越不"乱命中"）。

## 用法（本地，无需 GPU / 无需向量库）

    python utils/experiments/experiment_schema_aligned_query.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT  # noqa: E402
from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
DUMP = ROOT / "utils/experiments/weaviate_kb_dump.jsonl"
KB_SMOKE = ROOT / "utils/experiments/smoke_kb8.json"
LAYERS = ("semantic", "code_pattern", "solution", "full")


def load_dump() -> dict:
    """{layer: {sqlite_id: _vector}}"""
    out = {L: {} for L in LAYERS}
    with open(DUMP, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            L = str(r.get("vector_layer") or "")
            v = r.get("_vector")
            if L in out and isinstance(v, list) and v:
                out[L][int(r["sqlite_id"])] = [float(x) for x in v]
    return out


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def vuln_window(code: str, solution: str, width: int = 2000) -> str:
    """取"包含漏洞代码的那一段"作为分片 —— 这才是线上真正会送去检索的片段。

    为什么要特意做这件事：线上是按**分片**检索的（`source_code_chunk`），一个文件被切成很多段，
    其中**必然有一段包含漏洞代码**。若拿"文件开头 2000 字符"当查询，对多数文件来说那只是
    许可证声明和 include 列表 —— 与漏洞无关，会**把现状测得比实际差**（这个坑我在第一版就踩了）。
    """
    frag = re.search(r"Remove incorrect logic:\s*(.+?)(?:\.\s*Ensure corrected path:|$)",
                     solution or "", re.DOTALL)
    needle = ""
    if frag:
        needle = frag.group(1).split(";")[0].strip()[:60]
    lines = code.splitlines()
    if not lines:
        return ""
    idx = next((i for i, ln in enumerate(lines) if needle and needle[:40] in ln),
               len(lines) // 2)
    half = max(2, width // 80)
    lo = max(0, idx - half // 2)
    return "\n".join(lines[lo:lo + half])[:width]


def rank_of(agent, text: str, layer: str, own_id: int, index: dict, trunc: int):
    """用生产代码编码查询，返回 (own 排名, own 相似度, top1 相似度, 无关条目平均相似度)"""
    q = agent._query_embed(text, layer)
    if trunc:
        q = q[:trunc] + [0.0] * (len(q) - trunc)
    sims = []
    own = None
    for sid, v in index.items():
        s = dot(q, v[: len(q)] if len(v) > len(q) else v + [0.0] * (len(q) - len(v)))
        if sid == own_id:
            own = s
        else:
            sims.append(s)
    sims.sort(reverse=True)
    if own is None:
        return None, None, (sims[0] if sims else 0.0), (sum(sims) / len(sims) if sims else 0.0)
    rank = 1 + sum(1 for s in sims if s > own)
    return rank, own, (sims[0] if sims else 0.0), (sum(sims) / len(sims) if sims else 0.0)


def paraphrase(summary: str, rec: dict, seed: int = 20240930) -> str:
    """把摘要改写成"另一个人用自己的话描述同一件事"。

    刻意删掉三类**抄答案的线索**，否则测出来的是"能不能背出原文"而不是"语义是否对得上"：
      · CVE 编号、版本号（`before 4.20`、`CVE-2018-1234`）
      · 文件名与路径（`drivers/phy/mscc/...c`）
      · 再随机丢掉 40% 的词（打乱措辞）
    """
    t = str(summary or "")
    t = re.sub(r"CVE-\d{4}-\d+", " ", t, flags=re.I)
    t = re.sub(r"\b(before|after|through|prior to)\s+[\d.]+", " ", t, flags=re.I)
    for token in (rec.get("file_pattern") or "", Path(rec.get("file_pattern") or "").name):
        if token:
            t = t.replace(token, " ")
    words = t.split()
    rng = random.Random(seed)
    kept = [w for w in words if rng.random() > 0.40]
    return " ".join(kept) if len(kept) >= 5 else " ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--batch", type=Path, default=KB_SMOKE)
    ap.add_argument("--dump", type=Path, default=DUMP)
    ap.add_argument("--trunc", type=int, default=0,
                    help="把查询向量截断到前 k 维（0=不截断，按整条算）")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/schema_aligned_query.json")
    args = ap.parse_args()

    if not args.dump.exists():
        raise SystemExit("缺少向量导出: %s" % args.dump)
    index = load_dump()
    if not index["semantic"]:
        raise SystemExit("向量导出里没有数据")

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    con = sqlite3.connect(str(args.db))
    kb = {int(i): {"cve": (t or "").strip().upper(), "error_type": et or "",
                   "error_description": ed or "", "problematic_pattern": pp or "",
                   "file_pattern": fp or "", "class_pattern": cp or "",
                   "language": lang or "", "severity": sev or "", "framework": fw or ""}
          for i, t, et, ed, pp, fp, cp, lang, sev, fw in con.execute(
              "select id, title, error_type, error_description, problematic_pattern, "
              "file_pattern, class_pattern, language, severity, framework from issue_patterns")}
    con.close()
    by_cve = {v["cve"]: (k, v) for k, v in kb.items()}

    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    items = batch.get("rows") or batch.get("items") or []

    print("=" * 108)
    print("规则对齐实验：查询文本换成『与索引侧同构』的模板后，自己那条能不能排上来")
    print("=" * 108)
    print("  库=%s   向量导出=%s   样本=%d" % (args.db.name, args.dump.name, len(items)))

    results = {}
    for it in items:
        cve = str(it.get("cve") or "").strip().upper()
        hit = by_cve.get(cve)
        if not hit:
            print("  %-16s 库里没有该条目，跳过" % cve)
            continue
        own_id, rec = hit
        d = DS / "before" / cve
        code = ""
        if d.exists():
            code = "\n".join(f.read_text(encoding="utf-8", errors="ignore")
                             for f in sorted(d.rglob("*"))
                             if f.is_file() and f.suffix.lower() in SOURCE_EXT)
        snippet = code[:2000]
        vuln = vuln_window(code, rec.get("solution", "")) or code[len(code) // 3:][:2000]

        def chunk_query(text: str) -> str:
            iss = {"source": "source_code_chunk",
                   "description": "source_code_chunk L1-%d: %s" % (text.count("\n") + 1, text),
                   "code_snippet": text, "file": rec["file_pattern"], "severity": "medium"}
            return agent._build_query_text(iss, rec["file_pattern"])

        # V1a 现状（文件开头）—— 仅作对照，代表"内容与漏洞无关的分片"
        V1a = chunk_query(snippet)
        # V1b 现状（**漏洞所在窗口**）—— 这才是线上真正会送进检索的那种分片
        V1b = chunk_query(vuln)

        # V2 平凡天花板：直接用正确条目自己的 layer_text（rank=1 必然，仅标定用）
        V2 = "\n".join([
            "[error_type] %s" % rec["error_type"],
            "[language] %s" % rec["language"],
            "[framework] %s" % rec["framework"],
            "[description] %s" % rec["error_description"],
            "[pattern] %s" % rec["problematic_pattern"],
            "[file_pattern] %s" % rec["file_pattern"],
            "[class_pattern] %s" % rec["class_pattern"],
        ])

        # V3 只给散文（不含 error_type / 文件名 / 类名 这些额外锚点）
        V3 = "[description] %s" % rec["error_description"]

        # V4 只给**改写过的**散文 —— 对用户假设的严格检验
        V4 = "[description] %s" % paraphrase(rec["error_description"], rec)

        # V5 可实现：只用代码本身能给出的字段，按索引侧模板拼
        V5 = "\n".join([
            "[error_type] general",
            "[language] %s" % (rec["language"] or "C"),
            "[file_pattern] %s" % rec["file_pattern"],
            "[class_pattern] ",
            "[code] %s" % snippet,
        ])

        variants = {"V1a_文件开头(对照)": V1a, "V1b_漏洞窗口(现状)": V1b, "V2_self_text(平凡天花板)": V2,
                    "V3_prose_only(只给散文)": V3, "V4_prose_para(改写散文)": V4,
                    "V5_code_only(可实现)": V5}
        row = {}
        print("\n  %s  (自己条目 id=%d, 文件=%s)" % (cve, own_id, rec["file_pattern"]))
        print("    %-26s %s" % ("变体", "  ".join("%-18s" % L for L in LAYERS)))
        for name, text in variants.items():
            line, stat = [], {}
            for L in LAYERS:
                r, own, top1, avg = rank_of(agent, text, L, own_id, index[L], args.trunc)
                stat[L] = {"rank": r, "own": own, "top1": top1, "avg_other": avg}
                line.append("%-18s" % ("rank=%-3s sim=%.3f" % (r if r else "-", own if own else 0.0)))
            row[name] = stat
            print("    %-26s %s" % (name, "  ".join(line)))
        # top-5 命中率（线上每层只取 5 条）—— 先算完再写回，避免边遍历边改字典
        top5 = {n: sum(1 for L in LAYERS if row[n][L]["rank"] and row[n][L]["rank"] <= 5)
                for n in variants}
        row["_top5"] = top5
        results[cve] = row
        print("    → 进 top-5 的层数: " + "  ".join("%s=%d/4" % (n, top5[n]) for n in variants))

    # 汇总
    print("\n" + "=" * 108)
    print("汇总（%d 个样本）" % len(results))
    print("=" * 108)
    names = ["V1a_文件开头(对照)", "V1b_漏洞窗口(现状)", "V2_self_text(平凡天花板)", "V3_prose_only(只给散文)",
             "V4_prose_para(改写散文)", "V5_code_only(可实现)"]
    print("  %-26s %-14s %-14s %-16s" % ("变体", "进top-5层数", "平均排名", "无关条目平均相似度"))
    for n in names:
        t5 = sum(results[c]["_top5"][n] for c in results)
        ranks, avgs = [], []
        for c in results:
            for L in LAYERS:
                s = results[c][n][L]
                if s["rank"]:
                    ranks.append(s["rank"])
                avgs.append(s["avg_other"])
        print("  %-26s %-14s %-14s %-16s" % (
            n, "%d/%d" % (t5, len(results) * 4),
            "%.1f" % (sum(ranks) / len(ranks)) if ranks else "-",
            "%.4f" % (sum(avgs) / len(avgs)) if avgs else "-"))
    print("\n  怎么读：V1b 是**公平的现状**（用含漏洞代码的那一段当查询，线上必然有这么一段）；")
    print("          V3/V4 是**同语言同语域的散文**：如果它们远好于 V1b，说明瓶颈在**查询侧的规则**")
    print("          （语域 / 语言 / 分类体系），而不是检索本身。")
    print("          V4 还把 CVE 号、版本号、文件名删掉并随机丢掉 40% 的词 —— 用来证明")
    print("          『不抄原文也能对上』，这是对『LLM 用自己的话描述』的严格检验。")
    print("          V5 说明：只把代码套上索引侧的方括号标签**没用**，必须是真语义。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写出: %s" % args.out)


if __name__ == "__main__":
    main()
