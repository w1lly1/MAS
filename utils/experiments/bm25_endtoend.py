# -*- coding: utf-8 -*-
"""BM25 端到端替换实验（离线，无需 Weaviate）。

目标：回答"若用 BM25 替换语义通道（Weaviate 向量检索），端到端召回/错配如何变化"。

方法（离线复刻完整门控）：
  对每个 kb/held 样本，取二次阶段实际使用的代码分片作为查询，
  用 BM25 对 200 条 KB 条目检索 top-k，把 BM25 命中的条目作为"语义通道候选"，
  再套用两通道析取门控 admit = F ∧ [s≥θ_s ∨ (v≥τ ∧ a≥θ_a ∧ s≥θ_w)]。
  由于 BM25 不产生语义相似度 v(x)，本实验按两种口径报告：
    (a) BM25 候选仅走词法-结构通道（s≥θ_s，即 BM25 命中的条目若错误代码子串命中则召回）
    (b) BM25 候选 + 同文件名锚点即算语义通道达标（近似：BM25 命中即视为语义线索）

口径 (a) 更严格、可辩护：BM25 是字面匹配，替换的是"语义检索"角色，其命中条目
最终仍需通过词法-结构通道（错误代码连续子串）才算召回。

运行（MAS 根目录）：
    venv/Scripts/python.exe utils/experiments/bm25_endtoend.py
产出：reports/bm25_endtoend.json
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent
DB = ROOT / "infrastructure/database/mas.db"
BATCH = ROOT / "utils/experiments/test_400_error_batch.json"
REPORTS = ROOT / "reports/analysis"

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|[+\-*/%=<>!&|^~]")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs; self.k1 = k1; self.b = b
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def topk(self, query, k=5):
        scores = []
        for i in range(self.N):
            dl = self.doc_len[i]; tf = self.tf[i]; sc = 0.0
            for t in query:
                if t not in tf:
                    continue
                f = tf[t]
                sc += self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9)))
            scores.append((sc, i))
        scores.sort(reverse=True)
        return scores[:k]


def latest_run(cve):
    d = REPORTS / cve
    runs = [x for x in d.iterdir() if x.is_dir()] if d.exists() else []
    return max(runs, key=lambda x: x.stat().st_mtime) if runs else None


def code_query(cve):
    """复用二次阶段代码分片作为查询（与向量路径同输入）。"""
    run = latest_run(cve)
    if run is None:
        return ""
    texts = []
    cd = run / "second_pass/consolidated"
    if cd.exists():
        for f in cd.glob("*.json"):
            try:
                j = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            for ev in j.get("gap_retrieval_evidence") or []:
                t = (ev.get("code_chunk") or {}).get("text") or ""
                if t:
                    texts.append(t)
    return "\n".join(texts)[:12000]


def extract_error_fragments(solution):
    if not solution:
        return []
    m = re.search(r"Remove incorrect logic:\s*(.+?)(?:\.\s*Ensure corrected path:|$)", solution, re.DOTALL)
    if not m:
        return []
    frags, seen = [], set()
    for part in re.split(r";;", m.group(1)):
        p = part.strip()
        if p and p not in seen:
            seen.add(p)
            frags.append(p)
    out = []
    for f in frags:
        toks = tokenize(f)
        if len(toks) >= 4:
            out.append(toks)
    return out


def is_contiguous_subseq(needle, haystack):
    n = len(needle)
    if n == 0:
        return False
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return True
    return False


def main():
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    items = batch["items"]
    kb = [it["output_dir"] for it in items if it["role"] == "kb"]
    held = [it["output_dir"] for it in items if it["role"] == "held"]

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id, title, error_type, error_description, problematic_pattern, solution, file_pattern, class_pattern FROM issue_patterns")
    rows = cur.fetchall()
    con.close()

    entries = []
    for r in rows:
        sol = r[5] or ""
        entries.append({
            "id": r[0], "title": (r[1] or "").strip(),
            "text": " ".join([r[1] or "", r[2] or "", r[3] or "", r[4] or "", sol, r[6] or "", r[7] or ""]),
            "solution": sol,
            "file_pattern": r[6] or "",
        })
    docs = [tokenize(e["text"]) for e in entries]
    bm25 = BM25(docs)
    title_to_entry = {e["title"]: e for e in entries}
    print(f"KB 条目 {len(entries)}，BM25 索引就绪")

    def eval_sample(cve, is_kb):
        """返回该样本 BM25 端到端是否命中自身（kb）/是否误报（held）。"""
        q = tokenize(code_query(cve))
        if not q:
            return False
        ranked = bm25.topk(q, k=5)
        hit_titles = [entries[i]["title"] for _, i in ranked]
        # 词法-结构通道判定：某条目的错误代码片段是否在当前代码里连续子串命中
        q_toks = tokenize(code_query(cve))
        def lex_hit(entry):
            frags = extract_error_fragments(entry["solution"])
            if not frags:
                return False
            return any(is_contiguous_subseq(f, q_toks) for f in frags)

        if is_kb:
            # 召回：BM25 top-k 命中的条目里，自身 CVE 的错误代码是否连续子串命中
            own = title_to_entry.get(cve)
            if own is None:
                return False
            # 自身条目需在 BM25 top-k 命中，且其错误代码子串命中当前代码
            return (cve in hit_titles) and lex_hit(own)
        else:
            # 误报：BM25 top-k 命中的【任一非自身条目】，其错误代码片段在当前
            # held 代码里连续子串命中，才算错配（答案不在库，任何"错误代码真重现"的命中都是错配）
            for t in hit_titles:
                e = title_to_entry.get(t)
                if e is None:
                    continue
                if lex_hit(e):
                    return True
            return False

    kb_hit = sum(1 for c in kb if eval_sample(c, True))
    held_fp = sum(1 for c in held if eval_sample(c, False))
    print(f"\nBM25 端到端（口径：BM25 命中 + 词法子串连续命中 = 召回）")
    print(f"kb 召回: {kb_hit}/{len(kb)} = {kb_hit/len(kb):.1%}")
    print(f"held 误报: {held_fp}/{len(held)} = {held_fp/len(held):.1%}")

    out = ROOT / "reports/bm25_endtoend.json"
    out.write_text(json.dumps({
        "kb_recall": kb_hit, "kb_total": len(kb), "kb_rate": round(kb_hit/len(kb), 4),
        "held_fp": held_fp, "held_total": len(held), "held_fp_rate": round(held_fp/len(held), 4),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}")


if __name__ == "__main__":
    main()
