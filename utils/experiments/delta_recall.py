# -*- coding: utf-8 -*-
"""ΔRecall 实验：二次校验相对首轮大模型审查的独立净增。

判定标准（与词法通道一致，纯离线可算）：
  首轮 LLM"已发现"该 CVE 历史缺陷 ⟺ 首轮 consolidated 报告（pureLLM/consolidated）
    的任意一条 issue 的 code_snippet（或 description）中，出现了该 CVE 历史缺陷条目
    solution 的"Remove incorrect logic"段所载错误代码的连续词元子串。

  首轮召回 = kb 组 200 样本中被首轮"已发现"的比例
  二次召回 = kb 组 200 样本中被二次校验命中（new_findings 含 self）的比例 = 144
  ΔRecall = 二次校验命中、且首轮"漏检"的样本数（即二次校验的独立净增）

运行（MAS 根目录）：
    venv/Scripts/python.exe utils/experiments/delta_recall.py
产出：reports/delta_recall.json + 控制台
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports/analysis"
DB = ROOT / "infrastructure/database/mas.db"
BATCH = ROOT / "utils/experiments/test_400_error_batch.json"

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|[+\-*/%=<>!&|^~]")


def tokenize(text):
    return TOKEN_RE.findall(text or "")


def is_contiguous_subseq(needle, haystack):
    n = len(needle)
    if n == 0:
        return False
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return True
    return False


def extract_error_fragments(solution):
    """从 solution 提取错误代码片段 token 序列（复刻 agent 逻辑）。"""
    if not solution:
        return []
    m = re.search(r"Remove incorrect logic:\s*(.+?)(?:\.\s*Ensure corrected path:|$)",
                  solution, re.DOTALL)
    if not m:
        return []
    frags = []
    seen = set()
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


def latest_run(cve):
    d = REPORTS / cve
    runs = [x for x in d.iterdir() if x.is_dir()] if d.exists() else []
    return max(runs, key=lambda x: x.stat().st_mtime) if runs else None


def first_pass_issues(cve):
    """读首轮 pureLLM consolidated 的所有 issue 文本。"""
    run = latest_run(cve)
    if run is None:
        return []
    texts = []
    cd = run / "pureLLM/consolidated"
    if cd.exists():
        for f in cd.glob("*.json"):
            try:
                j = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            for iss in j.get("issues", []):
                snippet = str(iss.get("code_snippet") or "")
                desc = str(iss.get("description") or "")
                texts.append(snippet + "\n" + desc)
    return texts


def second_pass_hit(cve):
    """二次校验是否命中自身（new_findings 含 self）。"""
    run = latest_run(cve)
    if run is None:
        return False
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id FROM issue_patterns WHERE title=?", (cve,))
    own = cur.fetchone()
    cur.execute("SELECT id, pattern_id FROM curated_issues")
    ci_to_p = {i: p for i, p in cur.fetchall()}
    con.close()
    if own is None:
        return False
    own = own[0]
    cd = run / "second_pass/consolidated"
    if not cd.exists():
        return False
    for f in cd.glob("*.json"):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for nf in j.get("new_findings", []):
            ev = nf.get("evidence") or {}
            sid = ev.get("sqlite_id")
            ch = ev.get("channel")
            if ch == "curated_issue":
                if ci_to_p.get(sid) == own:
                    return True
            elif sid == own:
                return True
    return False


def main():
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    kb = [it["output_dir"] for it in batch["items"] if it["role"] == "kb"]

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT title, solution FROM issue_patterns")
    sol_by_title = {t: s for t, s in cur.fetchall()}
    con.close()

    first_hit = set()    # 首轮已发现
    second_hit = set()   # 二次校验命中
    both = set()         # 两者都命中
    detail = {}

    for cve in kb:
        # 首轮判定
        frags = extract_error_fragments(sol_by_title.get(cve))
        fp_issues = first_pass_issues(cve)
        fp_hit = False
        if frags:
            fp_toks = [tokenize(t) for t in fp_issues]
            for frag in frags:
                if any(is_contiguous_subseq(frag, toks) for toks in fp_toks):
                    fp_hit = True
                    break
        sp_hit = second_pass_hit(cve)
        if fp_hit:
            first_hit.add(cve)
        if sp_hit:
            second_hit.add(cve)
        if fp_hit and sp_hit:
            both.add(cve)
        detail[cve] = {"first_pass": fp_hit, "second_pass": sp_hit}

    n = len(kb)
    first_recall = len(first_hit) / n
    second_recall = len(second_hit) / n
    delta = len(second_hit - first_hit)  # 二次校验补回的、首轮漏检的
    delta_rate = delta / n

    print("=" * 70)
    print("ΔRecall：二次校验相对首轮大模型审查的独立净增")
    print("=" * 70)
    print(f"kb 组样本数: {n}")
    print(f"首轮 LLM 召回: {len(first_hit)}/{n} = {first_recall:.1%}")
    print(f"二次校验召回: {len(second_hit)}/{n} = {second_recall:.1%}")
    print(f"两者都命中: {len(both)}")
    print(f"ΔRecall（二次校验独立净增 = 二次命中且首轮漏检）: {delta}/{n} = {delta_rate:.1%}")
    print(f"二次校验命中但首轮已发现的（重叠）: {len(second_hit & first_hit)}")

    out = ROOT / "reports/delta_recall.json"
    out.write_text(json.dumps({
        "n": n, "first_pass_recall": len(first_hit), "second_pass_recall": len(second_hit),
        "both": len(both), "delta_recall": delta,
        "delta_rate": round(delta_rate, 4), "detail": detail,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写: {out}")


if __name__ == "__main__":
    main()
