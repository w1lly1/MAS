# -*- coding: utf-8 -*-
"""硬负样本（已修复代码）误报实验。

用 BigVul 的 after 目录（修复后代码）作为"已修复硬负样本"：
  对每个 kb CVE，取它的 after 版本代码（缺陷已修复），检查系统是否会"错报"。
  在已修复代码上，历史缺陷条目的"错误代码片段"理论上应已消失（被修复），
  若系统仍报出该缺陷，即为硬负样本误报。

口径：以词法-结构通道（错误代码连续子串）为判定——错误代码片段是否仍在 after 代码中出现。
  若仍出现 → 说明该 CVE 的"修复"未真正移除错误代码（或 before/after 差异很小），
            这本身是数据特性，不计为系统误报，但需如实报告。
  若不再出现 → 系统正确不再报该缺陷（词法通道正确）。

本实验回答审稿人"7.0% 错配不能外推到已修复代码"的关切，给出：
  已修复代码上，历史缺陷条目错误代码的"残留率"（即若原样重现判定，会有多少误报）。

运行（MAS 根目录）：
    venv/Scripts/python.exe utils/experiments/hard_negative.py
产出：reports/hard_negative.json
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent
DB = ROOT / "infrastructure/database/mas.db"
BATCH = ROOT / "utils/experiments/test_400_error_batch.json"

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
AFTER = DS / "after"

CODE_EXTS = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".s", ".S"}
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


def read_after_code(cve):
    d = AFTER / cve
    if not d.is_dir():
        return None
    texts = []
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.suffix.lower() in CODE_EXTS:
            try:
                texts.append(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    return "\n".join(texts)


def main():
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    kb = [it["output_dir"] for it in batch["items"] if it["role"] == "kb"]

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT title, solution FROM issue_patterns")
    sol_by_title = {t: s for t, s in cur.fetchall()}
    con.close()

    residue = []       # 错误代码仍残留在 after 代码中的样本
    no_residue = []    # 错误代码已消失的样本
    no_after = []      # 无 after 代码
    no_frag = []       # 无法提取错误代码片段

    for cve in kb:
        sol = sol_by_title.get(cve)
        frags = extract_error_fragments(sol)
        if not frags:
            no_frag.append(cve)
            continue
        code = read_after_code(cve)
        if code is None:
            no_after.append(cve)
            continue
        code_toks = tokenize(code)
        if any(is_contiguous_subseq(f, code_toks) for f in frags):
            residue.append(cve)
        else:
            no_residue.append(cve)

    n = len(kb)
    n_res = len(residue)
    n_nores = len(no_residue)
    print("=" * 70)
    print("硬负样本（已修复 after 代码）误报实验")
    print("=" * 70)
    print(f"kb 样本 {n}")
    print(f"  错误代码仍残留在 after 代码中（若原样重现判定会误报）: {n_res} ({n_res/n:.1%})")
    print(f"  错误代码已消失（词法通道正确不再报）: {n_nores} ({n_nores/n:.1%})")
    print(f"  无 after 代码: {len(no_after)}")
    print(f"  无法提取错误代码片段: {len(no_frag)}")

    out = ROOT / "reports/hard_negative.json"
    out.write_text(json.dumps({
        "n": n, "residue": residue, "no_residue": no_residue,
        "no_after": no_after, "no_frag": no_frag,
        "residue_rate": round(n_res / n, 4),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}")


if __name__ == "__main__":
    main()
