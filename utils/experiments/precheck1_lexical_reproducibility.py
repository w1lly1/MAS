#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A2 预检 1：知识条目侧"词元可复现性"——**针在不在 before 里、修复码在不在 after 里**。

## 要回答什么

门控的主匹配键是"针"（`Remove incorrect logic` 里那段**错误代码**的连续 token 序列）。
它只有在"这段错误代码**真的出现在被分析文件里**"时才可能命中。于是有两个必须先量清的比率：

| 指标 | 含义 | 低了会怎样 |
|---|---|---|
| **针 ∈ before** | 该条目的针能在它**自己那个样本的漏洞版代码**里连续命中 | 命中不了 ⇒ 这条知识**结构上不可能**被词元通道回收（不是门控的错，是原料/切片的问题） |
| **针 ∈ after** | 针**在修复版里还在不在**（残留） | 残留 ⇒ "错误代码已从文件里消失"这类判据失去意义，克隆证据也可能是假阳 |
| **修复码 ∈ after** | `Ensure corrected path:` 那段能在修复版里命中 | 命中不了 ⇒ 知识条目的"修复侧"文本与真实修复对不上 |

同一套 token 口径全部复用生产代码（`_tokenize_code` / `_is_contiguous_subseq` /
`_extract_error_code_fragments`），**不重写判据**。

用法：
    python -X utf8 utils/experiments/precheck1_lexical_reproducibility.py
    python -X utf8 utils/experiments/precheck1_lexical_reproducibility.py --db reports/mas_rebuild_candidate_v2.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from utils.experiments.a5_semantic_path_backtest import basename_of, pick_file  # noqa: E402

FIX_MARK = "Ensure corrected path:"


def fix_code_tokens(agent, solution: str):
    """取 `Ensure corrected path:` 后面**那一段**修复码的 token（多段则全部返回）。

    口径细节（第一版就错在这里）：不能把标记之后的**整条尾巴**都 token 化 —— 尾巴里接着
    `;;` 分段和叙述文字，token 一混进去，连续子串匹配必然失败，于是"修复码命中率"被低估。
    做法：每处标记只取到下一个 `;;`（或下一个标记）为止。
    """
    sol = solution or ""
    out = []
    pos = 0
    while True:
        i = sol.find(FIX_MARK, pos)
        if i < 0:
            break
        seg = sol[i + len(FIX_MARK):]
        for cut in (";;", FIX_MARK):
            j = seg.find(cut)
            if j >= 0:
                seg = seg[:j]
        out.append(agent._tokenize_code(seg))
        pos = i + len(FIX_MARK)
    return [t for t in out if t]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_rebuild_candidate_v2.db")
    ap.add_argument("--show-miss", type=int, default=8, help="打印多少条失配样例")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import (
        AIDrivenSecondPassAnalysisAgent,
    )
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    con = sqlite3.connect(str(args.db))
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT id, title, file_pattern, solution FROM issue_patterns").fetchall()
    con.close()
    print("知识条目 %d 条（%s）" % (len(rows), args.db.name))

    stat = {"n": 0, "no_needle": 0, "no_file": 0, "no_after": 0, "no_fix_mark": 0,
            "needle_before": 0, "needle_after": 0, "fix_after": 0}
    lens, miss = [], []
    for r in rows:
        cve = str(r["title"] or "").strip()
        base = basename_of(str(r["file_pattern"] or ""))
        sol = str(r["solution"] or "")
        frags = agent._extract_error_code_fragments(sol)
        if not frags:
            stat["no_needle"] += 1
            continue
        f_before = pick_file(cve, base, "before")
        f_after = pick_file(cve, base, "after")
        if not f_before:
            stat["no_file"] += 1
            continue
        if not f_after:
            stat["no_after"] += 1
        stat["n"] += 1
        toks_before = agent._tokenize_code(f_before.read_text(encoding="utf-8", errors="ignore"))
        toks_after = (agent._tokenize_code(f_after.read_text(encoding="utf-8", errors="ignore"))
                      if f_after else [])
        hit_b = any(agent._is_contiguous_subseq(fr, toks_before) for fr in frags)
        hit_a = any(agent._is_contiguous_subseq(fr, toks_after) for fr in frags) if f_after else False
        fxt = fix_code_tokens(agent, sol)
        if not fxt:
            stat["no_fix_mark"] += 1
        hit_fix = bool(toks_after) and any(
            agent._is_contiguous_subseq(t, toks_after) for t in fxt)
        stat["needle_before"] += int(hit_b)
        stat["needle_after"] += int(hit_a)
        stat["fix_after"] += int(hit_fix)
        min_len = min(len(fr) for fr in frags)
        lens.append((min_len, hit_b))
        if not hit_b:
            miss.append((cve, base, min_len, len(toks_before)))

    n = stat["n"]
    print("\n%-42s %s" % ("条目（有针且能定位 before 文件）", n))
    print("%-42s %d (%.1f%%)" % ("**针 ∈ before**（词元通道可复现）",
                                 stat["needle_before"], 100.0 * stat["needle_before"] / max(1, n)))
    print("%-42s %d (%.1f%%)" % ("针 ∈ after（残留 = 隐患）",
                                 stat["needle_after"], 100.0 * stat["needle_after"] / max(1, n)))
    print("%-42s %d (%.1f%%)" % ("修复码 ∈ after",
                                 stat["fix_after"], 100.0 * stat["fix_after"] / max(1, n)))
    print("\n--- 未计入的条目 ---")
    print("%-42s %d" % ("抽不出针（片段过短/通用 token）", stat["no_needle"]))
    print("%-42s %d" % ("找不到 before 文件", stat["no_file"]))
    print("%-42s %d" % ("有针有 before 但**没有 after 文件**", stat["no_after"]))
    print("%-42s %d" % ("solution 里没有『%s』标记" % FIX_MARK, stat["no_fix_mark"]))

    print("\n--- 按针的最短长度分组（tokens → 针 ∈ before 命中率）---")
    for lo, hi in ((1, 3), (4, 5), (6, 9), (10, 999)):
        grp = [h for L, h in lens if lo <= L <= hi]
        if grp:
            print("  %2d~%-3d tokens : %3d 条，命中 %3d (%.0f%%)"
                  % (lo, hi if hi < 999 else 99, len(grp), sum(grp), 100.0 * sum(grp) / len(grp)))

    if miss:
        print("\n--- 针在 before 里命中不了的样例（前 %d 条）---" % min(args.show_miss, len(miss)))
        for cve, base, L, nb in miss[: args.show_miss]:
            print("  %-16s %-40s 最短针 %d tokens，before 共 %d tokens" % (cve, base, L, nb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
