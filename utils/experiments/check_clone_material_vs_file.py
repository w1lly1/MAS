# -*- coding: utf-8 -*-
"""核对：知识库里记的"修复前错误代码"，在**样本文件**里到底找不找得到？

## 为什么查这个（冒烟里挖出来的具体线索）

冒烟 8 个样本召回 7 个，唯一失败的那个（CVE-2018-20854）被门控判成 `code_already_fixed`
（同一个文件、错误代码没找到 → 判定"已修复"）。它**不是**判据没生效，恰恰相反：
判据生效了，但前提"KB 里记的修复前代码确实存在于这个文件"不成立。

于是问题变成：**这种事有多普遍？** 如果相当一部分条目的"修复前代码"在它自己那个文件里都找不到，
那 `code_already_fixed` 会持续误杀真实召回，而这**不是门控能修的**，得回到知识库构建那一侧。

## 判据分三档（从严到松）

    exact       KB 里的片段 token 序列**连续原样**出现在文件里 → 克隆判据成立（最好）
    scattered   片段的 token **都**在文件里，但**不连续** → 多半是空白/格式/预处理差异
    missing     部分 token 压根不在文件里 → 多半是**版本不一致**（KB 的代码与样本文件不是同一版）

## 用法

    python utils/experiments/check_clone_material_vs_file.py --batch utils/experiments/smoke_kb8.json \
        --db infrastructure/database/mas.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT  # noqa: E402
from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"


def sample_text(cve: str) -> str:
    d = DS / "before" / cve
    if not d.exists():
        return ""
    buf = []
    for f in sorted(d.rglob("*")):
        try:
            if f.is_file() and f.suffix.lower() in SOURCE_EXT:
                buf.append(f.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=ROOT / "utils/experiments/smoke_kb8.json")
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    con = sqlite3.connect(str(args.db))
    rows = {i: {"cve": (t or "").strip().upper(), "solution": sol or ""}
            for i, t, sol in con.execute("select id, title, solution from issue_patterns")}
    con.close()
    by_cve = {v["cve"]: (k, v) for k, v in rows.items()}

    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    items = batch.get("rows") or batch.get("items") or []
    cves = [str(i.get("cve") or "").strip().upper() for i in items]

    print("=" * 104)
    print("KB 记的『修复前代码』 vs 样本文件（库=%s）" % args.db)
    print("=" * 104)
    print("  %-16s %-6s %-9s %-10s %-9s %s" % (
        "CVE", "片段数", "exact?", "缺哪些词", "判定", "说明"))

    stat = {"exact": 0, "scattered": 0, "missing": 0, "nofrag": 0, "nofile": 0}
    for cve in cves:
        hit = by_cve.get(cve)
        if not hit:
            continue
        _id, rec = hit
        text = sample_text(cve)
        if not text:
            stat["nofile"] += 1
            print("  %-16s %-6s %s" % (cve, "-", "读不到样本源码，跳过"))
            continue
        toks = agent._tokenize_code(text)
        tokset = set(toks)
        frags = agent._extract_error_code_fragments(rec["solution"])
        if not frags:
            stat["nofrag"] += 1
            print("  %-16s %-6d %-9s %-10s %-9s %s" % (
                cve, 0, "-", "-", "无原料", "条目自身没有可提取的错误代码片段"))
            continue
        exact = any(agent._is_contiguous_subseq(f, toks) for f in frags)
        missing_tokens = set()
        for f in frags:
            for t in f:
                if t.lower() not in {x.lower() for x in tokset}:
                    missing_tokens.add(t)
        if exact:
            stat["exact"] += 1
            verdict, note = "exact", "克隆判据成立"
        elif not missing_tokens:
            stat["scattered"] += 1
            verdict, note = "scattered", "词都在但不连续（格式/空白差异）"
        else:
            stat["missing"] += 1
            verdict, note = "missing", "**版本不一致**：KB 记的代码不在这个文件里"
        print("  %-16s %-6d %-9s %-10s %-9s %s" % (
            cve, len(frags), "是" if exact else "否",
            (str(len(missing_tokens)) + " 个") if missing_tokens else "0",
            verdict, note))

    n = sum(stat.values())
    print("\n  汇总（%d 个样本）：" % n)
    print("    exact   （KB 的修复前代码原样在文件里）: %d" % stat["exact"])
    print("    scattered（词都在、但不连续）          : %d" % stat["scattered"])
    print("    missing （**版本不一致**，词都缺）      : %d" % stat["missing"])
    print("    无原料 / 读不到源码                    : %d / %d" % (stat["nofrag"], stat["nofile"]))
    print("\n  怎么用：'missing' 的条目，克隆判据**不可能**命中，而门控还会据此判『已修复』→ 误杀召回。")
    print("          这类条目要在**知识库构建/入库**时用一个一致性检查挡掉（对比片段是否出现在源文件里）。")


if __name__ == "__main__":
    main()
