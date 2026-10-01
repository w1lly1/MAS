# -*- coding: utf-8 -*-
"""直接验证"粘针修复"到底有没有生效：重建后的 needle 能不能命中**被分析的那个文件**。

## 为什么要专门验这一条

实测：新系统臂里 `CVE-2018-20854` 仍被判 `code_already_fixed`（= "KB 记的修复前错误代码
在文件里找不到了" → 认定已修复 → 丢弃候选）。而离线预筛说重建后的两条 9-token 针**都能命中**。
两者矛盾 → 必须直接量，不能靠推理。

本脚本用**生产实现**提取针、用**被分析目录里的真实源码**当 haystack：
  * 老 needle（粘连的那条）命中吗？
  * 新 needle（重建后 `;;` 分开的两条）命中吗？
"""
from __future__ import annotations

import argparse
import json
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
PAYLOAD_RE = re.compile(r"Remove incorrect logic:\s*(.+?)(\.\s*Ensure corrected path:|$)", re.DOTALL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cve", default="CVE-2018-20854")
    ap.add_argument("--old-db", type=Path, default=ROOT / "reports/mas_live.db")
    ap.add_argument("--new-db", type=Path, default=ROOT / "reports/mas_rebuild_candidate.db")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent

    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    d = DS / "before" / args.cve
    files = [f for f in sorted(d.rglob("*")) if f.is_file() and f.suffix.lower() in SOURCE_EXT]
    print("被分析目录 %s：%d 个源文件" % (d.name, len(files)))
    hay_by_file = {}
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="ignore")
        hay_by_file[f.name] = agent._tokenize_code(txt)
    all_tokens = agent._tokenize_code("\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files))

    def report(label, db):
        con = sqlite3.connect("file:%s?mode=ro" % db.as_posix(), uri=True)
        row = list(con.execute("select id, title, solution, file_pattern from issue_patterns where upper(title)=?",
                               (args.cve.upper(),)))
        con.close()
        if not row:
            print("  %s：库里没有 %s" % (label, args.cve))
            return
        _id, _t, sol, fp = row[0]
        print("\n" + "=" * 92)
        print("%s（条目 id=%s，file_pattern=%r）" % (label, _id, fp))
        payload = PAYLOAD_RE.search(sol or "")
        payload = payload.group(1) if payload else ""
        # 用**生产实现**抽针
        frags = agent._extract_error_code_fragments(sol or "")
        print("  生产实现抽出 %d 条针：" % len(frags))
        for i, toks in enumerate(frags, 1):
            hit_any = agent._is_contiguous_subseq(toks, all_tokens)
            hits = [n for n, tk in hay_by_file.items() if agent._is_contiguous_subseq(toks, tk)]
            print("   针%d：%2d token  %s" % (i, len(toks), " ".join(toks)[:88]))
            print("        命中整个目录？%s   命中的文件：%s" % ("是" if hit_any else "**否**", hits or "（无）"))
        # 也打印原始 payload 里被 `;` 分隔的片段数（诊断切分）
        if payload:
            print("  原始 payload 按单分号切成 %d 段；按 `;;` 切成 %d 段"
                  % (len([x for x in payload.split(";") if x.strip()]),
                     len([x for x in payload.split(";;") if x.strip()])))

    report("【基线库（重建前，粘针）】", args.old_db)
    report("【重建后的候选库】", args.new_db)


if __name__ == "__main__":
    main()
