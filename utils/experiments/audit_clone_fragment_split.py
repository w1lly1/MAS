# -*- coding: utf-8 -*-
"""查清"错误代码片段"的**切分**对不对 —— 很可能是它在制造"不可能命中的长针"。

## 起因

冒烟里唯一失分的样本，KB 的 `solution` 长这样：

    Remove incorrect logic: for (i = 0; i <= SERDES_MAX; i++) {; for (i = 0; i <= SERDES_MAX; i++) {.
    Ensure corrected path: for (i = 0; i < SERDES_MAX; i++) {; ...

注意：被删掉的两行**是同一行出现两次**，中间用 **一个分号**（`{; for`）连起来。

而现有的切分逻辑是按 **两个分号 `;;`** 拆的：

    re.split(r";;", payload)

于是这里**拆不开**，两行被**粘成一条 18 个 token 的针**：

    for i = 0 i <= SERDES_MAX i ++  for i = 0 i <= SERDES_MAX i ++

这条针在文件里**永远不可能连续出现**（两处代码在文件里隔着 500 多个 token）→
克隆判据必然失败 → 门控据此判"已修复" → **误杀一次真实召回**。

**所以问题不是"词元匹配不行"，而是"切分把两行粘成了一行"。**
单行版本（10 个 token）应当能命中 —— 本脚本就验证这一点，并统计这个毛病在整库里有多普遍。

## 用法

    python utils/experiments/audit_clone_fragment_split.py           # 全库统计
    python utils/experiments/audit_clone_fragment_split.py --case CVE-2018-20854   # 单例细看
"""
from __future__ import annotations

import argparse
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
PAYLOAD_RE = re.compile(r"Remove incorrect logic:\s*(.+?)(?:\.\s*Ensure corrected path:|$)", re.DOTALL)


def payload_of(solution: str) -> str:
    m = PAYLOAD_RE.search(solution or "")
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--case", default=None, help="单独细看某个 CVE")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    con = sqlite3.connect(str(args.db))
    rows = [(i, (t or "").strip().upper(), fp or "", sol or "") for i, t, fp, sol in
            con.execute("select id, title, file_pattern, solution from issue_patterns")]
    con.close()

    def sample_tokens(cve: str):
        d = DS / "before" / cve
        if not d.exists():
            return None
        txt = "\n".join(f.read_text(encoding="utf-8", errors="ignore")
                        for f in d.rglob("*")
                        if f.is_file() and f.suffix.lower() in SOURCE_EXT)
        return agent._tokenize_code(txt) if txt else None

    if args.case:
        cve = args.case
        rec = next((r for r in rows if r[1] == cve), None)
        if not rec:
            raise SystemExit("库里没有 %s" % cve)
        _id, title, fp, sol = rec
        pay = payload_of(sol)
        toks = sample_tokens(title)
        print("=" * 100)
        print("%s  条目 id=%s" % (title, _id))
        print("=" * 100)
        print("\n【Remove incorrect logic 的原文 payload】\n  %s" % pay[:300].replace("\n", "\n  "))
        print("\n【按 `;;` 切（现有实现）】")
        old = re.split(r";;", pay)
        for i, p in enumerate(old, 1):
            t = agent._tokenize_code(p)
            print("  片段%d: %d token  %s" % (i, len(t), " ".join(t)[:120]))
            print("        连续命中? %s" % ("是" if toks and agent._is_contiguous_subseq(t, toks) else "**否**"))
        # 更细的切法：单行 / 单语句边界（分号 + 换行，或 `; ` 后面跟 for/if/while/return 等）
        print("\n【按『语句边界』细切（候选修法）】")
        better = [x for x in re.split(r";\s*(?=(?:for|if|while|return|switch|int|char|void|struct)\b)", pay) if x.strip()]
        for i, p in enumerate(better, 1):
            t = agent._tokenize_code(p)
            if len(t) < 4:
                continue
            print("  片段%d: %d token  %s" % (i, len(t), " ".join(t)[:120]))
            print("        连续命中? %s" % ("是" if toks and agent._is_contiguous_subseq(t, toks) else "否"))
        return

    # ---------------- 全库统计 ----------------
    print("=" * 100)
    print("全库统计：错误代码片段的切分质量（%d 条）" % len(rows))
    print("=" * 100)
    n_payload = 0
    n_has_double = 0        # payload 里含 `;;`（现有切分能拆开）
    n_only_single = 0       # payload 里没有 `;;` 但含 `; `（现有切分拆不开 → 有粘针风险）
    n_multi_glued = 0       # 现有实现切出的片段里，存在"同一 token 序列重复两次以上"的粘针
    n_better_more = 0       # 用细切法能切出更多片段（说明现有实现确实少切了）
    glue_examples = []
    for _id, cve, fp, sol in rows:
        pay = payload_of(sol)
        if not pay:
            continue
        n_payload += 1
        if ";;" in pay:
            n_has_double += 1
        elif re.search(r";\s", pay):
            n_only_single += 1
        old_frags = agent._extract_error_code_fragments(sol)
        glued = False
        for f in old_frags:
            half = len(f) // 2
            if half >= 4 and f[:half] == f[half:2 * half]:
                glued = True
        if glued:
            n_multi_glued += 1
            if len(glue_examples) < 6:
                glue_examples.append((cve, " ".join(old_frags[0])[:90]))
        better = [x for x in re.split(r";\s*(?=(?:for|if|while|return|switch|int|char|void|struct)\b)", pay)
                  if x.strip()]
        n_old = len(old_frags)
        n_new = len([1 for p in better if len(agent._tokenize_code(p)) >= 4])
        if n_new > n_old:
            n_better_more += 1

    print("\n  有『Remove incorrect logic』payload 的条目        : %d" % n_payload)
    print("  payload 里含 `;;`（现有切分能拆开）              : %d" % n_has_double)
    print("  payload 里**没有** `;;` 但有 `; `（拆不开）      : %d" % n_only_single)
    print("  现有实现切出了『自我重复的粘针』（同一序列×2）    : **%d**" % n_multi_glued)
    print("  用语句边界细切能切出**更多**片段的条目           : %d" % n_better_more)
    if glue_examples:
        print("\n  粘针样例（前 6 条）:")
        for cve, s in glue_examples:
            print("    %-16s %s" % (cve, s))
    print("\n  怎么读：'粘针'的片段在文件里**不可能**连续命中（两处代码在文件里挨不着），")
    print("          于是克隆判据必然失败 → 门控据此判『已修复』→ 误杀召回。")
    print("          这是**切分**的问题，不是匹配算法的问题，也不是数据版本的问题。")


if __name__ == "__main__":
    main()
