# -*- coding: utf-8 -*-
"""核对失败样本的"分散"到底长什么样：把库里记的片段与文件里真实的代码并排打出来。

这个检查很重要：'每个词都在但不连续' 有两种完全不同的成因，处置也完全不同：

  · **文件的写法确实与 KB 记的不同**（版本/格式差异）→ 属数据一致性，放宽判据可能是在掩盖问题
  · **文件里逐字一模一样，只是断行/有注释插在中间** → 属词元匹配的硬性缺陷，放宽判据是对的

所以必须把两边原文都打出来看，不能只看一个'不连续'的结论。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT  # noqa: E402
from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"


def main() -> None:
    cve = sys.argv[1] if len(sys.argv) > 1 else "CVE-2018-20854"
    db = ROOT / "infrastructure/database/mas.db"

    con = sqlite3.connect(str(db))
    row = con.execute("select id, title, file_pattern, solution from issue_patterns where title=?",
                      (cve,)).fetchone()
    con.close()
    if not row:
        raise SystemExit("库里没有 %s" % cve)
    _id, title, fp, sol = row

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    frags = agent._extract_error_code_fragments(sol)

    d = DS / "before" / cve
    files = [f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in SOURCE_EXT]
    text = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
    toks = agent._tokenize_code(text)

    print("=" * 100)
    print("%s  条目 id=%s  文件=%s" % (title, _id, fp))
    print("=" * 100)
    print("\n【KB 的 solution（修复前后原文）】")
    print("  " + sol[:400].replace("\n", "\n  "))
    print("\n【切出来的错误代码片段】%d 个" % len(frags))
    for i, f in enumerate(frags, 1):
        s = " ".join(f)
        print("  片段%d（%d 个 token）: %s" % (i, len(f), s[:200]))
        print("      连续原样出现? %s" % ("是" if agent._is_contiguous_subseq(f, toks) else "**否**"))
        # 找出每个 token 第一次出现的位置，看是不是"都在但被打散"
        pos = []
        for t in f:
            i2 = next((k for k, x in enumerate(toks) if x == t), None)
            pos.append(i2)
        if all(p is not None for p in pos):
            print("      各 token 首次出现位置: %s  → 跨距 %d 个 token（若远大于片段长度，说明被别的词打散）"
                  % (pos, (max(pos) - min(pos)) if pos else 0))
        else:
            miss = [t for t, p in zip(f, pos) if p is None]
            print("      **有 token 在文件里压根找不到**: %s" % miss)

    # 把文件里含片段首 token 的行打出来
    head = frags[0][0] if frags else ""
    print("\n【样本文件里含 %r 的行】" % head)
    shown = 0
    for ln, line in enumerate(text.splitlines(), 1):
        if head in line:
            print("  L%-5d %s" % (ln, line.strip()[:120]))
            shown += 1
            if shown >= 6:
                break
    if not shown:
        print("  （一行都没有 → 说明该 token 在文件里根本不出现）")

    print("\n【结论要点】")
    all_present = all(next((k for k, x in enumerate(toks) if x == t), None) is not None
                      for f in frags for t in f)
    exact = any(agent._is_contiguous_subseq(f, toks) for f in frags)
    if exact:
        print("  片段连续原样存在 → 克隆判据本应命中（若线上没命中，问题在别处）")
    elif all_present:
        print("  片段的所有 token 都在文件里，但不连续 → **词元匹配的硬性缺陷**：")
        print("     判据要求『连续原样』，而实际代码被断行/注释/中间语句打散，于是判据失效；")
        print("     门控随后会据此判成『已修复』→ 误杀一次真实召回。")
    else:
        print("  片段里有 token 在文件中找不到 → **版本/数据不一致**，不是匹配算法的问题；")
        print("     这种情况放宽判据等于掩盖数据问题，应当回到入库侧加一致性检查。")


if __name__ == "__main__":
    main()
