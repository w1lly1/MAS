# -*- coding: utf-8 -*-
"""冒烟批次体检：这 8 个样本的**知识条目质量**如何（决定"最强那把尺子"能不能用）。

## 为什么查这个

冒烟里 8 个样本、召回 7 个。要判断"剩下那个为什么没召回""下一步该优化什么"，
先得看清**库那边**给了什么料：某个条目要是压根没有可提取的"修复前错误代码"片段，
那么权重最高的那条证据（错误代码克隆）对它**天然不可用** —— 这时再怎么调门控都没用，
得回到知识库构建那一侧。

三件事逐条查：

1. 条目自带的 `solution` 里有没有 `Remove incorrect logic: …` 错误代码片段（克隆判据的原料）；
2. 有没有 **curated 片段**（`curated_issues` 里挂在这个条目下的记录）——
   实测 curated 通道贡献了八成的候选，是召回的主力；
3. 条目里 `file_pattern` / `class_pattern` 是否齐全（门控的同文件锚点、类名锚点要用）。

## 用法

    python utils/experiments/audit_smoke_entries.py --batch utils/experiments/smoke_kb8.json \
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

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=ROOT / "utils/experiments/smoke_kb8.json")
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--all-kb", action="store_true", help="不只看批次，把整个库都体检一遍")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    con = sqlite3.connect(str(args.db))
    rows = {i: {"cve": (t or "").strip().upper(), "file_pattern": fp or "",
                "class_pattern": cp or "", "solution": sol or ""}
            for i, t, fp, cp, sol in con.execute(
                "select id, title, file_pattern, class_pattern, solution from issue_patterns")}
    cur_cnt = {}
    cur_sols = {}
    for pid, csol in con.execute("select pattern_id, solution from curated_issues"):
        cur_cnt[pid] = cur_cnt.get(pid, 0) + 1
        cur_sols.setdefault(pid, []).append(csol or "")
    con.close()

    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    items = batch.get("rows") or batch.get("items") or []
    targets = [] if args.all_kb else sorted(
        {(str(i.get("cve") or "").strip().upper()) for i in items})
    by_cve = {v["cve"]: (k, v) for k, v in rows.items()}

    def check(_id, rec):
        """克隆判据的原料在**两处**：条目自己的 solution，以及它挂着的 curated 片段。

        这一点很关键：`_match_curated_issue` 抽的是 **curated 那一行**的 solution，
        不是条目的 solution。所以即使条目自己没有片段，只要它有 curated 片段，
        这把尺子照样能用 —— 只查条目 solution 会**低估**原料可得性。
        """
        own_frags = agent._extract_error_code_fragments(rec["solution"])
        cur_frag_rows = sum(1 for s in cur_sols.get(_id, [])
                            if agent._extract_error_code_fragments(s))
        return {
            "cve": rec["cve"], "id": _id,
            "own_frag": len(own_frags), "cur_frag": cur_frag_rows,
            "usable": bool(own_frags) or cur_frag_rows > 0,
            "n_curated": cur_cnt.get(_id, 0),
            "file": rec["file_pattern"], "cls": rec["class_pattern"],
        }

    pick = [by_cve[c] for c in targets if c in by_cve] if targets else list(rows.items())
    title = "批次里 %d 个样本" % len(pick) if targets else "整个知识库 %d 条" % len(rows)

    print("=" * 104)
    print("知识条目体检：%s   （库=%s）" % (title, args.db))
    print("=" * 104)
    print("  %-16s %-5s %-8s %-9s %-8s %-9s %-8s %s" % (
        "CVE", "id", "条目自带", "curated里", "可用?", "curated数", "class名", "file_pattern"))
    stat = {"usable": 0, "by_curated_only": 0, "unusable": 0, "no_file": 0, "no_cls": 0}
    checks = []
    for _id, rec in sorted(pick, key=lambda x: x[1]["cve"]):
        r = check(_id, rec)
        checks.append(r)
        stat["usable"] += int(r["usable"])
        stat["by_curated_only"] += int(not r["own_frag"] and r["cur_frag"] > 0)
        stat["unusable"] += int(not r["usable"])
        stat["no_file"] += int(not r["file"])
        stat["no_cls"] += int(not r["cls"])
        print("  %-16s %-5d %-8d %-9d %-8s %-9d %-8s %s" % (
            r["cve"], r["id"], r["own_frag"], r["cur_frag"],
            "是" if r["usable"] else "**否**", r["n_curated"],
            (r["cls"][:8] or "**空**"), r["file"][:34]))

    n = len(pick)
    print("\n  汇总（共 %d 条）：" % n)
    print("    克隆判据**可用**（条目自带 或 curated 里 有片段）: %d / %d = %.1f%%" % (
        stat["usable"], n, 100 * stat["usable"] / max(1, n)))
    print("      其中只靠 curated 片段才可用                   : %d" % stat["by_curated_only"])
    print("    **两处都没有片段 → 这把尺子对它天生用不上**      : %d / %d = %.1f%%" % (
        stat["unusable"], n, 100 * stat["unusable"] / max(1, n)))
    print("    class_pattern 为空（没有类名锚点）             : %d / %d = %.1f%%" % (
        stat["no_cls"], n, 100 * stat["no_cls"] / max(1, n)))
    print("    file_pattern 为空                              : %d" % stat["no_file"])

    cnts = sorted((r["n_curated"] for r in checks), reverse=True)
    if cnts:
        zero = sum(1 for c in cnts if c == 0)
        print("\n    curated 片段数分布: 最多 %d，中位 %d，**为 0 的有 %d 条**" % (
            cnts[0], cnts[len(cnts) // 2], zero))
        if zero:
            print("      （curated 为 0 的条目在 curated 通道上完全匹配不到 —— 而实测 curated 通道贡献了八成候选）")
    print("\n  怎么用：'两处都没有片段' 或 'class_pattern 为空' 的条目，**不是调门控能救的**，")
    print("          要回到知识库构建侧把原料补齐（这两项都能在入库时自动检查）。")


if __name__ == "__main__":
    main()
