# -*- coding: utf-8 -*-
"""召回失败**卡在哪一段**：检索没捞回来，还是捞回来了但被门控拦掉？

## 为什么需要它

"召回率 70%" 这个数字本身不告诉你要改什么。同一批失败样本里可能是两类完全不同的病：

* **检索段**：该检索到的知识条目**根本没出现在候选里**（每层只取前 5 条，正确条目可能排在第 6 名）
  → 该改的是**检索广度**（每层多取几条、或者改查询写法）
* **门控段**：条目**已经被检索回来**，但被门控拦掉了（理由是什么就查什么）
  → 该改的是**门控**（阈值、证据权重、跨文件规则）

两者的处置完全不同，所以要把每个样本**逐段定位**。

## 判据

对每个样本，看它**自己的库内条目 id** 在这条流水线上走到了哪一步：

    retrieved  该 id 出现在任一查询的 weaviate 命中列表里（说明检索捞回来了）
    candidate  该 id 出现在候选里（说明进了门控）
    admitted   门控判定为 formal_hit / explanatory_hit（说明真的报出来了）

于是失败样本分成：`没检索到` / `检索到了没进候选` / `进了候选被拦（附拒绝理由）`。

## 用法（MAS 根目录 / GPU 服务器）

    python utils/experiments/diagnose_recall_stage.py runs.txt --db infrastructure/database/mas.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.eval_strata import build_strata  # noqa: E402

ADMITTED = {"formal_hit", "explanatory_hit"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, help="run id 文件（每行 CVE/<uuid>）")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path,
                    default=ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured")
    args = ap.parse_args()

    rmap = {}
    for line in args.runs.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "/" in line:
            cve, run = line.split("/", 1)
            rmap[cve.strip().upper()] = run.strip()

    con = sqlite3.connect(str(args.db))
    id_by_title = {(t or "").strip().upper(): i
                   for i, t in con.execute("select id, title from issue_patterns")}
    ci_to_pattern = {i: p for i, p in con.execute("select id, pattern_id from curated_issues")}
    con.close()

    strata = build_strata(rmap.keys(), args.db, args.dataset_root)

    print("=" * 104)
    print("召回失败定位：检索段 vs 门控段")
    print("=" * 104)
    print("  %-16s %-16s %-6s %-9s %-9s %-9s %-9s %s" % (
        "CVE", "分层", "own_id", "检索到?", "进候选?", "放行过?", "报出来?", "被拦的理由 / 最近命中"))
    stage = Counter()
    for cve in sorted(rmap):
        own = id_by_title.get(cve)
        g = strata.get(cve, {}).get("group", "?")
        run = rmap[cve]
        if own is None:
            print("  %-16s %-16s %-6s %s" % (cve, g, "-", "该 CVE 不在本库（不是召回样本）"))
            continue
        d = ROOT / "reports/analysis" / cve / run / "second_pass/consolidated"
        retrieved = False          # own id 是否出现在 weaviate 命中里（只对向量通道有意义）
        n_cand = 0                 # own id 出现过多少次候选
        admitted_any = False       # 其中**至少一次**被判为 formal/explanatory
        best_sim = None
        reasons = Counter()
        reported = False           # 与 ab_eval_runsets 同源：final 报告里确有 owner==own 的命中
        for f in sorted(d.glob("*.json")):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for nf in j.get("new_findings", []):
                ev = nf.get("evidence") or {}
                sid = ev.get("sqlite_id")
                mapped = ci_to_pattern.get(sid) if ev.get("channel") == "curated_issue" else sid
                if mapped == own or sid == own:
                    reported = True
            for blk in ("retrieval_evidence", "gap_retrieval_evidence"):
                for it in j.get(blk) or []:
                    for h in (it.get("weaviate_hits") or []):
                        if h.get("sqlite_id") == own:
                            retrieved = True
                            s = h.get("similarity")
                            if isinstance(s, (int, float)) and (best_sim is None or s > best_sim):
                                best_sim = s
                    for c in (it.get("candidates") or []):
                        sid = c.get("sqlite_id")
                        mapped = ci_to_pattern.get(sid) if c.get("channel") == "curated_issue" else sid
                        if mapped == own or sid == own:
                            n_cand += 1
                            # **必须用"任一命中"**：同一个 own 条目会被多轮查询反复评估，
                            # 取"最后一条"会把"某轮放行过"错判成"全程被拦"（本轮踩过这个坑）。
                            if str(c.get("gating_decision")) in ADMITTED:
                                admitted_any = True
                            else:
                                reasons["%s/%s" % (c.get("gating_decision"),
                                                   c.get("rejection_reason"))] += 1
        if reported:
            stage["报出来"] += 1
        elif not retrieved and n_cand == 0:
            stage["检索段没捞到"] += 1
        elif n_cand == 0:
            stage["检索到但没进候选"] += 1
        else:
            stage["进候选被门控拦"] += 1
        why = "-"
        if not reported and reasons:
            why = "%s ×%d" % (reasons.most_common(1)[0][0], reasons.most_common(1)[0][1])
        elif not reported and not retrieved:
            why = "weaviate 命中里没有这条（每层只取前 5 条）"
        print("  %-16s %-16s %-6s %-9s %-9s %-9s %-9s %s" % (
            cve, g, own, "是" if retrieved else "否",
            ("是×%d" % n_cand) if n_cand else "否",
            "是" if admitted_any else "否",
            "是" if reported else "否",
            why + (("  最近相似度=%.4f" % best_sim) if best_sim else "")))

    print("\n" + "-" * 104)
    print("  汇总：")
    for k in ("报出来", "检索段没捞到", "检索到但没进候选", "进候选被门控拦"):
        if stage.get(k):
            print("      %-22s %d" % (k, stage[k]))
    tot = sum(stage.values())
    if tot:
        print("  可召回样本 %d 个，召回 %d 个 = %.1f%%" % (tot, stage.get("报出来", 0),
                                                     100 * stage.get("报出来", 0) / tot))
    print("\n  怎么用：'检索段没捞到' 占多数 → 改检索广度/查询写法；"
          "'进候选被门控拦' 占多数 → 改门控阈值或证据权重。")


if __name__ == "__main__":
    main()
