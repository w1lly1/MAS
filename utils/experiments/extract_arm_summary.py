# -*- coding: utf-8 -*-
"""把一整臂的 run 产物**压成决策摘要**（小体积、可长期留档、可复核）。

## 为什么需要它

服务器的 run 目录很贵：一个库内 30 样本臂 ≈ 0.5 GB、held 臂 ≈ 0.9 GB，
`reports/analysis` 累计 16 GB，而盘只剩 5.7 GB —— **400 样本约需 8 GB，装不下**。
但"删了旧臂"又会丢掉"当时到底放行了哪几条"的复核能力。

所以做法是：**先把判定相关的信息抽成摘要留档，确认摘要能复现出原来的结论，再删原目录**
（用户 2026-10-04 拍板的口径："先瘦身再删"）。

## 摘要里有什么 / 没有什么

有（判定真正依赖的）：每样本的 自己条目 id、候选总数、"自己条目是否进了候选"、
**每一条放行**（通道 / 条目 id / 文件）、new_findings 数、被分析文件列表。
没有（体积元凶，判定不用）：每条候选内嵌的整份源码、prompt、各 agent 的逐条日志。

## 一条逻辑只允许一个实现

候选/放行的解析全部复用 `compare_arms.load_arm`（与 `eval_held_runs.py` 同一实现），
本脚本只做"落盘 + 汇总 + 对照"。同时**内置正向对照**：汇总出来的
"自己条目被放行数 / 放行总数"必须等于 `--expect-own-admitted` / `--expect-total-admitted`
（跑之前从已归档的 `*_compare.txt` / `*_eval.txt` 抄进来）——对不上就拒绝写摘要。

用法：
    python -X utf8 utils/experiments/extract_arm_summary.py \
        --tag baseline_fix --runs reports/baseline_fix_runs.txt \
        --db infrastructure/database/mas.db \
        --expect-own-admitted 28 --expect-total-admitted 65
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.experiments.compare_arms import load_arm  # noqa: E402


def load_db(db: Path):
    con = sqlite3.connect("file:%s?mode=ro" % db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    ci_to_pattern = {int(i): int(p) for i, p in
                     con.execute("select id, pattern_id from curated_issues")}
    con.close()
    return id_by_title, ci_to_pattern


def summarize(arm: dict) -> tuple[list, dict]:
    """把 load_arm 的结果压成"每样本一行 + 整臂汇总"。

    ⚠️ **两个口径必须分开报，别混**（第一版就混了，正向对照当场抓出来）：
      * **样本级**："30 个样本里有 28 个的**自己条目**被放行了" —— 召回率口径；
      * **记录级**：一个样本有多个文件/多个 issue，每个都可能放行一次 ⇒ 65 条放行记录。
    所以下面 `samples_own_admitted` 与 `admission_records_*` 是两套数，注释写死各是什么。
    """
    rows = []
    chan_all, chan_own = Counter(), Counter()
    own_records = total_records = non_own_records = 0
    for cve, rec in sorted(arm.items()):
        own = rec.get("own")
        adm = []
        for a in rec.get("admitted") or []:
            sid = a.get("sqlite_id")
            is_own = own is not None and sid is not None and int(sid) == int(own)
            own_records += is_own
            total_records += 1
            non_own_records += (not is_own)
            chan_all[a.get("channel")] += 1
            if is_own:
                chan_own[a.get("channel")] += 1
            adm.append({"channel": a.get("channel"), "id": sid, "raw_id": a.get("raw_id"),
                        "file": a.get("file"), "is_own": bool(is_own)})
        rows.append({
            "cve": cve, "own": own, "cand": rec.get("cand"),
            "own_in_cand": rec.get("own_in_cand"),
            "sample_own_admitted": any(x["is_own"] for x in adm),
            "n_admission_records": len(adm), "n_new_findings": rec.get("nb_findings"),
            "files": sorted(x for x in (rec.get("files") or set()) if x),
            "admitted": adm,
        })
    agg = {
        "samples": len(rows),
        "candidates_total": sum(r["cand"] or 0 for r in rows),
        "samples_with_own_in_cand": sum(1 for r in rows if r["own_in_cand"]),
        # 样本级（召回口径）：自己条目被放行的**样本**数
        "samples_own_admitted": sum(1 for r in rows if r["sample_own_admitted"]),
        # 记录级：放行**记录**数（一个样本多文件时会 >1）
        "admission_records_total": total_records,
        "admission_records_own": own_records,
        "admission_records_non_own": non_own_records,
        "channel_all": dict(chan_all),
        "channel_own": dict(chan_own),
        "samples_own_not_admitted": [r["cve"] for r in rows
                                     if r["own_in_cand"] and not r["sample_own_admitted"]],
    }
    return rows, agg


def main() -> int:
    ap = argparse.ArgumentParser(description="把一臂压成决策摘要（供删除原目录前留档）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/arm_summaries")
    ap.add_argument("--expect-own-admitted", type=int, default=None,
                    help="期望的**样本级**\"自己条目被放行\"样本数（如基线 28）")
    ap.add_argument("--expect-total-admitted", type=int, default=None,
                    help="期望的**记录级**放行记录总数（如基线 65）")
    ap.add_argument("--expect-non-own", type=int, default=None,
                    help="期望的非 own 放行记录数（库内基线应为 0）")
    args = ap.parse_args()

    runs = args.runs if args.runs.is_absolute() else ROOT / args.runs
    db = args.db if args.db.is_absolute() else ROOT / args.db
    if not runs.is_file():
        raise SystemExit("run 清单不存在: %s" % runs)
    if not db.is_file():
        raise SystemExit("库不存在: %s" % db)

    id_by_title, ci_to_pattern = load_db(db)
    arm = load_arm(runs, id_by_title, ci_to_pattern)
    rows, agg = summarize(arm)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.out_dir / ("%s.jsonl" % args.tag)
    with jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg_path = args.out_dir / ("%s_aggregate.json" % args.tag)
    agg_full = dict(agg, tag=args.tag, runs_file=str(runs.relative_to(ROOT))
                    if runs.is_relative_to(ROOT) else str(runs))
    agg_path.write_text(json.dumps(agg_full, ensure_ascii=False, indent=1), encoding="utf-8")

    print("臂 %s（%s）" % (args.tag, runs.name))
    print("  样本 %d / 候选 %d / 有自己条目的样本 %d"
          % (agg["samples"], agg["candidates_total"], agg["samples_with_own_in_cand"]))
    print("  **样本级**: 自己条目被放行的样本 %d / %d"
          % (agg["samples_own_admitted"], agg["samples"]))
    print("  **记录级**: 放行记录总 %d（其中 own %d、非 own %d）"
          % (agg["admission_records_total"], agg["admission_records_own"],
             agg["admission_records_non_own"]))
    print("  放行通道: %s" % agg["channel_all"])
    print("  own 放行通道: %s" % agg["channel_own"])
    if agg["samples_own_not_admitted"]:
        print("  有候选但自己条目没被放行的样本: %s" % " ".join(agg["samples_own_not_admitted"]))
    print("  → %s" % jsonl.relative_to(ROOT))
    print("  → %s" % agg_path.relative_to(ROOT))

    bad = []
    if (args.expect_own_admitted is not None
            and agg["samples_own_admitted"] != args.expect_own_admitted):
        bad.append("样本级 own 放行 %d != 期望 %d"
                   % (agg["samples_own_admitted"], args.expect_own_admitted))
    if (args.expect_total_admitted is not None
            and agg["admission_records_total"] != args.expect_total_admitted):
        bad.append("放行记录总数 %d != 期望 %d"
                   % (agg["admission_records_total"], args.expect_total_admitted))
    if (args.expect_non_own is not None
            and agg["admission_records_non_own"] != args.expect_non_own):
        bad.append("非 own 放行 %d != 期望 %d"
                   % (agg["admission_records_non_own"], args.expect_non_own))
    if bad:
        print("\n❌ 正向对照不通过（摘要与已归档结论对不上，**不许据此删原目录**）:")
        for b in bad:
            print("   - %s" % b)
        return 1
    if args.expect_own_admitted is not None or args.expect_total_admitted is not None:
        print("\n✅ 正向对照通过（摘要能复现已归档结论）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
