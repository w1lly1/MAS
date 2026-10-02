#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""③ 第 6 步：**干净层两臂 A/B**（关闭 vs 开启门控融合），逐样本比对放行集合。

口径与 `eval_held_runs.py` / `compare_arms.py` 完全同源：
候选与放行的取数一律复用 `compare_arms.load_arm`，KB 真值来自线上 `mas.db`，
同文件/跨文件分类复用 `eval_held_runs.classify`（**不重写任何判据**）。

输出三件事：
1. 两臂的放行集合差（新增 / 消失）；**由单调性保证不应有"消失"**
2. 每条**新增放行**的性质：同文件 or 跨文件、走哪个通道、是否走了融合分支（`gate_branch`）
3. 预登记预测的逐条对照（跨文件新增必须为 0）

用法（在服务器 MAS 根目录下）:
    venv/bin/python utils/experiments/ab_clean30_arms.py \
        --arms "关=reports/held_clean30_clean30_off_runs.txt" "开=reports/held_clean30_clean30_on_runs.txt" \
        --db infrastructure/database/mas.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.experiments.compare_arms import load_arm  # noqa: E402
from utils.experiments.eval_held_runs import classify  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help='形如 "关=reports/xxx_runs.txt"')
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    file_by_id = {int(i): (fp or "") for i, fp in
                  con.execute("select id, file_pattern from issue_patterns")}
    ci_to_pattern = {int(i): int(p) for i, p in
                     con.execute("select id, pattern_id from curated_issues")}
    con.close()

    if len(args.arms) != 2:
        print("需要两个臂（先关后开）")
        return 1
    (n_off, p_off), (n_on, p_on) = [a.split("=", 1) for a in args.arms]
    A = load_arm(ROOT / p_off if not Path(p_off).is_absolute() else Path(p_off),
                 id_by_title, ci_to_pattern)
    B = load_arm(ROOT / p_on if not Path(p_on).is_absolute() else Path(p_on),
                 id_by_title, ci_to_pattern)

    def key(a):
        return (a.get("sqlite_id"), str(a.get("channel") or ""), str(a.get("file") or ""))

    print("=" * 100)
    print("③ 干净层两臂 A/B：%s（关） vs %s（开）" % (n_off, n_on))
    print("=" * 100)
    tot_off = sum(len(r["admitted"]) for r in A.values())
    tot_on = sum(len(r["admitted"]) for r in B.values())
    print("  样本数        : %d vs %d" % (len(A), len(B)))
    print("  候选总数      : %d vs %d" % (sum(r["cand"] for r in A.values()),
                                        sum(r["cand"] for r in B.values())))
    print("  **放行总数**  : %d vs %d  （差 %+d）" % (tot_off, tot_on, tot_on - tot_off))

    new, gone = [], []
    for cve in sorted(set(A) | set(B)):
        a = {key(x) for x in (A.get(cve) or {}).get("admitted", [])}
        b = {key(x) for x in (B.get(cve) or {}).get("admitted", [])}
        for k in b - a:
            new.append((cve, k))
        for k in a - b:
            gone.append((cve, k))

    same = cross = 0
    print("\n  --- 新增放行逐条（这是本实验的重点）---")
    if not new:
        print("     没有新增放行")
    for cve, (sid, chan, f) in new:
        rec = B.get(cve) or {}
        info = classify({"sqlite_id": sid, "channel": chan, "file": f},
                        rec.get("files") or set(), file_by_id)
        same += info["kind"] == "same_file"
        cross += info["kind"] == "cross_file"
        print("     %-16s sid=%-4s 通道=%-14s 性质=%-10s KB文件=%s"
              % (cve, sid, chan, info["kind"], info["kb_file"][:44]))
    if gone:
        print("\n  ⚠️ 消失的放行（由单调性保证不应出现，出现即为实现错误）: %d 条" % len(gone))
        for cve, (sid, chan, f) in gone[:10]:
            print("     %-16s sid=%-4s 通道=%s" % (cve, sid, chan))

    print("\n  --- 该臂的开关生效性线索 ---")
    import glob as _glob
    for name, runs in ((n_on, p_on), (n_off, p_off)):
        n_fs = n_gb = 0
        for ln in Path(runs).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            d = ROOT / "reports/analysis" / ln / "second_pass"
            files = _glob.glob(str(d / "**" / "*_r2.json"), recursive=True)
            for f in files:
                try:
                    txt = Path(f).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if '"fusion_score"' in txt:
                    n_fs += 1
                if '"gate_branch"' in txt:
                    n_gb += 1
        print("     %-6s 含 fusion_score 的文件 %d 个，含 gate_branch 的 %d 个" % (name, n_fs, n_gb))

    print("\n  --- 预登记预测对照（《06》§七点五）---")
    checks = [
        ("新增放行里的跨文件条数 = 0", cross, 0),
        ("没有「消失的放行」", len(gone), 0),
    ]
    for label, val, want in checks:
        print("     [%s] %-32s 实测 %s（期望 %s）"
              % ("成立" if val == want else "**不成立**", label, val, want))

    out = ROOT / "reports/ab_clean30_arms.json"
    out.write_text(json.dumps({"off_total": tot_off, "on_total": tot_on,
                               "new": [[c, list(k)] for c, k in new],
                               "gone": [[c, list(k)] for c, k in gone],
                               "new_same_file": same, "new_cross_file": cross},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已落盘: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
