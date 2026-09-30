# -*- coding: utf-8 -*-
"""给**已有的**评测 CSV 补上分层列，并重算分层指标（不需要重跑流水线）。

## 为什么需要它

`evaluate_400.py` 现在自己就会写分层列，但历史那几份 CSV（seed=2024 原始 / v4 / seed=2025）
是在分层口径固化**之前**产出的。分层只依赖「CVE + 知识库 + 数据集 metadata」，
与被测代码、与运行产物都无关，所以完全可以**事后补算**——不必为了报一个新口径而重跑 GPU。

这也是口径固化的意义：改的是"怎么读结果"，不是"结果本身"。

## 用法

    python utils/experiments/restamp_eval_strata.py --in reports/eval_400_error_v4.csv \
        --out reports/eval_400_error_v4_strat.csv
    # 一次处理多份
    python utils/experiments/restamp_eval_strata.py \
        --in reports/eval_400_seed2025.csv --in reports/eval_400_error_v4.csv --out-dir reports/strat

输出：新 CSV（原列保留 + group/subclass/same_file_siblings/sibling_hit_same_file 四列）+ 分层指标表。
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.eval_strata import (  # noqa: E402
    G_HELD_PURE, G_HELD_SAME, G_KB_SELF, G_KB_SHARED, SUBCLASS_LABEL, build_strata,
    check_role_consistency, warn_if_inconsistent,
)
from utils.kb_coverage import normalize_key  # noqa: E402

DB = ROOT / "infrastructure/database/mas.db"
EXTRA_COLS = ["group", "subclass", "same_file_siblings", "sibling_hit_same_file"]


def truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "是")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", type=Path, action="append", required=True,
                    help="已有评测 CSV（可重复传多次）")
    ap.add_argument("--out", type=Path, default=None, help="单文件输出路径（只传一个 --in 时用）")
    ap.add_argument("--out-dir", type=Path, default=None, help="多文件时的输出目录")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if len(args.inputs) > 1 and not args.out_dir:
        raise SystemExit("多份输入请用 --out-dir")

    con = sqlite3.connect(str(args.db))
    kb_file_by_id = {i: normalize_key(fp, 2)
                     for i, fp in con.execute("select id, file_pattern from issue_patterns")}
    con.close()

    for path in args.inputs:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        if not rows:
            print("[跳过] %s 是空的" % path)
            continue
        strata = build_strata([r["cve"] for r in rows], args.db)
        consistency = check_role_consistency(
            strata, {str(r["cve"]).strip().upper(): r.get("role", "") for r in rows})

        for r in rows:
            st = strata.get(str(r["cve"]).strip().upper(), {})
            r["group"] = st.get("group", "")
            r["subclass"] = st.get("subclass", "")
            r["same_file_siblings"] = ",".join(s["cve"] for s in (st.get("kb_siblings") or []))
            # kb 样本：没命中自己却命中了别的库内条目时，那个条目是不是同一个文件
            r["sibling_hit_same_file"] = ""
            if r.get("role") == "kb" and not truthy(r.get("captured")):
                mine = set(st.get("file_keys") or [])
                ids = [x.strip() for x in str(r.get("cross_match_ids") or "").split(",") if x.strip()]
                verdict = ""
                for i in ids:
                    try:
                        k = kb_file_by_id.get(int(i))
                    except ValueError:
                        k = None
                    if k and k in mine:
                        verdict = "same_file"
                        break
                if ids and not verdict:
                    verdict = "cross_file"
                r["sibling_hit_same_file"] = verdict

        # ---- 输出路径 ----
        if args.out and len(args.inputs) == 1:
            out = args.out
        elif args.out_dir:
            out = args.out_dir / (path.stem + "_strat.csv")
        else:
            out = path.with_name(path.stem + "_strat.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0].keys())
        for c in EXTRA_COLS:
            if c not in cols:
                cols.append(c)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

        if args.quiet:
            print("已写出: %s" % out)
            continue

        # ---- 分层指标 ----
        kb = [r for r in rows if r.get("role") == "kb"]
        held = [r for r in rows if r.get("role") == "held"]
        print("=" * 88)
        print("%s   (kb=%d, held=%d)" % (path.name, len(kb), len(held)))
        print("=" * 88)
        usable = warn_if_inconsistent(consistency, path.name)
        if not usable:
            print("  → 已写出带分层列的 CSV 供排查，但**下面不出数**。\n")
            continue

        def rate(grp, metric):
            if not grp:
                return "  (无样本)"
            hit = sum(1 for r in grp if truthy(r.get(metric)))
            return "%3d/%3d = %5.1f%%" % (hit, len(grp), 100 * hit / len(grp))

        for label, key, metric in (
            ("召回率  kb-self      ", G_KB_SELF, "captured"),
            ("召回率  kb-shared    ", G_KB_SHARED, "captured"),
            ("误报率  held-pure    ", G_HELD_PURE, "fp"),
            ("误报率  held-samefile", G_HELD_SAME, "fp"),
        ):
            print("  %s %s" % (label, rate([r for r in rows if r.get("group") == key], metric)))

        # 召回失败细分
        nf = [r for r in kb if not truthy(r.get("captured"))]
        sf = [r for r in nf if r.get("sibling_hit_same_file") == "same_file"]
        cf = [r for r in nf if r.get("sibling_hit_same_file") == "cross_file"]
        if sf or cf:
            print("  召回失败中：命中同文件兄弟 %d 个 %s；命中异文件条目 %d 个" % (
                len(sf), [r["cve"] for r in sf], len(cf)))

        # held-same-file 子类
        same = [r for r in rows if r.get("group") == G_HELD_SAME]
        if same:
            by = {}
            for r in same:
                by.setdefault(r.get("subclass") or "?", []).append(r)
            print("  held-same-file 子类：", end="")
            print("；".join("%s(%s) %d/%d 命中" % (
                s, SUBCLASS_LABEL[s].split("（")[0],
                sum(1 for r in by[s] if truthy(r.get("fp"))), len(by[s]))
                for s in ("A", "B", "C", "?") if s in by))
        print("  已写出: %s\n" % out)

    print("完成。分层只依赖 CVE + 知识库 + 数据集 metadata，与被测代码版本无关，故可事后补算。")


if __name__ == "__main__":
    main()
