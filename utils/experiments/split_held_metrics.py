# -*- coding: utf-8 -*-
"""把库外(held)组按"文件是否在库"拆成两层，重算指标；同时核查召回(kb)侧的同类现象。

产出：
  1. held 拆分：held-pure（文件不在库） vs held-same-file（文件在库但条目属别的 CVE）
     并给出各子组的"命中率"，与旧口径（一刀切）对照
  2. held-same-file 内部再分三类：
     A 同漏洞异号（修复文本几乎相同）
     B 同文件 + 同类型 + 不同修复
     C 同文件 + 不同类型
  3. kb 侧核查：哪些样本"没命中自己的条目，却命中了兄弟条目"（用 cross_match_ids 作证据）
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import normalize_key  # noqa: E402

DB = ROOT / "infrastructure/database/mas.db"
EVAL = ROOT / "reports/eval_400_error_v4.csv"
AUDIT = ROOT / "reports/role_audit_400_error.json"
PAIRS = ROOT / "reports/held_same_file_cross_cve_analysis.json"
OUT = ROOT / "reports/held_split_metrics.json"


def truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "是")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=EVAL, help="逐 CVE 评测 CSV")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    eval_path, out_path = args.eval, args.out
    print("评测文件: %s\n" % eval_path)
    rows = list(csv.DictReader(open(eval_path, encoding="utf-8")))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    conf = {r["cve"] for r in audit["items"] if r["role_auto"] == "held" and r.get("file_in_kb")}
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]
    by_cve = defaultdict(list)
    for p in pairs:
        by_cve[p["held_cve"]].append(p)

    c = sqlite3.connect(str(DB))
    kb_rows = {r[0]: {"cve": (r[1] or "").strip().upper(), "file_pattern": r[2] or ""}
               for r in c.execute("select id, title, file_pattern from issue_patterns").fetchall()}
    c.close()
    # 每个 CVE 自己在库里的条目对应的文件（kb 组样本必然有）
    own_key2 = {v["cve"]: normalize_key(v["file_pattern"], 2)
                for v in kb_rows.values() if v["cve"] and v["file_pattern"]}

    held = [r for r in rows if r["role"] == "held"]
    kb = [r for r in rows if r["role"] == "kb"]

    print("=" * 96)
    print("一、库外(held)组拆分：按『源文件是否在知识库里』")
    print("=" * 96)
    pure = [r for r in held if r["cve"] not in conf]
    same = [r for r in held if r["cve"] in conf]
    fp_pure = [r for r in pure if truthy(r["fp"])]
    fp_same = [r for r in same if truthy(r["fp"])]
    n, a, b = len(held), len(pure), len(same)
    na, nb = len(fp_pure), len(fp_same)
    print("  旧口径（一刀切）: 命中 %d / %d = %.1f%%" % (na + nb, n, 100 * (na + nb) / n))
    print("  ├─ held-pure       （文件不在库）: %d / %d = %.1f%%   ← 这才叫『真误报』" % (
        na, a, 100 * na / max(1, a)))
    print("  └─ held-same-file  （文件在库）  : %d / %d = %.1f%%   ← 那 22 个样本" % (
        nb, b, 100 * nb / max(1, b)))
    print("  两者相差 %s" % (
        "%.1f 倍" % ((nb / max(1, b)) / (na / max(1, a))) if na > 0 else "无穷大（真误报为 0）"))

    print("\n" + "=" * 96)
    print("二、held-same-file 内部三类细分（只统计被标记命中的那 %d 个）" % nb)
    print("=" * 96)
    cat = {"A 同漏洞异号": [], "B 同文件+同类型+不同修复": [], "C 同文件+不同类型": [], "D 无法判定": []}
    for r in same:
        cve = r["cve"]
        ps = sorted(by_cve.get(cve, []), key=lambda x: -x["solution_similarity"])
        if not ps:
            cat["D 无法判定"].append(cve)
            continue
        best = ps[0]
        if best["solution_similarity"] > 0.95:
            cat["A 同漏洞异号"].append(cve)
        elif best["same_cwe"] or best["same_classification"] or best["same_error_type"]:
            cat["B 同文件+同类型+不同修复"].append(cve)
        else:
            cat["C 同文件+不同类型"].append(cve)
    for k, v in cat.items():
        if not v:
            continue
        hit = sum(1 for x in v if any(truthy(r["fp"]) for r in same if r["cve"] == x))
        print("  %-26s 共 %-3d 个，其中被标记命中 %-3d 个" % (k, len(v), hit))
        print("       %s" % (v[:8]))

    print("\n" + "=" * 96)
    print("三、召回(kb)侧核查：是否也存在『命中兄弟而非自己』")
    print("=" * 96)
    captured = [r for r in kb if truthy(r["captured"])]
    notcap = [r for r in kb if not truthy(r["captured"])]
    cross = [r for r in kb if str(r["cross_match_ids"]).strip()]
    print("  kb 组 %d 个：命中自己条目 %d（%.1f%%），未命中 %d" % (
        len(kb), len(captured), 100 * len(captured) / len(kb), len(notcap)))
    print("  带 cross_match_ids（额外命中了别的库内条目）的行数: %d" % len(cross))
    print("\n  未命中自己、但命中了兄弟条目的样本（这才是『同文件同问题、换了 CVE』的召回失败）：")
    print("  %-16s %-30s %-9s %s" % ("样本CVE", "样本文件", "命中库行", "库行属于哪个CVE / 是否同文件"))
    sib_n = 0
    sib_same_file = 0
    for r in notcap:
        ids = [x.strip() for x in str(r["cross_match_ids"]).split(",") if x.strip()]
        if not ids:
            continue
        sib_n += 1
        # 本样本自己的文件：从知识库里它自己那条记录取（kb 组样本必然在库）
        my_key = own_key2.get(str(r["cve"]).strip().upper(), "")
        detail = []
        same_file = False
        for i in ids:
            try:
                kr = kb_rows.get(int(i))
            except ValueError:
                kr = None
            if not kr:
                continue
            k2 = normalize_key(kr["file_pattern"], 2)
            is_same = bool(my_key and k2 and k2 == my_key)
            same_file = same_file or is_same
            detail.append("%s(%s)" % (kr["cve"], "同文件" if is_same else "异文件"))
        sib_same_file += int(same_file)
        print("  %-16s %-30s %-9s %s" % (r["cve"], my_key[:30], ",".join(ids), "; ".join(detail[:3])))
    print("\n  ⇒ 未命中自己但命中兄弟的样本: %d 个；其中兄弟与本样本**同文件**的: %d 个" % (
        sib_n, sib_same_file))

    print("\n" + "=" * 96)
    print("四、新旧口径对照表（论文可用）")
    print("=" * 96)
    print("  %-34s %-16s %-16s" % ("指标", "旧口径", "拆分后"))
    print("  %-34s %-16s %-16s" % ("库外组命中率(错配样本率)", "%.1f%% (%d/%d)" % (
        100 * (na + nb) / n, na + nb, n), "—"))
    print("  %-34s %-16s %-16s" % ("  held-pure 命中率", "未单列", "%.1f%% (%d/%d)" % (
        100 * na / max(1, a), na, a)))
    print("  %-34s %-16s %-16s" % ("  held-same-file 命中率", "未单列", "%.1f%% (%d/%d)" % (
        100 * nb / max(1, b), nb, b)))
    print("  %-34s %-16s %-16s" % ("库内组召回率", "%.1f%% (%d/%d)" % (
        100 * len(captured) / len(kb), len(captured), len(kb)), "同上（未拆）"))

    json.dump({
        "held_total": n, "held_pure": a, "held_same_file": b,
        "fp_total": na + nb, "fp_pure": na, "fp_same_file": nb,
        "fp_rate_old": round(100 * (na + nb) / n, 2),
        "fp_rate_pure": round(100 * na / max(1, a), 2),
        "fp_rate_same_file": round(100 * nb / max(1, b), 2),
        "subclasses": {k: v for k, v in cat.items() if v},
        "kb_total": len(kb), "kb_captured": len(captured),
        "kb_not_captured": len(notcap),
        "kb_cross_match_rows": len(cross),
        "kb_not_captured_but_sibling_hit": sib_n,
        "kb_sibling_same_file": sib_same_file,
    }, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写出:", out_path)


if __name__ == "__main__":
    main()
