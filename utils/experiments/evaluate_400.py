# -*- coding: utf-8 -*-
"""400 样本评测：召回(200 kb) + 误报(200 held)。**输出自带分层口径**。

召回口径（kb）：报告 new_findings 里存在 evidence.sqlite_id == 该 CVE 自身知识库条目 id。
误报口径（held）：报告 new_findings 条数 > 0（答案不在库，任何命中都是误报）。

## 分层（口径已固化，实现见 utils/eval_strata.py）

`held` 组内部混着两类性质不同的样本，一刀切算出来的"误报率"含义不唯一：

* `held-pure`      —— 该文件在库里根本没有 → 命中即**真误报**
* `held-same-file` —— 该文件在库、但库里那条记录属于**另一个 CVE** → 第三类现象，
                       既不是干净误报（文件确实是同一个）也不是干净召回（编号对不上）

同理 `kb` 组也分两层：`kb-self`（同文件无别的条目）/ `kb-shared-file`（同文件还有别的 CVE）。

本脚本因此对每一行都写出 `group` / `subclass` 字段，并在结尾分开展示。
分层只依赖 SQLite + 数据集 metadata，可离线复算。

前置：已跑完 utils/experiments/test_400_error_batch.json。
运行（MAS 根目录）：
    python utils/experiments/evaluate_400.py
输出：
    reports/eval_400.csv（含 group / subclass / same_file_siblings 列）
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.eval_strata import (  # noqa: E402
    G_HELD_PURE, G_HELD_SAME, G_KB_SELF, G_KB_SHARED, SUBCLASS_LABEL, build_strata,
    check_role_consistency, counts, warn_if_inconsistent,
)
from utils.kb_coverage import normalize_key  # noqa: E402

REPORTS = ROOT / "reports/analysis"
DB = ROOT / "infrastructure/database/mas.db"


def _latest_run(cve_dir: Path):
    runs = [d for d in cve_dir.iterdir() if d.is_dir()] if cve_dir.exists() else []
    if not runs:
        return None
    return max(runs, key=lambda d: d.stat().st_mtime)


def _collect(cve: str):
    run = _latest_run(REPORTS / cve)
    if run is None:
        return False, []
    findings = []
    for f in (run / "second_pass/consolidated").glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for nf in j.get("new_findings", []):
            ev = nf.get("evidence") or {}
            findings.append({
                "channel": ev.get("channel"),
                "sqlite_id": ev.get("sqlite_id"),
                "file": ev.get("file_pattern") or "",
                "class": ev.get("class_pattern") or "",
            })
    return True, findings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "reports/negative_exp_manifest_400_error.json"))
    ap.add_argument("--csv", default=str(ROOT / "reports/eval_400.csv"))
    ap.add_argument("--no-strata", action="store_true",
                    help="不计算分层（仅为兼容旧输出，正常评测不要加）")
    ap.add_argument("--strata-json", default=None, help="把分层明细另存为 JSON")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["rows"]
    kb_cves = [r["cve"] for r in rows if r["role"] == "kb"]
    held_cves = [r["cve"] for r in rows if r["role"] == "held"]

    # ---- 分层口径（唯一权威实现 utils/eval_strata.py）----
    strata = {} if args.no_strata else build_strata([r["cve"] for r in rows], DB)

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id, title FROM issue_patterns")
    id_by_title = {t: i for i, t in cur.fetchall()}
    # curated_issues 是另一套 id（通过 pattern_id 关联到 issue_patterns）
    cur.execute("SELECT id, pattern_id FROM curated_issues")
    ci_id_to_pattern = {i: p for i, p in cur.fetchall()}
    # 库内条目 id → 它自己的文件（末两级），用于判断"命中的兄弟是不是同一个文件"
    cur.execute("SELECT id, file_pattern FROM issue_patterns")
    kb_file_by_id = {i: normalize_key(fp, 2) for i, fp in cur.fetchall()}
    con.close()

    def _is_self(f, own_ip_id):
        if own_ip_id is None:
            return False
        if f["channel"] == "curated_issue":
            return ci_id_to_pattern.get(f["sqlite_id"]) == own_ip_id
        return f["sqlite_id"] == own_ip_id

    def _sibling_hit_same_file(cve: str, cross_ids):
        """kb 样本没命中自己、却命中了别的库内条目时：那个条目是不是同一个文件？

        这是"召回失败"里唯一值得单独拎出来的子类：同一个文件、常常还是同一个位置，
        只是库里那条记录挂着另一个 CVE 编号 —— 与 held-same-file 是同一现象的另一面。
        """
        mine = set(strata.get(cve, {}).get("file_keys") or [])
        if not mine or not cross_ids:
            return ""
        for i in cross_ids:
            k = kb_file_by_id.get(i)
            if k and k in mine:
                return "same_file"
        return "cross_file"

    def _sib_cves(cve: str) -> str:
        st = strata.get(cve, {})
        return ",".join(s["cve"] for s in (st.get("kb_siblings") or []))

    out_rows = []
    for cve in kb_cves:
        ok, fs = _collect(cve)
        own = id_by_title.get(cve)
        captured = any(_is_self(f, own) for f in fs)
        cross = [f["sqlite_id"] for f in fs if not _is_self(f, own)]
        st = strata.get(cve, {})
        out_rows.append({"role": "kb", "cve": cve, "status": "ok" if ok else "missing",
                         "n_findings": len(fs), "captured": captured,
                         "cross_match_ids": ",".join(str(x) for x in cross),
                         "group": st.get("group", ""),
                         "subclass": st.get("subclass", ""),
                         "same_file_siblings": _sib_cves(cve),
                         "sibling_hit_same_file": (
                             _sibling_hit_same_file(cve, cross) if not captured else "")})
    for cve in held_cves:
        ok, fs = _collect(cve)
        st = strata.get(cve, {})
        out_rows.append({"role": "held", "cve": cve, "status": "ok" if ok else "missing",
                         "n_findings": len(fs), "fp": len(fs) > 0, "cross_match_ids": "",
                         "group": st.get("group", ""),
                         "subclass": st.get("subclass", ""),
                         "same_file_siblings": _sib_cves(cve),
                         "sibling_hit_same_file": ""})

    kb_ok = [r for r in out_rows if r["role"] == "kb" and r["status"] == "ok"]
    kb_missing = [r for r in out_rows if r["role"] == "kb" and r["status"] == "missing"]
    held_ok = [r for r in out_rows if r["role"] == "held" and r["status"] == "ok"]
    held_missing = [r for r in out_rows if r["role"] == "held" and r["status"] == "missing"]

    captured_n = sum(1 for r in kb_ok if r["captured"])
    fp_n = sum(1 for r in held_ok if r.get("fp"))

    print("=" * 62)
    print(f"400 样本评测  (kb={len(kb_cves)}, held={len(held_cves)})")
    print("=" * 62)
    print(f"召回池(kb, 答案在库): 有效 {len(kb_ok)} / 缺失 {len(kb_missing)}")
    if kb_ok:
        print(f"  正确捕捉: {captured_n}/{len(kb_ok)} = {captured_n/len(kb_ok):.1%}")
    else:
        print("  正确捕捉: (无有效样本)")
    print(f"误报池(held, 答案不在库): 有效 {len(held_ok)} / 缺失 {len(held_missing)}")
    if held_ok:
        print(f"  误报(旧口径一刀切): {fp_n}/{len(held_ok)} = {fp_n/len(held_ok):.1%}")
        avg = sum(r["n_findings"] for r in held_ok) / len(held_ok)
        print(f"  平均误报条数: {avg:.2f}")

    # ---------------- 分层展示（旧输出只有上面两行） ----------------
    if strata:
        print("\n" + "-" * 62)
        print("分层口径（逐层给率，替代一刀切）")
        print("-" * 62)
        _usable = warn_if_inconsistent(
            check_role_consistency(strata, {r["cve"]: r["role"] for r in rows}), "该批次")
        if not _usable:
            print("  → 分层数字不可用：批次标签对应的是**另一份知识库**，"
                  "请用该轮的知识库重算（--no-strata 可先看旧口径）。")
        _note = {
            G_KB_SELF: "干净召回",
            G_KB_SHARED: "同文件另有库内条目（编号可能对不上）",
            G_HELD_PURE: "真误报 ← 论文该报这一层",
            G_HELD_SAME: "命中同文件、但属别的 CVE ← 第三类现象",
        }
        for label, keys, metric in (
            ("召回池", (G_KB_SELF, G_KB_SHARED), "captured"),
            ("误报池", (G_HELD_PURE, G_HELD_SAME), "fp"),
        ):
            for k in keys:
                grp = [r for r in out_rows if r.get("group") == k and r["status"] == "ok"]
                if not grp:
                    continue
                hit = sum(1 for r in grp if r.get(metric))
                print("  %s %-16s %3d/%3d = %5.1f%%   %s" % (
                    label, k, hit, len(grp), 100 * hit / len(grp), _note[k]))
        same = [r for r in held_ok if r.get("group") == G_HELD_SAME]
        if same:
            print("\n  held-same-file 子类（被标记命中数）:")
            by = {}
            for r in same:
                by.setdefault(r.get("subclass") or "?", []).append(r)
            for s in ("A", "B", "C", "?"):
                grp = by.get(s)
                if grp:
                    print("    %s %-36s %2d/%2d 命中" % (
                        s, SUBCLASS_LABEL[s], sum(1 for r in grp if r.get("fp")), len(grp)))
        sib_same = [r for r in kb_ok if r.get("sibling_hit_same_file") == "same_file"]
        sib_cross = [r for r in kb_ok if r.get("sibling_hit_same_file") == "cross_file"]
        if sib_same or sib_cross:
            print("\n  召回失败细分（未命中自己、却命中了别的库内条目）:")
            print("    命中同文件兄弟（第三类现象的另一面）: %d  %s" % (
                len(sib_same), [r["cve"] for r in sib_same]))
            print("    命中异文件条目（纯误检）          : %d  %s" % (
                len(sib_cross), [r["cve"] for r in sib_cross][:8]))

    if kb_missing:
        print(f"\n召回池缺失: {', '.join(r['cve'] for r in kb_missing)}")
    if held_missing:
        print(f"误报池缺失: {', '.join(r['cve'] for r in held_missing)}")

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["role", "cve", "group", "subclass", "status", "n_findings",
              "captured", "fp", "cross_match_ids", "same_file_siblings", "sibling_hit_same_file"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n明细已写: {out}")

    if args.strata_json and strata:
        Path(args.strata_json).write_text(json.dumps(
            {"counts": counts(strata), "items": strata}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"分层明细已写: {args.strata_json}")


if __name__ == "__main__":
    main()
