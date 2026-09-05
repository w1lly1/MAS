# -*- coding: utf-8 -*-
"""400 样本评测：召回(200 kb) + 误报(200 held)。

召回口径（kb）：报告 new_findings 里存在 evidence.sqlite_id == 该 CVE 自身知识库条目 id。
误报口径（held）：报告 new_findings 条数 > 0（答案不在库，任何命中都是误报）。

前置：已跑完 utils/experiments/test_400_error_batch.json。
运行（MAS 根目录）：
    python utils/experiments/evaluate_400.py
输出：
    reports/eval_400.csv
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
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["rows"]
    kb_cves = [r["cve"] for r in rows if r["role"] == "kb"]
    held_cves = [r["cve"] for r in rows if r["role"] == "held"]

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id, title FROM issue_patterns")
    id_by_title = {t: i for i, t in cur.fetchall()}
    # curated_issues 是另一套 id（通过 pattern_id 关联到 issue_patterns）
    cur.execute("SELECT id, pattern_id FROM curated_issues")
    ci_id_to_pattern = {i: p for i, p in cur.fetchall()}
    con.close()

    def _is_self(f, own_ip_id):
        if own_ip_id is None:
            return False
        if f["channel"] == "curated_issue":
            return ci_id_to_pattern.get(f["sqlite_id"]) == own_ip_id
        return f["sqlite_id"] == own_ip_id

    out_rows = []
    for cve in kb_cves:
        ok, fs = _collect(cve)
        own = id_by_title.get(cve)
        captured = any(_is_self(f, own) for f in fs)
        cross = [f["sqlite_id"] for f in fs if not _is_self(f, own)]
        out_rows.append({"role": "kb", "cve": cve, "status": "ok" if ok else "missing",
                         "n_findings": len(fs), "captured": captured,
                         "cross_match_ids": ",".join(str(x) for x in cross)})
    for cve in held_cves:
        ok, fs = _collect(cve)
        out_rows.append({"role": "held", "cve": cve, "status": "ok" if ok else "missing",
                         "n_findings": len(fs), "fp": len(fs) > 0, "cross_match_ids": ""})

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
        print(f"  误报: {fp_n}/{len(held_ok)} = {fp_n/len(held_ok):.1%}")
        avg = sum(r["n_findings"] for r in held_ok) / len(held_ok)
        print(f"  平均误报条数: {avg:.2f}")
    else:
        print("  误报: (无有效样本)")

    if kb_missing:
        print(f"\n召回池缺失: {', '.join(r['cve'] for r in kb_missing)}")
    if held_missing:
        print(f"误报池缺失: {', '.join(r['cve'] for r in held_missing)}")

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["role", "cve", "status", "n_findings",
                                           "captured", "fp", "cross_match_ids"])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"\n明细已写: {out}")


if __name__ == "__main__":
    main()
