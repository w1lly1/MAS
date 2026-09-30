# -*- coding: utf-8 -*-
"""从 reports/analysis/（seed=2024 完整树）提取通道归因 + 语义独立命中的逐层归属。

删除 23G 树之前先存档，输出 reports/channel_layers_v4_extract.json。
口径与 diag_channel_attribution.py 一致（词法=error_code_clone，语义=weaviate）。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports/analysis"
DB = ROOT / "infrastructure/database/mas.db"
BATCH = next((p for p in [ROOT / "utils/experiments/test_400_error_batch.json", ROOT / "论文/test_400_error_batch.json"] if p.exists()), ROOT / "utils/experiments/test_400_error_batch.json")


def latest_run(cve):
    d = REPORTS / cve
    runs = [x for x in d.iterdir() if x.is_dir()] if d.exists() else []
    return max(runs, key=lambda x: x.stat().st_mtime) if runs else None


def _extract_new_findings(head):
    marker = '"new_findings"'
    idx = head.find(marker)
    if idx < 0:
        return []
    b = head.find("[", idx)
    if b < 0:
        return []
    depth = 0
    in_str = False
    escape = False
    for i in range(b, len(head)):
        c = head[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(head[b:i + 1])
                    except Exception:
                        return []
    return []


def collect(cve):
    run = latest_run(cve)
    if run is None:
        return []
    findings = []
    cd = run / "second_pass" / "consolidated"
    if cd.exists():
        for f in cd.glob("*.json"):
            try:
                with f.open("r", encoding="utf-8", errors="ignore") as fh:
                    head = fh.read(2 * 1024 * 1024)
            except Exception:
                continue
            for nf in _extract_new_findings(head):
                ev = nf.get("evidence") or {}
                findings.append({
                    "channel": ev.get("channel"),
                    "sqlite_id": ev.get("sqlite_id"),
                    "matched_fields": ev.get("matched_fields") or [],
                    "vector_layer": ev.get("vector_layer"),
                    "matched_layers": ev.get("matched_layers") or [],
                })
    return findings


def main():
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    kb = [it["output_dir"] for it in batch["items"] if it["role"] == "kb"]

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id, title FROM issue_patterns")
    id_by_title = {t: i for i, t in cur.fetchall()}
    cur.execute("SELECT id, pattern_id FROM curated_issues")
    ci_to_p = {i: p for i, p in cur.fetchall()}
    con.close()

    def is_self(ch, sid, own):
        if own is None:
            return False
        if ch == "curated_issue":
            return ci_to_p.get(sid) == own
        return sid == own

    total = lex = sem = 0
    only_lex = only_sem = both = 0
    only_sem_layers = {}
    for cve in kb:
        own = id_by_title.get(cve)
        self_hits = [f for f in collect(cve) if is_self(f["channel"], f["sqlite_id"], own)]
        if not self_hits:
            continue
        total += 1
        has_lex = any("error_code_clone" in f["matched_fields"] for f in self_hits)
        has_sem = any(f["channel"] == "weaviate" for f in self_hits)
        if has_lex:
            lex += 1
        if has_sem:
            sem += 1
        if has_lex and has_sem:
            both += 1
        elif has_lex:
            only_lex += 1
        else:
            only_sem += 1
            layers = set()
            for f in self_hits:
                if f["channel"] == "weaviate":
                    for l in (f["matched_layers"] or [f["vector_layer"]]):
                        if l:
                            layers.add(str(l).strip().lower())
            only_sem_layers[cve] = sorted(layers)

    out = {
        "total": total,
        "lex": lex,
        "sem": sem,
        "both": both,
        "only_lex": only_lex,
        "only_sem": only_sem,
        "only_sem_layers": only_sem_layers,
    }
    dest = ROOT / "reports" / "channel_layers_v4_extract.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n已写: {dest}")


if __name__ == "__main__":
    main()
