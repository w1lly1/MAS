# -*- coding: utf-8 -*-
"""为什么这 4 个样本"捞到了却不放行"？—— 打印它们自己那条候选的分数与拒因。

这是把"平局"翻译成"下一步该修哪里"的关键一步：
  * 若 `s(x)`（结构化证据）不足以过线 → 瓶颈是**证据/口径**（同文件锚点、clone 尺子）
  * 若被判 `code_already_fixed` → 是**否决判据**在误杀
  * 若 `v/a`（语义/锚点）差一点 → 才轮到阈值或语义
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_live.db")
    ap.add_argument("--cves", nargs="*", default=None)
    ap.add_argument("--show-candidates", type=int, default=2)
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    ci_to_pattern = {int(i): int(p) for i, p in con.execute("select id, pattern_id from curated_issues")}
    con.close()

    runs = [ln.strip() for ln in args.runs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in runs:
        cve, run = line.split("/", 1)
        if args.cves and cve not in args.cves:
            continue
        own = id_by_title.get(cve.upper())
        d = ROOT / "reports/analysis" / cve / run / "second_pass" / "consolidated"
        if not d.exists():
            print("%s: 缺产物" % cve)
            continue
        print("=" * 96)
        print("%s（自己的条目 id=%s）" % (cve, own))
        printed = 0
        for f in sorted(d.glob("*_r2.json")):
            j = json.loads(f.read_text(encoding="utf-8"))
            for key in ("retrieval_evidence", "gap_retrieval_evidence"):
                for ev in (j.get(key) or []):
                    for c in (ev.get("candidates") or []):
                        if not isinstance(c, dict):
                            continue
                        chan = c.get("channel") or c.get("primary_channel")
                        sid = c.get("sqlite_id")
                        if sid is None:
                            sid = (c.get("evidence") or {}).get("sqlite_id")
                        try:
                            resolved = ci_to_pattern.get(int(sid)) if chan == "curated_issue" else int(sid)
                        except (TypeError, ValueError):
                            continue
                        if resolved != own:
                            continue
                        if printed >= args.show_candidates:
                            break
                        printed += 1
                        print("  通道=%s 通道内 id=%s → 条目 %s  层=%s" %
                              (chan, sid, resolved, c.get("vector_layer") or c.get("layer")))
                        for k in ("structured_score", "semantic_score", "anchor_score",
                                  "context_score", "total_score", "similarity", "score"):
                            if k in c:
                                v = c[k]
                                print("      %-18s %s" % (k, round(v, 4) if isinstance(v, (int, float)) else v))
                        mf = c.get("matched_fields") or c.get("matched") or []
                        print("      matched_fields     %s" % (mf or "(空)"))
                        for k in ("gating_decision", "rejection_reason", "gate_reason", "decision"):
                            if c.get(k):
                                print("      %-18s %s" % (k, c[k]))
                    if printed >= args.show_candidates:
                        break
                if printed >= args.show_candidates:
                    break
            if printed >= args.show_candidates:
                break
        if printed == 0:
            print("  （没找到自己条目的候选 —— 需要复查检索段）")


if __name__ == "__main__":
    main()
