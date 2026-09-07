# -*- coding: utf-8 -*-
"""从 reports/analysis/ 提取离线分析所需的关键字段，输出紧凑 JSON（~1-2MB）。

用途：把 GPU 上约 23GB 的逐 CVE 报告，压缩成只有 #1(漏洞分类) 和 #6(漏检归因)
所需字段的精简文件，scp 到本地做离线分析，然后 GPU 磁盘即可释放。

运行（MAS 根目录，需 GPU/数据）：
    python utils/experiments/extract_analysis_fields.py
产出：reports/analysis_extract.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

BASE = ROOT / "reports" / "analysis"
DB = ROOT / "infrastructure" / "database" / "mas.db"
OUT = ROOT / "reports" / "analysis_extract.json"


def main() -> None:
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id, title FROM issue_patterns")
    id_by_title = {t: i for i, t in cur.fetchall()}
    cur.execute("SELECT id, pattern_id FROM curated_issues")
    ci_to_p = {i: p for i, p in cur.fetchall()}
    con.close()

    result = {}
    cve_dirs = sorted(d for d in BASE.iterdir() if d.is_dir())
    for d in cve_dirs:
        runs = sorted((x for x in d.iterdir() if x.is_dir()),
                      key=lambda x: x.stat().st_mtime)
        if not runs:
            continue
        run = runs[-1]
        cve = d.name
        own = id_by_title.get(cve)
        entry: dict = {"cve": cve, "own": own}

        # 1) security agent 漏洞 + 融合策略
        secs = sorted((run / "agents" / "security").glob("*.json")) if (run / "agents" / "security").exists() else []
        if secs:
            try:
                j = json.loads(secs[0].read_text(encoding="utf-8"))
            except Exception:
                j = {}
            a = j.get("security_result", {}).get("ai_security_analysis", {})
            fusion = (a.get("overall_security_rating") or {}).get("fusion") or {}
            entry["security"] = {
                "strategy": fusion.get("strategy"),
                "llm_weight": fusion.get("llm_weight"),
                "vulns": [
                    {
                        "type": v.get("type"),
                        "severity": v.get("severity"),
                        "source": v.get("source"),
                        "desc": str(v.get("description") or "")[:200],
                    }
                    for v in (a.get("vulnerabilities_detected") or [])
                ],
            }

        # 2) second_pass 门控 + 自条目命中
        sp_files = sorted((run / "second_pass" / "consolidated").glob("*.json"))
        if not sp_files:
            sp_files = sorted((run / "fullLayer" / "consolidated").glob("*.json"))
        if sp_files:
            try:
                j = json.loads(sp_files[-1].read_text(encoding="utf-8"))
            except Exception:
                j = {}
            nf = j.get("new_findings", [])
            self_hit = any(
                (
                    ((x.get("evidence") or {}).get("channel") == "curated_issue"
                     and ci_to_p.get((x.get("evidence") or {}).get("sqlite_id")) == own)
                    or (
                        (x.get("evidence") or {}).get("sqlite_id") == own
                        and (x.get("evidence") or {}).get("channel") != "curated_issue"
                    )
                )
                for x in nf
            )
            gating = []
            for ev in j.get("gap_retrieval_evidence", []):
                for cand in (ev.get("candidates") or []):
                    if cand.get("sqlite_id") == own:
                        gating.append({
                            "channel": cand.get("channel"),
                            "layer": cand.get("vector_layer"),
                            "gate": cand.get("gating_decision"),
                            "reject": cand.get("rejection_reason"),
                            "semantic": cand.get("semantic_score"),
                            "structured": cand.get("structured_score"),
                            "unified_s": cand.get("unified_structured_score"),
                            "matched_fields": cand.get("matched_fields"),
                        })
            entry["second_pass"] = {
                "n_new_findings": len(nf),
                "self_hit": self_hit,
                "gating": gating,
            }

        result[cve] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"提取 {len(result)} 个 CVE → {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
