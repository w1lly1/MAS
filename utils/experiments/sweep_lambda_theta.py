#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事后扫 (λ, θ)：用**已有产物**算"换组参数会多放行几条、是 own 还是别的条目"。

为什么能事后算：每条走到 DNF 的候选都落盘了
`s(x)`（`unified_structured_score`）、语义项（`fusion_semantic_term`，**与 λ 无关**）、
以及它是不是过了否决。于是 `s + λ·term ≥ θ` 可以在离线精确重算，**不必为每组参数各跑一臂**。

判据：
* **新增放行** = `s(x) < 0.65`（老规则本来不放行）**且** `s(x) + λ·term ≥ θ`；
* 再按"这条候选是不是**本样本自己的知识条目**"分成 own / 其他 ——
  库内样本上 own 是"召回"，其他是"多放行的面"（是否误报还要看同/跨文件）。
* 候选的"知识条目 id"取 `kb_pattern_id or sqlite_id`（`curated_issues` 的 sqlite_id 是实例 id，见《01》缺陷 23）。

用法（服务器上，MAS 根目录）:
    venv/bin/python utils/experiments/sweep_lambda_theta.py --runs reports/kbself_fix_runs.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

THETA_S = 0.65
LAMBDAS = [1.0, 1.5, 2.0, 3.0, 4.0]
THETAS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.90]


def cand_pid(c: dict):
    """候选对应的**知识条目 id**（模式 id）。"""
    ch = str(c.get("channel") or "").strip().lower()
    if ch == "curated_issue":
        v = c.get("kb_pattern_id")
        return int(v) if v is not None else None
    v = c.get("sqlite_id")
    return int(v) if v is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "reports/kbself_fix_runs.txt")
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--json-out", type=Path, default=ROOT / "reports/sweep_lambda_theta.json")
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    con.close()

    runs = [ln.strip() for ln in args.runs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = []          # 每个"未达阈值"的候选一行
    for rel in runs:
        cve = rel.split("/")[0]
        own = id_by_title.get(cve.upper())
        for f in glob.glob(os.path.join(str(ROOT / "reports/analysis"), rel,
                                        "second_pass", "**", "*_r2.json"), recursive=True):
            try:
                j = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception:
                continue
            for key in ("retrieval_evidence", "gap_retrieval_evidence"):
                for b in (j.get(key) or []):
                    for c in (b.get("candidates") or []):
                        if not isinstance(c, dict) or "fusion_score" not in c:
                            continue
                        s = float(c.get("unified_structured_score") or 0.0)
                        if s >= THETA_S:
                            continue
                        rows.append({
                            "cve": cve, "own": own, "pid": cand_pid(c),
                            "s": s, "term": float(c.get("fusion_semantic_term") or 0.0),
                            "ch": c.get("channel"),
                            "mf": list(c.get("matched_fields") or []),
                        })

    n_cand = len(rows)
    n_term = sum(1 for r in rows if r["term"] > 0)
    print("未达 θ_s=%.2f 的候选：%d 条（其中语义项>0 的 %d 条）" % (THETA_S, n_cand, n_term))
    terms = sorted(r["term"] for r in rows if r["term"] > 0)
    if terms:
        print("  语义项分布（>0 的）: 最小 %.4f / 中位 %.4f / 最大 %.4f"
              % (terms[0], terms[len(terms) // 2], terms[-1]))
    s_vals = sorted({round(r["s"], 2) for r in rows})
    print("  这些候选的 s(x) 取值: %s" % s_vals[:10])

    print("\n%-6s %-6s %10s %10s %12s %s" % ("λ", "θ", "新放行", "其中 own", "其他条目", "会新增 own 的样本"))
    table = []
    for lam in LAMBDAS:
        for th in THETAS:
            hit = [r for r in rows if r["s"] + lam * r["term"] >= th]
            own_hit = [r for r in hit if r["own"] is not None and r["pid"] == r["own"]]
            others = len(hit) - len(own_hit)
            cves = sorted({r["cve"] for r in own_hit})
            table.append({"lam": lam, "theta": th, "new": len(hit),
                          "own": len(own_hit), "other": others, "own_cves": cves})
            print("%-6.1f %-6.2f %10d %10d %12d %s"
                  % (lam, th, len(hit), len(own_hit), others, ",".join(cves[:4]) or "-"))

    print("\n--- 读法 ---")
    print("  · '新放行' = 老规则不放行、但融合分过线 ⇒ 这才是融合**真的多放行**的条数；")
    print("  · '其中 own' = 落在**本样本自己的知识条目**上 ⇒ 那才是**召回**；")
    print("  · '其他条目' = 落在别的条目上 ⇒ 在库内样本上是「多放行的面」，")
    print("    会不会成为误报，取决于它是否同文件（同文件撞车性质较轻，跨文件才是纯误报）。")
    args.json_out.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已落盘: %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
