# -*- coding: utf-8 -*-
"""三臂对照：把"检索段"和"门控段"分开量，回答"为什么最终召回没变"。

## 为什么不能只看最终召回

最终召回（自己的条目有没有被放进 new_findings）是**检索 × 门控**两段的乘积。
两臂最终数字一样，可能是"检索变了但门控拦回来"，也可能是"检索根本没变"。
所以这里逐臂各量四个量：

  A. **候选总量**（所有证据里 candidates 的条数之和）—— 检索规模有没有变
  B. **检索段召回**：自己的条目 id 有没有出现在**候选**里（不管门控放不放行）
  C. **门控段放行**：自己的条目有没有进 new_findings
  D. **放行的构成**：同文件 / 跨文件；以及非自己条目的放行条数（误报面）

用法：
    python utils/experiments/compare_arms.py --arms "基线=reports/arm2_runs.txt" \
        "新系统=reports/arm1_runs.txt" "消融=reports/arm3_runs.txt" --db reports/mas_live.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# **复用** ab_eval_runsets 的 owner_of：curated 通道的 sqlite_id 是 curated_issues 的主键，
# 必须先映射到 issue_patterns.id 才能和"自己的条目"比。
# 第一版我在这里直接比 sqlite_id，结果把 26 条"自己的条目"全算成别人（1/30 vs 26/30 严重不一致），
# 靠"两个独立统计对不上就怀疑度量"这条纪律才发现（《03》坑 24）。
from utils.experiments.ab_eval_runsets import owner_of  # noqa: E402


def norm_key(path: str, n: int = 2) -> str:
    p = str(path or "").strip().replace("\\", "/").split("/")
    return "/".join(x.lower() for x in p[-n:]) if p else ""


def load_arm(runs_file: Path, id_by_title: dict, ci_to_pattern: dict) -> dict:
    samples = [ln.strip() for ln in runs_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = {}
    for line in samples:
        cve, run = line.split("/", 1)
        d = ROOT / "reports/analysis" / cve / run / "second_pass" / "consolidated"
        own = id_by_title.get(cve.upper())
        rec = {"cve": cve, "own": own, "cand": 0, "own_in_cand": False,
               "admitted": [], "nb_findings": 0, "files": set()}
        if not d.exists():
            out[cve] = rec
            continue

        def resolve(sid, channel):
            """把命中/候选的 id 解析成"知识条目 id"（curated 通道要过一层映射）。"""
            if sid is None:
                return None
            if str(channel or "") == "curated_issue":
                return ci_to_pattern.get(int(sid))
            return int(sid)

        for f in sorted(d.glob("*_r2.json")):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            rec["files"].add(str(j.get("file") or ""))
            for key in ("retrieval_evidence", "gap_retrieval_evidence"):
                for ev in (j.get(key) or []):
                    cands = ev.get("candidates") or []
                    rec["cand"] += len(cands)
                    for c in cands:
                        if not isinstance(c, dict):
                            continue
                        sid = c.get("sqlite_id")
                        if sid is None:
                            sid = (c.get("evidence") or {}).get("sqlite_id")
                        resolved = resolve(sid, c.get("channel") or c.get("primary_channel"))
                        if resolved is not None and own is not None and resolved == int(own):
                            rec["own_in_cand"] = True
            for nf in (j.get("new_findings") or []):
                ev = nf.get("evidence") or {}
                channel = ev.get("channel") or ev.get("primary_channel")
                rec["nb_findings"] += 1
                rec["admitted"].append({"sqlite_id": resolve(ev.get("sqlite_id"), channel),
                                        "raw_id": ev.get("sqlite_id"),
                                        "channel": channel,
                                        "file": nf.get("file") or ""})
        out[cve] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="形如 名称=路径")
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_live.db")
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    file_by_id = {int(i): (fp or "") for i, fp in
                  con.execute("select id, file_pattern from issue_patterns")}
    ci_to_pattern = {int(i): int(p) for i, p in
                     con.execute("select id, pattern_id from curated_issues")}
    con.close()

    arms = {}
    for item in args.arms:
        name, path = item.split("=", 1)
        arms[name] = load_arm(Path(path), id_by_title, ci_to_pattern)

    labels = list(arms)
    print("=" * 104)
    print("三臂对照（每臂 %d 个样本）" % len(next(iter(arms.values()))))
    print("=" * 104)
    print("  %-10s %10s %14s %14s %10s %12s" %
          ("臂", "候选总量", "检索段捞到自己", "门控段放行自己", "放行总数", "放行非自己"))
    for L in labels:
        d = arms[L]
        cand = sum(r["cand"] for r in d.values())
        inc = sum(1 for r in d.values() if r["own_in_cand"])
        adm = sum(1 for r in d.values()
                  if r["own"] is not None and any(a["sqlite_id"] == r["own"] for a in r["admitted"]))
        tot = sum(r["nb_findings"] for r in d.values())
        other = sum(1 for r in d.values() for a in r["admitted"] if a["sqlite_id"] != r["own"])
        print("  %-10s %10d %14s %14s %10d %12d" %
              (L, cand, "%d/%d" % (inc, len(d)), "%d/%d" % (adm, len(d)), tot, other))

    print()
    print("=" * 104)
    print("逐样本：检索段(候选里有没有自己) / 门控段(有没有放行自己)")
    print("=" * 104)
    print("  %-16s %s" % ("CVE", "  ".join("%-22s" % L for L in labels)))
    for cve in next(iter(arms.values())):
        cells = []
        for L in labels:
            r = arms[L].get(cve) or {}
            cells.append("%-22s" % ("候选%s 放行%s" % ("有" if r.get("own_in_cand") else "无",
                                                      "有" if r.get("own") is not None and any(
                                                          a["sqlite_id"] == r["own"] for a in r.get("admitted", [])) else "无")))
        print("  %-16s %s" % (cve, "  ".join(cells)))

    print()
    print("=" * 104)
    print("放行明细（只看自己条目被放行的样本：走的是哪条通道）")
    print("=" * 104)
    for L in labels:
        d = arms[L]
        chans = {}
        for r in d.values():
            for a in r["admitted"]:
                if a["sqlite_id"] == r["own"]:
                    chans[a["channel"]] = chans.get(a["channel"], 0) + 1
        print("  %-10s %s" % (L, chans or "(无)"))


if __name__ == "__main__":
    main()
