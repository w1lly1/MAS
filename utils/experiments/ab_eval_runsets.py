# -*- coding: utf-8 -*-
"""按 **run 集合** 算召回/误报，并且**分层口径用"线上正在查的那个知识库"来定**。

## 为什么必须显式指定知识库（本轮踩到的坑）

我第一版把批次里的 `role`（kb/held）当成事实用，结果是 7 个"库内"样本在两组里召回都是 0/7。
追下去发现根因不在代码：**线上流水线查的 mas.db，和论文那套评测用的 mas.db，是两份不同的
200 条知识库**（同一数据集的不同随机划分，CVE 只交集 36 个）。批次标签是按本地库算的，
线上库根本不认识那些 CVE —— 于是"召回"这一侧根本没有被测到。

教训：**"这个样本在不在库里"必须按"被测系统实际查询的那个库"来判定**，
不能沿用别处算好的标签。所以本脚本自己调 `utils/eval_strata` 现算分层，
知识库路径必须由 `--db` 明确给出。

## 口径（与 evaluate_400.py 一致）

* 召回（kb 样本）：new_findings 里存在 `evidence.sqlite_id == 该 CVE 在**这个库**里的条目 id`
  （curated 通道的 id 先经 `curated_issues.pattern_id` 映射）
* 误报（held 样本）：new_findings 条数 > 0

## 用法（MAS 根目录 / GPU 服务器）

    python utils/experiments/ab_eval_runsets.py A_runs.txt B_runs.txt --db infrastructure/database/mas.db \
        --labels g0_基线 g1_已修
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.eval_strata import (  # noqa: E402
    G_HELD_PURE, G_HELD_SAME, G_KB_SELF, G_KB_SHARED, SUBCLASS_LABEL, build_strata,
)
from utils.kb_coverage import normalize_key  # noqa: E402


def load_run_map(path: Path) -> dict:
    """run id 文件 → {CVE: run_id}；同时容忍"只有 uuid"的写法。"""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "/" in line:
            cve, run = line.split("/", 1)
            out[cve.strip().upper()] = run.strip()
        else:
            out.setdefault("__bare__", []).append(line)
    return out


def collect_findings(cve: str, run: str) -> list:
    d = ROOT / "reports/analysis" / cve / run / "second_pass/consolidated"
    if not d.exists():
        return None
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for nf in j.get("new_findings", []):
            ev = nf.get("evidence") or {}
            out.append({"channel": ev.get("channel"), "sqlite_id": ev.get("sqlite_id"),
                        "file": ev.get("file_pattern") or "",
                        "matched_fields": (ev.get("matched_fields")
                                           or nf.get("matched_fields") or []),
                        "gating_decision": nf.get("gating_decision")
                                           or ev.get("gating_decision") or ""})
    return out


def owner_of(f, own_ip, ci_to_pattern):
    """"这条命中属于哪条知识条目"——curated 通道的 id 要先映射。"""
    if f["channel"] == "curated_issue":
        return ci_to_pattern.get(f["sqlite_id"])
    return f["sqlite_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a_runs", type=Path)
    ap.add_argument("b_runs", type=Path)
    ap.add_argument("--db", type=Path, required=True,
                    help="**线上正在查的那个**知识库（不是别处算标签用的那份）")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--dataset-root", type=Path,
                    default=ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured")
    args = ap.parse_args()

    ma, mb = load_run_map(args.a_runs), load_run_map(args.b_runs)
    cves = sorted((set(ma) | set(mb)) - {"__bare__"})
    strata = build_strata(cves, args.db, args.dataset_root)

    con = sqlite3.connect(str(args.db))
    id_by_title = {(t or "").strip().upper(): i
                   for i, t in con.execute("select id, title from issue_patterns")}
    ci_to_pattern = {i: p for i, p in con.execute("select id, pattern_id from curated_issues")}
    kb_file = {str(i): (fp or "") for i, fp in con.execute("select id, file_pattern from issue_patterns")}
    con.close()

    def expected_role(cve):
        g = strata.get(cve, {}).get("group", "")
        return "kb" if g in (G_KB_SELF, G_KB_SHARED) else "held"

    print("=" * 100)
    print("按 run 集合评测   （分层按 --db 指定的知识库现算：%s）" % args.db)
    print("=" * 100)
    print("  %-16s %-16s %-6s %s" % ("CVE", "分层", "role", "说明"))
    for cve in cves:
        st = strata.get(cve, {})
        print("  %-16s %-16s %-6s %s" % (
            cve, st.get("group", "?"), expected_role(cve),
            ",".join(st.get("kb_siblings") and [s["cve"] for s in st["kb_siblings"]] or []) or
            (st.get("subclass") and SUBCLASS_LABEL.get(st["subclass"], "")) or ""))

    arms = [(args.labels[0], ma), (args.labels[1], mb)]
    res = {}
    for label, rmap in arms:
        rows = []
        for cve in cves:
            r = rmap.get(cve)
            fs = collect_findings(cve, r) if r else None
            rows.append({"cve": cve, "group": strata.get(cve, {}).get("group", "?"),
                         "role": expected_role(cve), "n": 0 if fs is None else len(fs),
                         "missing": fs is None, "findings": fs or []})
        res[label] = rows

    print("\n" + "=" * 100)
    print("指标（**逐层给**，不再一刀切）")
    print("=" * 100)
    for label, _ in arms:
        print("\n  【%s】" % label)
        for group, metric, title in (
            (G_KB_SELF, "captured", "召回率 kb-self     "),
            (G_KB_SHARED, "captured", "召回率 kb-shared   "),
            (G_HELD_PURE, "fp", "误报率 held-pure    "),
            (G_HELD_SAME, "fp", "误报率 held-samefile"),
        ):
            grp = [x for x in res[label] if x["group"] == group and not x["missing"]]
            if not grp:
                continue
            if metric == "captured":
                hit = [x for x in grp if any(
                    owner_of(f, id_by_title.get(x["cve"]), ci_to_pattern) == id_by_title.get(x["cve"])
                    for f in x["findings"])]
            else:
                hit = [x for x in grp if x["n"] > 0]
            print("    %s %2d/%2d = %-6s   %s" % (
                title, len(hit), len(grp),
                ("%.1f%%" % (100 * len(hit) / len(grp))) if grp else "-",
                [x["cve"] for x in hit]))

    print("\n" + "=" * 100)
    print("逐样本对照")
    print("=" * 100)
    a_map = {x["cve"]: x for x in res[args.labels[0]]}
    b_map = {x["cve"]: x for x in res[args.labels[1]]}
    print("  %-16s %-16s %-22s %-22s %s" % ("CVE", "分层", args.labels[0], args.labels[1], "变化"))
    for cve in cves:
        xa, xb = a_map.get(cve), b_map.get(cve)
        if xa is None or xb is None or xa["missing"] or xb["missing"]:
            print("  %-16s %-16s %-22s %-22s 缺产物，跳过" % (
                cve, xa["group"] if xa else "?", "-", "-"))
            continue
        own = id_by_title.get(cve)
        if xa["role"] == "kb":
            ca = any(owner_of(f, own, ci_to_pattern) == own for f in xa["findings"])
            cb = any(owner_of(f, own, ci_to_pattern) == own for f in xb["findings"])
            sa = "召回" if ca else ("命中别的(%d)" % xa["n"] if xa["n"] else "无命中")
            sb = "召回" if cb else ("命中别的(%d)" % xb["n"] if xb["n"] else "无命中")
            chg = "" if ca == cb else ("★ 召回成功" if cb else "⚠ 召回丢失")
            print("  %-16s %-16s %-22s %-22s %s" % (cve, xa["group"], sa, sb, chg))
        else:
            sa = "误报(%d)" % xa["n"] if xa["n"] else "干净"
            sb = "误报(%d)" % xb["n"] if xb["n"] else "干净"
            chg = "" if (xa["n"] > 0) == (xb["n"] > 0) else ("⚠ 新增误报" if xb["n"] else "★ 误报消除")
            print("  %-16s %-16s %-22s %-22s %s" % (cve, xa["group"], sa, sb, chg))

    print("\n" + "=" * 100)
    print("新出现的命中明细（判断新增的是不是『同一个文件』）")
    print("=" * 100)
    shown = False
    for cve in cves:
        xb, xa = b_map.get(cve), a_map.get(cve)
        if xb is None or xb["missing"] or (xa and xa["missing"]):
            continue
        old = {f["sqlite_id"] for f in (xa["findings"] if xa else [])}
        for f in xb["findings"]:
            if f["sqlite_id"] in old:
                continue
            own = id_by_title.get(cve)
            o = owner_of(f, own, ci_to_pattern)
            kf = kb_file.get(str(o), "")
            mykey = (strata.get(cve, {}).get("file_keys") or [""])[0]
            print("  %-16s %-14s sid=%-6s→条目%-5s 同文件=%s  是本人条目=%s  证据=%s" % (
                cve, str(f["channel"]), str(f["sqlite_id"]), str(o),
                "是" if (kf and normalize_key(kf, 2) == mykey) else "否",
                "是" if o == own else "否",
                ",".join(map(str, f["matched_fields"]))[:36]))
            shown = True
    if not shown:
        print("  （没有新出现的命中）")


if __name__ == "__main__":
    main()
