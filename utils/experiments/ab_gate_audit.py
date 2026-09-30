# -*- coding: utf-8 -*-
"""门控改动的 A/B 审计：不只看总数，要看**每一条**判定是怎么变的。

## 为什么不能只看"晋升数"

门控一改，"晋升数变了多少"是最容易看、也最容易骗人的指标：它不告诉你
**新晋升的那些是不是对的**。本次改动（《01》问题 4/5/6/7）恰恰有一个明确的正确性方向：

* `error_code_clone`（错误代码克隆）只在"知识条目与受检文件**同一个文件**"时才有意义
  —— 所以新增的晋升候选应当**集中在同文件**上；
* 如果新晋升里出现大量**异文件**候选，说明改动等于把跨文件闸门偷偷打开了，
  必须立刻停下（用户已明确决定**暂不开放**跨文件闸门）。

于是本脚本除了汇总，还**逐条列出新晋升的候选并标注它是不是同文件**，把方向问题摆到台面上。

## 用法（MAS 根目录 / GPU 服务器）

    python utils/experiments/ab_gate_audit.py A_runs.txt B_runs.txt A标签 B标签
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.kb_coverage import normalize_key  # noqa: E402

ADMITTED = {"formal_hit", "explanatory_hit"}


def kb_files(db: Path) -> dict:
    import sqlite3
    con = sqlite3.connect(str(db))
    out = {str(i): fp or "" for i, fp in
           con.execute("select id, file_pattern from issue_patterns")}
    con.close()
    return out


def load_run_ids(path: Path) -> set:
    """run id 文件 → **裸 uuid 集合**。

    文件里写的是 `CVE-x/<uuid>`，而遍历报告目录时拿到的是 `<uuid>`（路径第 4 段）。
    第一版直接用整行去比，结果两组都匹配到 0 条候选、审计表全是 0 —— 看起来像"没有变化"，
    其实是一个字符串格式没对齐。故这里统一剥掉前缀。
    """
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.add(line.split("/", 1)[1] if "/" in line else line)
    return out


def collect(run_ids: set, files_by_id: dict, ci_to_pattern: dict = None) -> dict:
    """把某个 run 集合的候选全抓出来，并逐条标注"是不是同一个文件"。

    **同文件怎么判**：优先用候选自己记下来的 `file_pattern`（那是知识条目所属文件，
    与门控运行时 `_same_analysis_target` 用的是同一个字段），和 `_analysis_file` 比末两级路径。
    不要用"sqlite_id → issue_patterns.file_pattern"去查：curated 通道的 id 属于
    `curated_issues` 表，拿它当 issue_patterns 的 id 查会查错行，于是把同文件误判成异文件
    （第一版就是这么得出"94/129 是异文件"的，属于**度量本身的 bug**）。
    """
    ci_to_pattern = ci_to_pattern or {}
    cands = []
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        parts = f.split("/")
        if len(parts) < 4 or parts[3] not in run_ids:
            continue
        cve = parts[2]
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        for blk in ("retrieval_evidence", "gap_retrieval_evidence"):
            for qi, it in enumerate(d.get(blk) or []):
                for ci, c in enumerate(it.get("candidates") or []):
                    sid = str(c.get("sqlite_id") or "")
                    kfp = str(c.get("file_pattern") or "")
                    if not kfp and str(c.get("channel")) != "curated_issue":
                        kfp = files_by_id.get(sid, "")
                    af = str(c.get("_analysis_file") or "")
                    ka = normalize_key(af, 2)
                    kk = normalize_key(kfp, 2)
                    cands.append({
                        "cve": cve, "ch": str(c.get("channel")),
                        "sid": sid, "sid_int": sid,
                        "blk": blk, "qi": qi, "ci": ci,
                        "gd": str(c.get("gating_decision")),
                        "rr": str(c.get("rejection_reason")),
                        "scope": str(c.get("code_fixed_scope") or ""),
                        "mf": list(c.get("matched_fields") or []),
                        "sim": float(c.get("semantic_score") or 0.0),
                        "s": float(c.get("unified_structured_score") or
                                   c.get("structured_score") or 0.0),
                        "same_file": bool(ka and kk and ka == kk),
                        "ana_file": af, "kb_file": kfp,
                    })
    # 逐条对照用的键：**必须包含"被分析文件"和"哪一轮查询"**。
    # 只用 (CVE, 通道, sid) 会塌缩：同一个样本有多个文件、或同一文件被切成多段时，
    # 同一 sid 会出现很多次，字典只留最后一条 —— 于是"哪条判定变了"会被漏掉大半。
    return {"cands": cands,
            "key": {(c["cve"], c["ch"], c["sid"], c["blk"], c["qi"], c["ci"]): c
                    for c in cands}}


def summarize(label: str, data: dict) -> dict:
    c = data["cands"]
    gd = Counter(x["gd"] for x in c)
    rr = Counter(x["rr"] for x in c if x["rr"])
    mf = Counter(m for x in c for m in x["mf"])
    admitted = [x for x in c if x["gd"] in ADMITTED]
    same = [x for x in admitted if x["same_file"]]
    print("\n" + "=" * 92)
    print("### %s" % label)
    print("=" * 92)
    print("  候选 %d 条" % len(c))
    print("  判定分布: %s" % dict(gd))
    print("  晋升 %d 条，其中**同文件** %d 条（%.0f%%），异文件 %d 条" % (
        len(admitted), len(same),
        100 * len(same) / max(1, len(admitted)), len(admitted) - len(same)))
    print("  拒绝理由（全部）:")
    for k, v in rr.most_common():
        print("      %-28s %6d  (%.1f%%)" % (k, v, 100 * v / max(1, len(c))))
    print("  证据字段命中次数:")
    for k, v in mf.most_common(14):
        print("      %-28s %6d" % (k, v))
    clone_ch = Counter(x["ch"] for x in c if "error_code_clone" in x["mf"])
    if clone_ch:
        print("  error_code_clone 命中按通道: %s" % dict(clone_ch))
    sc = Counter(x["scope"] for x in c if x["scope"])
    if sc:
        print("  已修复判据适用范围: %s" % dict(sc))
    return {"n": len(c), "gd": dict(gd), "rr": dict(rr), "mf": dict(mf),
            "admit": len(admitted), "admit_same_file": len(same),
            "admitted": admitted}


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit("用法: ab_gate_audit.py A_runs.txt B_runs.txt A标签 B标签")
    A = load_run_ids(Path(sys.argv[1]))
    B = load_run_ids(Path(sys.argv[2]))
    la, lb = sys.argv[3], sys.argv[4]
    fbi = kb_files(ROOT / "infrastructure/database/mas.db")
    print("A=%s (%d runs)   B=%s (%d runs)   知识库 %d 条" % (la, len(A), lb, len(B), len(fbi)))

    import sqlite3
    _c = sqlite3.connect(str(ROOT / "infrastructure/database/mas.db"))
    ci_to_pattern = {i: p for i, p in _c.execute("select id, pattern_id from curated_issues")}
    _c.close()

    da, db_ = collect(A, fbi, ci_to_pattern), collect(B, fbi, ci_to_pattern)
    ra, rb = summarize(la, da), summarize(lb, db_)

    print("\n" + "=" * 92)
    print("### 逐条对照：判定发生变化的候选")
    print("=" * 92)
    keys = set(da["key"]) | set(db_["key"])
    moved = []
    for k in sorted(keys):
        xa, xb = da["key"].get(k), db_["key"].get(k)
        if xa is None or xb is None:
            moved.append((k, xa, xb, "仅一侧存在"))
            continue
        if xa["gd"] != xb["gd"] or xa["rr"] != xb["rr"]:
            moved.append((k, xa, xb, ""))
    if not moved:
        print("  （没有判定发生变化的候选）")
    new_admits = []
    for k, xa, xb, note in moved:
        a_gd = xa["gd"] if xa else "(缺)"
        b_gd = xb["gd"] if xb else "(缺)"
        a_rr = xa["rr"] if xa else "-"
        b_rr = xb["rr"] if xb else "-"
        flag = ""
        if xb and xb["gd"] in ADMITTED and (not xa or xa["gd"] not in ADMITTED):
            flag = "  ★新晋升 %s" % ("同文件" if xb["same_file"] else "**异文件**")
            new_admits.append(xb)
        print("  %-15s %-9s sid=%-5s  %s/%s → %s/%s%s%s" % (
            k[0], k[1], k[2], a_gd, a_rr or "-", b_gd, b_rr or "-",
            ("  " + note) if note else "", flag))
    print("\n" + "-" * 92)
    print("方向检查（本次改动的正确性判据）")
    print("-" * 92)
    print("  晋升数: %s %d → %s %d  (%+d)" % (
        la, ra["admit"], lb, rb["admit"], rb["admit"] - ra["admit"]))
    print("  其中同文件: %s %d → %s %d" % (
        la, ra["admit_same_file"], lb, rb["admit_same_file"]))
    if new_admits:
        same_n = sum(1 for x in new_admits if x["same_file"])
        print("  新晋升 %d 条：同文件 %d 条，**异文件 %d 条**" % (
            len(new_admits), same_n, len(new_admits) - same_n))
        print("  %-15s %-9s %-6s %-9s %-28s %s" % (
            "CVE", "通道", "sid", "是否同文件", "证据字段", "语义分"))
        for x in new_admits:
            print("  %-15s %-9s %-6s %-9s %-28s %.3f" % (
                x["cve"], x["ch"], x["sid"], "同文件" if x["same_file"] else "异文件",
                ",".join(x["mf"])[:28], x["sim"]))
        if len(new_admits) - same_n:
            print("\n  ⚠️ 出现异文件的新晋升 —— 这可能等于把跨文件闸门打开了，"
                  "必须先判定这些是否可接受，再决定保留哪一项改动。")
    else:
        print("  （没有新晋升）")

    out = ROOT / "reports/ab_gate_audit.json"
    out.write_text(json.dumps({la: {k: v for k, v in ra.items() if k != "admitted"},
                               lb: {k: v for k, v in rb.items() if k != "admitted"},
                               "new_admits": new_admits},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n汇总已写出: %s" % out)


if __name__ == "__main__":
    main()
