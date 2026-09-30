# -*- coding: utf-8 -*-
"""诊断：**正在跑的知识库** 与 **论文那套评测用的知识库** 是不是同一个。

## 为什么必须问这个问题

本轮 A/B 里，7 个"库内(kb)"样本在两组里召回都是 **0/7**。追下去发现根因不是代码，
而是：**批次的分层是按本地 mas.db 算的，而线上流水线查的是服务器上的 mas.db，两者是不同的 200 条知识库。**

这个脚本把事实摆清楚：两份知识库各自的 CVE 集合、文件集合，以及与
① 400 实验的批次标签 ② 冒烟批次(smoke8) ③ 本轮 A/B 批次 的重叠情况。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import normalize_key  # noqa: E402


def load(db: Path):
    c = sqlite3.connect(str(db))
    rows = [(i, (t or "").strip(), (fp or "")) for i, t, fp in
            c.execute("select id, title, file_pattern from issue_patterns")]
    c.close()
    return {"rows": rows, "cves": {t for _, t, _ in rows if t},
            "key2": {normalize_key(fp, 2) for _, _, fp in rows if fp}}


def mnf(p: Path):
    if not p.exists():
        return set()
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d.get("rows") or d.get("items") or []
    return {str(r.get("cve") or "").strip().upper() for r in rows if r.get("cve")}


def main() -> None:
    local = load(ROOT / "infrastructure/database/mas.db")
    live = load(ROOT / "reports/mas_live.db")
    print("=" * 92)
    print("两份知识库")
    print("=" * 92)
    for name, kb in (("本地 mas.db（论文那套评测用的）", local),
                     ("服务器 mas.db（线上流水线正在查的）", live)):
        print("  %-34s 条目 %d  CVE %d  文件(末两级) %d" % (
            name, len(kb["rows"]), len(kb["cves"]), len(kb["key2"])))
    print("\n  CVE 交集: %d 个 %s" % (len(local["cves"] & live["cves"]),
                                     sorted(local["cves"] & live["cves"])[:6]))
    print("  文件交集: %d 个" % len(local["key2"] & live["key2"]))

    sets = {
        "400实验 kb 组": mnf(ROOT / "reports/negative_exp_manifest_400_error.json"),
        "smoke8": mnf(ROOT / "utils/experiments/smoke8.json"),
        "本轮 A/B(16)": mnf(ROOT / "utils/experiments/gate_ab16.json"),
    }
    # 400 的 held 组
    m = json.loads((ROOT / "reports/negative_exp_manifest_400_error.json").read_text(encoding="utf-8"))
    sets["400实验 held 组"] = {str(r["cve"]).strip().upper() for r in m["rows"]
                              if r.get("role") == "held"}

    print("\n" + "=" * 92)
    print("这些批次里的 CVE，各自有多少真的在**线上知识库**里")
    print("=" * 92)
    print("  %-18s %6s %14s %14s" % ("批次", "样本数", "在本地库里", "在线上库里"))
    for name, cves in sets.items():
        print("  %-18s %6d %14d %14d" % (name, len(cves),
                                         len(cves & local["cves"]), len(cves & live["cves"])))

    print("\n" + "=" * 92)
    print("结论")
    print("=" * 92)
    inter = sets["400实验 kb 组"] & live["cves"]
    if not inter:
        print("  ⚠️ 400 实验的 200 个『库内』样本，**没有任何一个**在线上知识库里。")
        print("     也就是说：现在这条流水线的知识库，与论文那套评测用的知识库，是两套东西。")
    else:
        print("  400 实验 kb 组在线上库里的命中: %d 个" % len(inter))
    sm = sets["smoke8"] & live["cves"]
    print("  smoke8（之前 C2/C3 的 A/B 批次）在线上库里的命中: %d 个 %s" % (len(sm), sorted(sm)))
    smf = {normalize_key(fp, 2) for fp in
           [r["target_dir"] for r in json.loads(
               (ROOT / "utils/experiments/smoke8.json").read_text(encoding="utf-8"))["items"]]}
    print("  smoke8 的源文件路径在线上库文件表里的命中: %d 个" % len(smf & live["key2"]))
    ab = sets["本轮 A/B(16)"]
    print("  本轮 A/B 的 CVE 在线上库里的命中: %d 个 %s" % (len(ab & live["cves"]), sorted(ab & live["cves"])))


if __name__ == "__main__":
    main()
