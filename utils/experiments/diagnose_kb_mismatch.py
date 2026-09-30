# -*- coding: utf-8 -*-
"""**漂移守卫**：仓库里的知识库，是不是就是线上流水线正在查的那一份？

## 为什么需要它（《03》坑 23 的直接产物）

本项目出现过一次事故：**评测用的知识库和线上跑的知识库不是同一份**（同一个数据集 BigVul 的
两次不同随机划分，CVE 只交集 36 个）。后果是"按 A 库算的分层标签，去评测查 B 库的系统"——
**不报错、不出空值**，每个数字看起来都正常，只是**测错了对象**。

事故之后定的规矩（见《01》问题 0）：**线上正在查的那份库就是唯一基准**。
本脚本用来**随时验证这条规矩有没有被破坏**，以及看清"每个批次各自的标签是按哪份库算的"。

## 用法（MAS 根目录）

    python utils/experiments/diagnose_kb_mismatch.py
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import normalize_key  # noqa: E402

REPO_DB = ROOT / "infrastructure/database/mas.db"
LIVE_SNAPSHOT = ROOT / "reports/mas_live.db"          # 从服务器拉回来的线上库存档
PAPER_BACKUP = ROOT / "reports/mas_kb_paper_seed2024_backup.db"   # 论文那套评测用的旧库


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "(缺)"


def load(db: Path):
    c = sqlite3.connect(str(db))
    rows = [(i, (t or "").strip().upper(), (fp or "")) for i, t, fp in
            c.execute("select id, title, file_pattern from issue_patterns")]
    c.close()
    return {"rows": rows, "cves": {t for _, t, _ in rows if t},
            "key2": {normalize_key(fp, 2) for _, _, fp in rows if fp}}


def main() -> None:
    print("=" * 92)
    print("知识库基准检查")
    print("=" * 92)
    for label, p in (("仓库里的库（评测默认用它）", REPO_DB),
                     ("线上库的存档（从服务器拉回）", LIVE_SNAPSHOT),
                     ("旧库备份（论文那套评测用的划分）", PAPER_BACKUP)):
        if not p.exists():
            print("  %-30s 不存在: %s" % (label, p))
            continue
        kb = load(p)
        print("  %-30s %10d 字节  条目 %3d  CVE %3d  文件 %3d  md5=%s" % (
            label, p.stat().st_size, len(kb["rows"]), len(kb["cves"]),
            len(kb["key2"]), md5(p)[:12]))

    if not REPO_DB.exists() or not LIVE_SNAPSHOT.exists():
        print("\n（缺少可选存档之一，跳过对比）")
        return

    same = md5(REPO_DB) == md5(LIVE_SNAPSHOT)
    print("\n" + "-" * 92)
    if same:
        print("  ✅ 仓库里的库与线上库存档**完全一致**（md5 相同）→ 评测基准对齐")
    else:
        a, b = load(REPO_DB), load(LIVE_SNAPSHOT)
        print("  ❌ 仓库里的库与线上库**不一致** —— 这正是《03》坑 23 的事故形态！")
        print("     CVE 交集 %d 个；文件交集 %d 个" % (
            len(a["cves"] & b["cves"]), len(a["key2"] & b["key2"])))
        print("     ⇒ 用仓库的库算出来的 kb/held 标签，**不能**用来评测线上流水线。")
        print("     处置：若线上那份才是基准，用存档覆盖仓库的库；否则先查清哪份是基准。")
    print("-" * 92)

    # 各批次的标签是按哪份库算的
    kb = load(REPO_DB)
    batches = {
        "冒烟 smoke_kb8（本轮）": ROOT / "utils/experiments/smoke_kb8.json",
        "门控A/B gate_ab16": ROOT / "utils/experiments/gate_ab16.json",
        "冒烟 smoke8（旧，手写）": ROOT / "utils/experiments/smoke8.json",
        "400 实验 manifest": ROOT / "reports/negative_exp_manifest_400_error.json",
    }
    print("\n  各批次的样本，有多少真的在**当前基准库**里：")
    print("  %-26s %6s %10s %10s   %s" % ("批次", "样本数", "在基准库里", "不符", "结论"))
    for name, p in batches.items():
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d.get("rows") or d.get("items") or []
        cves = [str(r.get("cve") or "").strip().upper() for r in rows if r.get("cve")]
        role = {str(r.get("cve") or "").strip().upper(): r.get("role", "") for r in rows}
        if not cves:
            continue
        in_kb = {c for c in cves if c in kb["cves"]}
        # 标签自检：标签说 kb 就应当在库里，反之亦然
        bad = [c for c in cves
               if role.get(c) and ((role[c] == "kb") != (c in kb["cves"]))]
        verdict = "✅ 标签与基准库一致" if not bad else "⚠️ 标签按别的库算的（%d 条不符）" % len(bad)
        print("  %-26s %6d %10d %10d   %s" % (name, len(cves), len(in_kb), len(bad), verdict))
    print("\n  注：'不符'为 0 才说明该批次的 kb/held 标签可以直接用于评测；否则必须用 --db 指定它对应的那份库。")


if __name__ == "__main__":
    main()
