# -*- coding: utf-8 -*-
"""造一个小体量、**对着线上知识库**的冒烟批次：优先挑"源文件小"的库内样本。

## 为什么这样挑

* **必须是线上库里的样本**：只有"该 CVE 自己的条目在**被测系统实际查询的那个库**里"，
  "召回"这一侧才测得出来。用别的库算出来的标签会把整套结论测错对象（见《03》坑 23）。
* **优先小文件**：本项目的单样本产物没有上限，一个 `sqlite3.h` amalgamation 样本能产出 7.9 GB；
  冒烟要的是"30 分钟内跑完并看得出问题"，所以按源文件大小从小到大挑。

## 用法

    python utils/experiments/make_kb_smoke_batch.py --db reports/mas_live.db --n 8 \
        --out utils/experiments/smoke_kb8.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT  # noqa: E402

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
SERVER_DS = "/root/autodl-tmp/MAS/tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_live.db",
                    help="**线上正在查的那个**知识库（不是别处的）")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "utils/experiments/smoke_kb8.json")
    ap.add_argument("--max-kb", type=int, default=400, help="源文件大小上限（KB）")
    ap.add_argument("--min-kb", type=int, default=4, help="源文件大小下限（KB），太小的样本没内容可分析")
    args = ap.parse_args()

    con = sqlite3.connect(str(args.db))
    kb = [(i, (t or "").strip().upper(), (fp or "")) for i, t, fp in
          con.execute("select id, title, file_pattern from issue_patterns")]
    con.close()
    if not kb:
        raise SystemExit("知识库为空：%s" % args.db)

    # 统计每个库内 CVE 的源文件大小
    cands = []
    for _id, cve, fp in kb:
        d = DS / "before" / cve
        if not d.exists():
            continue
        files = [f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in SOURCE_EXT]
        if not files:
            continue
        total = sum(f.stat().st_size for f in files)
        cands.append({"cve": cve, "kb_id": _id, "kb_file": fp, "bytes": total,
                      "files": [f.name for f in files]})

    lo, hi = args.min_kb * 1024, args.max_kb * 1024
    ok = [c for c in cands if lo <= c["bytes"] <= hi]
    ok.sort(key=lambda c: c["bytes"])
    picked = ok[: args.n]
    if len(picked) < args.n:
        print("[警告] 只有 %d 个样本落在 %d-%d KB 区间，少于要求的 %d 个"
              % (len(ok), args.min_kb, args.max_kb, args.n))

    items = []
    for c in picked:
        items.append({
            "role": "kb",
            "cve": c["cve"],
            "target_dir": "%s/before/%s" % (SERVER_DS, c["cve"]),
            "output_dir": c["cve"],
            "kb_entry_id": c["kb_id"],
            "kb_file_pattern": c["kb_file"],
            "src_bytes": c["bytes"],
        })

    cfg = {
        "description": ("线上知识库冒烟：%d 个『库内』样本（各自条目在线上库里），"
                        "按源文件从小到大挑，目标是 30 分钟内跑完并能看出问题" % len(items)),
        "kb": str(args.db),
        "why_kb": ("分层/标签必须按『被测系统实际查询的那个库』算；"
                   "用别的库算出来的标签会把召回测成 0（见《03》坑 23）"),
        "selection": {"min_kb": args.min_kb, "max_kb": args.max_kb,
                      "sort": "源文件总字节升序"},
        "items": items,
    }
    args.out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("批次已写出: %s" % args.out)
    print("  线上库里共 %d 个条目，本地数据集里能找到源码的 %d 个，落在尺寸区间的 %d 个"
          % (len(kb), len(cands), len(ok)))
    print("  %-16s %9s  %-34s %s" % ("CVE", "源文件", "知识库记录的文件", "样本文件"))
    for c in picked:
        print("  %-16s %7.1f KB  %-34s %s" % (
            c["cve"], c["bytes"] / 1024, c["kb_file"][:34], c["files"][0][:38]))
    print("\n  预期：这 %d 个样本的**自己的条目**都在库里 → 是干净的召回样本（kb-self 量级）"
          % len(items))


if __name__ == "__main__":
    main()
