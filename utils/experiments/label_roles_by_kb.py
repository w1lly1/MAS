# -*- coding: utf-8 -*-
"""按"文件是否真的在知识库里"自动改写批次配置里的 role（kb / held）标签。

用法（MAS 根目录）：
    python utils/experiments/label_roles_by_kb.py --batch utils/experiments/smoke8.json
    python utils/experiments/label_roles_by_kb.py --batch 论文/test_400_batch.json --dry-run

行为：
  · 对每个 item，用它的 target_dir 去知识库的 file_pattern 里查（归一化文件名后比较）
  · 查到 → role 应为 kb；查不到 → role 应为 held
  · 改写时保留原值到 `role_declared`，并写入 `role_source`，便于审计
  · 默认先 dry-run 打印对照表，确认后再加 --write 真正落盘
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import (  # noqa: E402
    cve_in_kb, explain, load_kb_index, sample_in_kb,
)

DEFAULT_DB = ROOT / "infrastructure/database/mas.db"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True, help="批次配置 JSON")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="知识库 SQLite")
    ap.add_argument("--criterion", choices=["cve", "file"], default="cve",
                    help="cve=该样本自身的条目是否入库（权威，决定 kb/held）；"
                         "file=库里是否存在同路径文件（仅用于标注混淆风险）")
    ap.add_argument("--mode", choices=["relpath2", "relpath3", "basename", "project"],
                    default="relpath2",
                    help="--criterion file 时的路径比较粒度")
    ap.add_argument("--write", action="store_true", help="真正写回文件（默认只预览）")
    ap.add_argument("--report", type=Path, default=None, help="把逐项明细写成 JSON")
    args = ap.parse_args()

    kb = load_kb_index(args.db)
    print("知识库: %s" % args.db)
    print("  库内 CVE 条目 %d 个（权威依据：title 列）" % len(kb["cves"]))
    print("  库内出现过的文件路径：末级 %d / 末两级 %d / 项目 %d 个"
          % (len(kb["key1"]), len(kb["key2"]), len(kb["projects"])))
    print("  判定依据: %s" % args.criterion)

    cfg = json.loads(args.batch.read_text(encoding="utf-8"))
    items = cfg.get("items") or cfg.get("rows") or []
    if not items:
        raise SystemExit("批次里没有 items/rows")

    rows, changed, stat = [], 0, {"kb": 0, "held": 0}
    print("\n%-4s %-16s %-26s %-7s %-7s %-9s %s" % (
        "#", "CVE", "文件", "原标签", "判定", "库有同文件", "依据"))
    for i, it in enumerate(items, 1):
        td = it.get("target_dir") or (ROOT / str(it.get("before") or ""))
        info = explain(td, kb)
        by_cve = cve_in_kb(it.get("cve"), kb)
        by_file = sample_in_kb(td, kb, mode=args.mode)
        auto = ("kb" if by_cve else "held") if args.criterion == "cve" else ("kb" if by_file else "held")
        old = str(it.get("role") or "")
        stat[auto] += 1
        if old != auto:
            changed += 1
        fname = (info["files"][0] if info["files"] else "(无源文件)")
        if args.criterion == "cve":
            why = "该条目在库内" if by_cve else "该条目不在库内"
        else:
            why = ("命中 %s" % info["hit_relpaths"][:2]) if info["hit_relpaths"] else "库中无同路径文件"
        rows.append({"cve": it.get("cve"), "target_dir": str(td), "role_declared": old,
                     "role_auto": auto, "cve_in_kb": by_cve, "file_in_kb": by_file,
                     "hit_relpaths": info["hit_relpaths"], "hit_projects": info["hit_projects"]})
        print("%-4d %-16s %-26s %-7s %-7s %-9s %s" % (
            i, str(it.get("cve"))[:16], fname[:26], old or "-", auto,
            "是" if by_file else "否", why))

    # 混淆标注：标签说 held，但库里存在同路径文件 → 命中不一定是误报
    confound = [r for r in rows if r["role_auto"] == "held" and r["file_in_kb"]]
    print("\n判定汇总: kb %d 条，held %d 条；与原标签不一致 %d 条" % (
        stat["kb"], stat["held"], changed))
    print("【混淆风险】判定为 held、但库中存在同路径文件的样本: %d 条" % len(confound))
    if confound:
        print("   这些样本即使被检索命中，也可能是撞上了库里**别的 CVE** 的同名文件，"
              "不能直接计为误报。样例:", [r["cve"] for r in confound[:6]])
    if changed == 0:
        print("[OK] 标签与知识库实际内容一致，无需改动")
    else:
        print("[警告] 有 %d 条标签与事实不符（会造成评测泄漏）" % changed)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"batch": str(args.batch), "db": str(args.db), "criterion": args.criterion,
             "mode": args.mode,
             "summary": {**stat, "changed": changed, "held_but_same_file_in_kb": len(confound)},
             "items": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("明细已写出:", args.report)

    if args.write:
        for it, r in zip(items, rows):
            if it.get("role") != r["role_auto"]:
                it.setdefault("role_declared", it.get("role"))
            it["role"] = r["role_auto"]
            it["role_source"] = "auto_by_kb_coverage@%s" % args.criterion
            it["kb_has_same_file"] = bool(r["file_in_kb"])
        cfg["role_labeling"] = {
            "method": "auto_by_kb_coverage",
            "criterion": args.criterion,
            "db": str(args.db.relative_to(ROOT)) if str(args.db).startswith(str(ROOT)) else str(args.db),
            "changed": changed,
            "held_but_same_file_in_kb": len(confound),
        }
        args.batch.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n已写回:", args.batch, "（原标签保留在 role_declared 字段）")
    else:
        print("\n（预览模式，未改动文件。确认无误后加 --write）")


if __name__ == "__main__":
    main()
