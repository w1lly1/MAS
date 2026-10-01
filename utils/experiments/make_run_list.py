# -*- coding: utf-8 -*-
"""从批次日志里收集"每个样本最终用哪个 run"，输出评测工具要的 `CVE/run_id` 列表。

## 为什么需要它

一个臂可能由**多次运行**组成（首次 + 补跑），而评测要的是"每个样本的**最终**那次 run"。
日志里每个条目两行：`[n/N] 📂 <样本目录>` 然后 `🆔 Run ID: <uuid>`。
后出现的（补跑）自然覆盖先出现的（失败那次）—— 顺序读、直接覆盖即可。

## 用法

    python utils/experiments/make_run_list.py --out arm1_runs.txt \
        --logs arm1_new_new.log arm1_retry.log
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ITEM_RE = re.compile(r"^\s*\[(\d+)/(\d+)\]\s+\S+\s+(.+?)\s*$")
RUN_RE = re.compile(r"Run ID:\s*([0-9a-fA-F-]{8,})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--logs", nargs="+", type=Path, required=True)
    ap.add_argument("--only-complete", action="store_true",
                    help="只保留在该日志里判定为『完成』的条目（默认全部保留）")
    args = ap.parse_args()

    mapping = {}          # CVE -> run_id（顺序覆盖）
    order = []            # 保持样本顺序
    stats = {}
    for log in args.logs:
        if not log.exists():
            print("跳过（不存在）:", log)
            continue
        current_cve = None
        pending_run = None
        pending_partial = False
        ok = partial = 0
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = ITEM_RE.match(line)
            if m:
                current_cve = m.group(3).strip().split("/")[-1]
                pending_run = None
                pending_partial = False
                continue
            r = RUN_RE.search(line)
            if r and current_cve:
                pending_run = r.group(1)
                continue
            if "未确认完成" in line and current_cve:
                pending_partial = True
            if "✅ 完成" in line and current_cve and pending_run:
                mapping[current_cve] = pending_run
                if current_cve not in order:
                    order.append(current_cve)
                ok += 1
                current_cve = None
            elif pending_partial and current_cve and pending_run:
                # partial 也记下来，但可以被后面日志里的成功覆盖
                mapping.setdefault(current_cve, pending_run)
                if current_cve not in order:
                    order.append(current_cve)
                partial += 1
                current_cve = None
        stats[log.name] = (ok, partial)
        print("%s：完成 %d，未确认完成 %d" % (log.name, ok, partial))

    lines = ["%s/%s" % (cve, mapping[cve]) for cve in order if cve in mapping]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n写出 %d 条 -> %s" % (len(lines), args.out))
    for ln in lines[:5]:
        print("   ", ln)
    if len(lines) > 5:
        print("    ...")


if __name__ == "__main__":
    main()
