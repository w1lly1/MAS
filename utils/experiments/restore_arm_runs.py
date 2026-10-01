# -*- coding: utf-8 -*-
"""把服务器上打到包里的"臂数据"在本机还原成评测脚本需要的目录结构。

## 为什么需要

评测脚本（`ab_eval_runsets.py`）按 `reports/analysis/<CVE>/<run_id>/second_pass/consolidated/*.json`
去读产物。而服务器上打包时用的是**扁平暂存**（`<arm>/runs/<CVE>/<run_id>/*.json`），
以免把 GB 级的 fullLayer 一起背回来。所以在本机要按评测要求的层级放回去。

**注意**：只还原**小文件**（r2/run_summary/debug jsonl），不还原 fullLayer ——
评测不需要它们；要复盘大产物就回服务器（那些 run 目录还在）。

## 用法

    python utils/experiments/restore_arm_runs.py --tarball reports/artifacts_arm1_x.tgz \
        --runs-out reports/arm1_runs.txt
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", type=Path, required=True)
    ap.add_argument("--into", type=Path, default=ROOT / "reports/analysis")
    ap.add_argument("--runs-out", type=Path, default=None, help="把包里的 runs.txt 复制到这个路径")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    placed = 0
    runs_member = None
    with tarfile.open(args.tarball, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            parts = Path(m.name).parts
            # 期望形状: <arm>/runs/<CVE>/<run_id>/<file>
            if "runs" in parts:
                i = parts.index("runs")
                if len(parts) >= i + 4:
                    cve, run, name = parts[i + 1], parts[i + 2], parts[i + 3]
                    dest = args.into / cve / run / "second_pass" / "consolidated" / name
                    if name == "run_summary.json":
                        dest = args.into / cve / run / name
                    elif name.endswith(".jsonl"):
                        dest = args.into / cve / run / "debug" / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(m) as src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    placed += 1
                    if args.verbose:
                        print("  ", dest.relative_to(ROOT))
                    continue
            if parts[-1] == "runs.txt":
                runs_member = m

        if runs_member is not None and args.runs_out is not None:
            args.runs_out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(runs_member) as src, args.runs_out.open("wb") as out:
                shutil.copyfileobj(src, out)
            n = len([ln for ln in args.runs_out.read_text(encoding="utf-8").splitlines() if ln.strip()])
            print("run 列表: %s（%d 条）" % (args.runs_out, n))

    print("已还原文件: %d 个 → %s" % (placed, args.into))


if __name__ == "__main__":
    main()
