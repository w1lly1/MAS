#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""紧急腾空间：给**已经写完的** run 做大证据瘦身（服务器上跑）。

## 为什么需要"紧急版"
`trim_run_evidence.py` 是按 run 清单来的，而清单要等**整批跑完**才生成。
于是批次跑了 2 小时、15 个样本写了几十 G 证据，磁盘冲到 98%，而清单还没出现
—— 最后一个样本随时可能因为写不进去而失败。

## 安全边界（三条）
1. **只碰"安静"的文件**：mtime 早于 `--min-age-min` 分钟的才处理（默认 8 分钟），
   正在写的那一个样本不会被碰；
2. **只看大文件**（>20MB）——小文件留着不折腾，也降低风险；
3. 复用 `trim_run_evidence.process()`（**一条逻辑一个实现**）：原文先 gzip 留档并校验，
   再写精简版，并核对"判定相关字段"前后完全一致；任何不一致就停下报错。

用法:
    venv/bin/python -X utf8 srv_emergency_trim.py --free-target-mb 8000
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/root/autodl-tmp/MAS")
sys.path.insert(0, str(REPO / "local_libs"))
sys.path.insert(0, str(REPO))

from utils.experiments.trim_run_evidence import PATTERNS, process  # noqa: E402


def free_mb(path: str = "/root/autodl-tmp") -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize // (1 << 20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-age-min", type=int, default=8)
    ap.add_argument("--min-size-mb", type=int, default=20)
    ap.add_argument("--free-target-mb", type=int, default=8000)
    ap.add_argument("--max-str", type=int, default=20000)
    args = ap.parse_args()

    analysis = REPO / "reports/analysis"
    cutoff = time.time() - args.min_age_min * 60
    print("起始可用空间: %d MB，目标: %d MB" % (free_mb(), args.free_target_mb))

    # 收集候选：大文件 + 够老
    cands = []
    for run_dir in sorted(analysis.glob("*/*")):
        if not run_dir.is_dir():
            continue
        for pattern in PATTERNS:
            for f in run_dir.glob(pattern):
                if f.suffix == ".gz" or not f.is_file():
                    continue
                st = f.stat()
                if st.st_size < args.min_size_mb * (1 << 20):
                    continue
                if st.st_mtime > cutoff:
                    continue          # 太新：可能正在写
                cands.append((st.st_size, f))
    cands.sort(reverse=True)
    total_mb = sum(s for s, _ in cands) / 1e6
    print("候选大文件: %d 个，合计 %.0f MB（只处理 mtime 早于 %d 分钟前的）"
          % (len(cands), total_mb, args.min_age_min))

    done = 0
    for size, f in cands:
        if free_mb() >= args.free_target_mb:
            print("已达到目标空间，停下")
            break
        try:
            b, a, ok = process(f, args.max_str, apply=True)
        except Exception as exc:  # noqa: BLE001
            print("  跳过 %s（%s: %s）" % (f.name[:60], type(exc).__name__, exc))
            continue
        done += 1
        print("  [%d] %-58s %7.1fMB -> %6.1fMB 一致=%s  可用=%dMB"
              % (done, f.name[:58], b / 1e6, a / 1e6, ok, free_mb()))
        if not ok:
            raise SystemExit("*** 判定字段不一致，停止（不要继续动数据）")
    print("处理 %d 个文件，最终可用 %d MB" % (done, free_mb()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
