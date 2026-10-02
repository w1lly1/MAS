#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量一个批次的真实速率并估算剩余时间（在服务器上跑，只读日志）。

为什么单独写成工具：这个数被反复问（"还要多久"），而**凭感觉估**会差很多——
held 样本实测 3.5~10 分钟/个，取决于该样本捞到多少外来候选。

用法: venv/bin/python srv_batch_eta.py /root/autodl-tmp/held_overlap15_log.txt 15
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
total = int(sys.argv[2]) if len(sys.argv) > 2 else None

ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
mk_re = re.compile(r"^\s*\[(\d+)/(\d+)\]\s")

marks: list[tuple[int, int, str]] = []
last_ts = None
last_line_ts = None
for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
    m = ts_re.match(line)
    if m:
        last_ts = m.group(1)
        last_line_ts = m.group(1)
    mk = mk_re.match(line)
    if mk and last_ts:
        marks.append((int(mk.group(1)), int(mk.group(2)), last_ts))

if not marks:
    print("日志里还没有样本标记")
    raise SystemExit(0)

print("批次: %s" % path.name)
print("样本标记数: %d" % len(marks))
prev = None
for i, tot, t in marks:
    d = ""
    if prev:
        dt = (datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
              - datetime.strptime(prev, "%Y-%m-%d %H:%M:%S")).total_seconds()
        d = "   （距上一个 %d 分 %d 秒）" % (dt // 60, dt % 60)
    print("  [%d/%d] 开始于 %s%s" % (i, tot, t, d))
    prev = t

start = datetime.strptime(marks[0][2], "%Y-%m-%d %H:%M:%S")
latest = datetime.strptime(last_line_ts, "%Y-%m-%d %H:%M:%S")
started = len(marks)                      # 已经开始了几个样本
finished = max(0, started - 1)            # 上一个已经开始 → 至少完成这么多个
elapsed = (latest - start).total_seconds()
if finished:
    per = elapsed / finished
    print("\n已完成 %d 个（第 %d 个在跑），平均 **%.1f 分钟/样本**"
          % (finished, started, per / 60))
    remain = (total or marks[0][1]) - finished
    eta = latest.timestamp() + per * remain
    print("剩余 %d 个 → 约 %.0f 分钟，预计 **%s** 完成"
          % (remain, per * remain / 60,
             datetime.fromtimestamp(eta).strftime("%H:%M")))
else:
    print("\n还没跑完第一个样本（已耗时 %.1f 分钟）" % (elapsed / 60))
print("日志最后写入: %s（现在 %s）"
      % (last_line_ts, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
