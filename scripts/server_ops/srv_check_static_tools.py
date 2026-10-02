#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证小事 11 在服务器上的效果：外部静态工具**能不能被找到、并且真的跑起来**。

背景（《01》小事 11）：历史实现只用 `shutil.which(tool)` 查**系统 PATH**；
而工具通常装在**当前 Python 所在环境**（venv/bin）里，那个目录未必在 PATH 上
→ 于是"明明装了却报未安装"，外部工具被整体跳过。

本脚本三问：
  1. venv/bin 里到底有没有这些工具（用文件系统直接看，作为事实基线）；
  2. 经过 `_check_tool_availability()` 后，agent 认为哪些可用（**这就是小事 11 修的东西**）；
  3. 真的跑一次 pylint/flake8/bandit（**跑得起来才算数**，顺带覆盖小事 15 的临时文件修复）。

用法: venv/bin/python -u scripts/server_ops/srv_check_static_tools.py
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "local_libs"))
sys.path.insert(0, str(REPO))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from core.agents.static_scan_agent import StaticCodeScanAgent  # noqa: E402

TOOLS = ["pylint", "flake8", "bandit", "radon", "mypy", "semgrep", "cppcheck"]
SAMPLE = (
    "import os\n"
    "import sys\n"
    "\n"
    "def f(x):\n"
    "    y = 1\n"
    "    pw = 'hardcoded-secret'\n"
    "    return eval(x)\n"
)

print("=" * 88)
print("1) 事实基线：venv/bin 与系统 PATH 里各有什么")
print("=" * 88)
py_dir = os.path.dirname(os.path.abspath(sys.executable))
print("  当前解释器: %s" % sys.executable)
print("  venv 可执行目录: %s" % py_dir)
print("  该目录在 PATH 上吗: %s" % (py_dir in os.environ.get("PATH", "")))
for t in TOOLS:
    in_venv = shutil.which(t, path=py_dir) is not None
    on_path = shutil.which(t) is not None
    print("  %-9s venv里=%-5s 系统PATH上=%-5s  ← 历史实现只看后者" % (t, in_venv, on_path))

print()
print("=" * 88)
print("2) agent 的可用性判定（小事 11 的修复点）")
print("=" * 88)
agent = StaticCodeScanAgent()
asyncio.run(agent._check_tool_availability())
usable = [k for k, v in agent.available_tools.items() if v]
print("  判定可用: %s" % usable)
for t in TOOLS:
    if agent.available_tools.get(t):
        print("    %-9s -> %s" % (t, agent.resolved_tool_paths.get(t)))

print()
print("=" * 88)
print("3) 真的跑一次（小事 11 的收益 + 小事 15 的临时文件修复）")
print("=" * 88)
for name, fn in (("pylint", agent._run_pylint), ("flake8", agent._run_flake8),
                 ("bandit", agent._run_bandit)):
    if not agent.available_tools.get(name):
        print("  %-8s 未安装，跳过" % name)
        continue
    try:
        out = asyncio.run(fn(SAMPLE, str(REPO)))
        print("  %-8s 产出 %d 项" % (name, len(out)))
    except Exception as exc:  # noqa: BLE001
        print("  %-8s **抛异常**: %s: %s" % (name, type(exc).__name__, exc))
print()
print("结论：如果第 2 步的可用工具数 > 第 1 步『系统PATH上=是』的个数，说明小事 11 的修复在服务器上生效。")
