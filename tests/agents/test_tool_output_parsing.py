# -*- coding: utf-8 -*-
"""回归测试：外部工具输出的**字段位置**必须解析正确（《01》小事 19/20）。

两个缺陷：
* **19**：`_run_flake8` 传了 flake8 ≥6 已不支持的 `--format=json` —— 它把 "json" 当格式串，
  每个错误打印一行字面量 `json` → 这个运行器**永远产出 0 项**（实测 flake8 7.3.0 退出码 -1）。
* **20**：解析按 `split(':')` 的**固定下标**取数，而真实输出第一段是**文件路径**
  （Windows 还带盘符冒号）→ `line` 恒为 0、真行号被当列号、规则号位置是数字。**Linux 上同样错**。

这里的测试**不依赖本机装了什么工具、也不联网**：解析部分用构造的输出行，
运行器部分用 mock 掉的 `subprocess.run` 喂进"真实格式"的 stdout，
另外再用真工具跑一次做端到端确认（没装就 skip）。
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "local_libs"))
sys.path.insert(0, str(REPO_ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from core.agents.static_scan_agent import StaticCodeScanAgent  # noqa: E402

SOURCE = REPO_ROOT / "core/agents/static_scan_agent.py"

# 真实格式样本：Windows 盘符 + 行 + 列 + 规则号
WIN_FLAKE8 = r"E:\proj\venv\tmp\x.py:12:5: F841 local variable 'y' is assigned to but never used"
LINUX_FLAKE8 = "/tmp/static_scan_ab12.py:3:1: E302 expected 2 blank lines, found 1"
MYPY_WITH_COL = r"E:\proj\x.py:7:12: error: Incompatible return value type (got ""int"", expected ""str"")"
MYPY_NO_COL = "x.py:9: error: Name \"foo\" is not defined"


class TestToolOutputParsing(unittest.TestCase):
    def setUp(self):
        self.agent = StaticCodeScanAgent()

    # ---------- 结构守卫 ----------
    def test_no_json_format_flag(self):
        """结构：不许再传 flake8 不支持的 `--format=json`（小事 19）。"""
        code = "\n".join(l for l in SOURCE.read_text(encoding="utf-8").splitlines()
                         if not l.strip().startswith("#"))
        self.assertNotIn("--format=json", code,
                         "flake8 又不支持 JSON 格式化了：会永远产出 0 项")

    def test_runners_use_shared_parser(self):
        """结构：flake8/mypy 两个运行器都用同一个解析器（一条逻辑一个实现）。"""
        text = SOURCE.read_text(encoding="utf-8")
        for name in ("_run_flake8", "_run_mypy"):
            body = _function_body(text, name)
            self.assertIn("_parse_tool_location", body, "%s 没有用共享解析器" % name)
            self.assertNotIn("parts[1]", body, "%s 又回到按固定下标取数了" % name)

    # ---------- 解析单测（含 Windows 盘符、无列号）----------
    def test_parse_windows_path_with_drive_colon(self):
        p = self.agent._parse_tool_location(WIN_FLAKE8)
        self.assertIsNotNone(p)
        self.assertEqual(p["line"], 12)
        self.assertEqual(p["column"], 5)
        self.assertEqual(p["rest"], "F841 local variable 'y' is assigned to but never used")
        self.assertIn(r"E:\proj", p["path"])

    def test_parse_linux_path(self):
        p = self.agent._parse_tool_location(LINUX_FLAKE8)
        self.assertEqual((p["line"], p["column"]), (3, 1))
        self.assertTrue(p["rest"].startswith("E302"))

    def test_parse_without_column(self):
        p = self.agent._parse_tool_location(MYPY_NO_COL)
        self.assertEqual(p["line"], 9)
        self.assertEqual(p["column"], 0)
        self.assertTrue(p["rest"].startswith("error:"))

    def test_parse_rejects_garbage(self):
        for bad in ("", "not a location line", "*** 报告结束 ***", "json"):
            self.assertIsNone(self.agent._parse_tool_location(bad), bad)

    # ---------- 运行器：喂真实格式的 stdout ----------
    def test_flake8_runner_parses_line_col_code(self):
        class _R:
            returncode = 1
            stdout = "\n".join([WIN_FLAKE8, LINUX_FLAKE8])
            stderr = ""
        with patch.object(subprocess, "run", return_value=_R()):
            out = asyncio.run(self.agent._run_flake8("x = 1\n", "."))
        self.assertEqual(len(out), 2, "两条输出应当都解析出来")
        self.assertEqual([i["line"] for i in out], [12, 3])
        self.assertEqual([i["column"] for i in out], [5, 1])
        self.assertEqual([i["code"] for i in out], ["F841", "E302"])

    def test_mypy_runner_parses_line(self):
        class _R:
            returncode = 1
            stdout = "\n".join([MYPY_WITH_COL, MYPY_NO_COL])
            stderr = ""
        with patch.object(subprocess, "run", return_value=_R()):
            out = asyncio.run(self.agent._run_mypy("x = 1\n", "."))
        self.assertEqual([i["line"] for i in out], [7, 9])
        self.assertTrue(out[0]["message"].startswith("error:"))

    # ---------- 变异对照 ----------
    def test_mutation_old_index_based_parsing_would_fail(self):
        """变异对照：把老写法（按 split(':') 固定下标）套到真实输出上，字段一定是错的。

        这条保证上面的断言**真的在管解析**，而不是写了几个恒真的断言。
        """
        line = WIN_FLAKE8
        parts = line.split(":")
        old_line = int(parts[1]) if parts[1].isdigit() else 0
        old_code = parts[3].strip().split()[0] if parts[3].strip() else ""
        self.assertEqual(old_line, 0, "老写法在带盘符的路径上取不到行号")
        self.assertNotEqual(old_code, "F841", "老写法取到的不是规则号")
        # 新写法两条都对
        p = self.agent._parse_tool_location(line)
        self.assertEqual(p["line"], 12)
        self.assertEqual(p["rest"].split()[0], "F841")

    # ---------- 端到端（有工具才跑）----------
    def test_flake8_end_to_end(self):
        asyncio.run(self.agent._check_tool_availability())
        if not self.agent.available_tools.get("flake8"):
            self.skipTest("本机没有 flake8")
        out = asyncio.run(self.agent._run_flake8("import os\nimport sys\n\n\ndef f(x):\n    y = 1\n    return x\n", "."))
        self.assertTrue(out, "flake8 又产出空结果了 —— 19 或 20 可能回归了")
        self.assertTrue(all(i["line"] > 0 for i in out),
                        "行号应当 > 0（历史实现恒为 0）")
        self.assertTrue(all(i["code"].startswith(("F", "E", "W", "C", "N")) for i in out),
                        "code 应当是规则号，不是数字")


def _function_body(text: str, name: str) -> str:
    import re
    m = re.search(r"\n    async def %s\(" % re.escape(name), text)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"\n    (async )?def ", rest)
    return rest[: nxt.start()] if nxt else rest


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
