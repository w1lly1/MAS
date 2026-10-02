# -*- coding: utf-8 -*-
"""回归测试：5 个 Python 工具运行器**不能把临时文件写死在 `/tmp`**。

现象（《01》小事 15）：`_run_pylint/_run_flake8/_run_bandit/_run_radon_analysis/_run_mypy`
把待分析代码写死到 `/tmp/code_analysis.py`。Linux 上能用，非 POSIX 主机上
`/tmp` 会解析成"当前盘符下的 \\tmp"（通常不可写）→ 运行器**直接失败**。
它很隐蔽：小事 11 把工具探测修好之后，**卡点就换成了这一步**——工具找到了却跑不起来。

三层保证：
  1. **结构**：源码里不许再出现把该路径当文件写的代码（注释里提到它不算）；
     5 个运行器必须走同一个 `_write_temp_source`，并带 `temp_file = None` + `finally` 清理；
  2. **功能**：真的跑一次 pylint，必须产出结果（修之前这里会因为 PermissionError 返回空）；
  3. **变异对照**：把写临时文件这一步改成抛异常，运行器必须优雅返回 —— 少了
     `temp_file = None` 初始化，`finally` 会抛 NameError 并盖掉原异常，这条就会红。
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "local_libs"))
sys.path.insert(0, str(REPO_ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from core.agents.static_scan_agent import StaticCodeScanAgent  # noqa: E402

SOURCE = REPO_ROOT / "core/agents/static_scan_agent.py"
RUNNERS = ("_run_pylint", "_run_flake8", "_run_bandit", "_run_radon_analysis", "_run_mypy")
SAMPLE = "import os\nimport sys\n\n\ndef f(x):\n    y = 1\n    return x\n"


class TestNoHardcodedTempPath(unittest.TestCase):
    def test_no_code_line_writes_to_tmp(self):
        """结构：代码行里不许再把 "/tmp/code_analysis.py" 当文件写（注释里提到可以）。"""
        hits = []
        for i, line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if "/tmp/code_analysis.py" in line and "open(" in line:
                hits.append(i)
        self.assertEqual(hits, [], "第 %s 行仍在写死 /tmp 路径" % hits)

    def test_runners_use_the_shared_temp_writer(self):
        """结构：5 个运行器都用同一个 `_write_temp_source`（一条逻辑只允许一个实现）。"""
        text = SOURCE.read_text(encoding="utf-8")
        for name in RUNNERS:
            body = _function_body(text, name)
            self.assertIn("self._write_temp_source(code_content", body,
                          "%s 没有用共享的临时文件写法" % name)
            self.assertIn("suffix=\".py\"", body, "%s 的临时文件后缀丢了" % name)

    def test_runners_clean_up_in_finally(self):
        """结构：清理必须在 finally 里，且 temp_file 先置空（异常路径也要删得掉）。"""
        text = SOURCE.read_text(encoding="utf-8")
        for name in RUNNERS:
            body = _function_body(text, name)
            self.assertIn("temp_file = None", body, "%s 少了 temp_file = None 初始化" % name)
            self.assertIn("finally:", body, "%s 的清理没放在 finally 里" % name)
            self.assertIn("os.remove(temp_file)", body, "%s 没有删除临时文件" % name)

    def test_pylint_actually_runs(self):
        """功能：真的跑一次 pylint 并产出结果（这就是修好前失败的那条路径）。"""
        agent = StaticCodeScanAgent()
        asyncio.run(agent._check_tool_availability())
        if not agent.available_tools.get("pylint"):
            self.skipTest("本机没有 pylint，跳过功能校验")
        issues = asyncio.run(agent._run_pylint(SAMPLE, str(REPO_ROOT)))
        self.assertTrue(issues, "pylint 跑出来了空结果 —— 很可能是临时文件那一步又坏了")
        self.assertTrue(all(i.get("tool") == "pylint" for i in issues))

    def test_mutation_temp_writer_raises_is_handled(self):
        """变异对照：写临时文件抛异常 → 运行器优雅返回，且**不能**出 NameError。"""
        agent = StaticCodeScanAgent()
        original = agent._write_temp_source

        def boom(*a, **k):
            raise RuntimeError("模拟写临时文件失败")

        agent._write_temp_source = boom
        try:
            for name in RUNNERS:
                try:
                    out = asyncio.run(getattr(agent, name)(SAMPLE, str(REPO_ROOT)))
                except NameError as exc:  # temp_file 没初始化时 finally 会抛这个
                    self.fail("%s 在异常路径上抛了 NameError（temp_file 没置空）: %s"
                              % (name, exc))
                self.assertEqual(out, [] if isinstance(out, list) else
                                 {"cyclomatic_complexity": {}, "maintainability_index": 0.0,
                                  "average_complexity": 0.0},
                                 "%s 应当返回空结果" % name)
        finally:
            agent._write_temp_source = original


def _function_body(text: str, name: str) -> str:
    """取出某个方法的方法体（到下一个同级 def 为止）。"""
    m = re.search(r"\n    async def %s\(" % re.escape(name), text)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"\n    (async )?def ", rest)
    return rest[: nxt.start()] if nxt else rest


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
