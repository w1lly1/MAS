# -*- coding: utf-8 -*-
"""小事 11 回归测试：外部静态分析工具的**定位**不能只看系统 PATH。

现象（《01》小事 11）：工具装在当前 Python 解释器所属环境里（`venv/Scripts`、`venv/bin`），
而这些目录未必在 PATH 上 —— 历史实现只用 `shutil.which(tool)`，于是"明明装了却报未安装"，
`available_tools` 全为假，外部工具被整体跳过。

这里用"假的解释器目录 + 假的可执行文件"离线复现，不依赖本机装了哪些工具、不联网、不建向量库。

临时目录放在仓库内（`.tmp_test/`，已在 .gitignore）——受限环境里系统临时目录可能不可写（见《03》坑 22）。
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import unittest
from unittest.mock import mock_open, patch

from utils.offline_imports import install_weaviate_stub

install_weaviate_stub()

from core.agents.static_scan_agent import StaticCodeScanAgent  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_executable(directory: str, name: str) -> str:
    """在 directory 下造一个"看起来可执行"的占位文件，返回其路径。"""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name + ".exe" if os.name == "nt" else name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")
    if os.name != "nt":
        os.chmod(path, 0o755)
    return path


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


_REAL_WHICH = shutil.which


def _which_without_system_path(name, path=None):
    """模拟"系统 PATH 上什么都没有"，但仍按 path 参数真实查找。"""
    if path is None:
        return None
    return _REAL_WHICH(name, path=path)


class TestStaticToolDiscovery(unittest.TestCase):
    """工具探测：当前 Python 所在目录优先于系统 PATH。"""

    def setUp(self):
        # 固定名字的脚手架目录，放在仓库内（`.tmp_test/` 已 gitignore）。
        # 注意两点环境约束（都实测过）：
        #   ① 受限沙箱会拒绝在 `tempfile.mkdtemp()` 造出的目录下再建子目录，
        #      所以这里用**一次 `os.makedirs` 把多层一次建出**的固定路径；
        #   ② 不要用系统临时目录（见《03》坑 22）。
        self.tmp = os.path.join(REPO_ROOT, ".tmp_test", "static_tool_discovery_fixture")
        # 造一个"虚拟环境目录"：<tmp>/interp_bin/python 与同目录下的工具
        self.venv_bin = os.path.join(self.tmp, "interp_bin")
        os.makedirs(self.venv_bin, exist_ok=True)
        self.fake_python = _make_executable(self.venv_bin, "python")
        self.fake_flake8 = _make_executable(self.venv_bin, "flake8")
        self.agent = StaticCodeScanAgent()

    def tearDown(self):
        # 只删自己造的两个文件与目录；删不掉也不影响结论（目录已 gitignore）
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_dirs_start_with_current_python_dir(self):
        with patch.object(sys, "executable", self.fake_python):
            dirs = self.agent._tool_search_dirs()
        self.assertEqual(dirs[0], self.venv_bin)

    def test_resolve_finds_tool_next_to_python_even_if_not_on_path(self):
        """核心回归：工具只在解释器目录里、PATH 上没有，也必须找得到。"""
        with patch.object(sys, "executable", self.fake_python), \
                patch("shutil.which", side_effect=_which_without_system_path):
            resolved = self.agent._resolve_tool_path("flake8")
        self.assertIsNotNone(resolved, "解释器同目录的 flake8 没有被找到")
        self.assertEqual(os.path.normcase(os.path.abspath(resolved)),
                         os.path.normcase(os.path.abspath(self.fake_flake8)))

    def test_missing_tool_returns_none(self):
        with patch.object(sys, "executable", self.fake_python), \
                patch("shutil.which", return_value=None):
            self.assertIsNone(self.agent._resolve_tool_path("no_such_tool_xyz"))

    def test_explicit_config_path_wins(self):
        self.agent.agent_config = {"tool_paths": {"flake8": self.fake_flake8}}
        with patch.object(sys, "executable", self.fake_python):
            self.assertEqual(self.agent._resolve_tool_path("flake8"), self.fake_flake8)

    def test_bad_explicit_config_path_does_not_silently_fall_back(self):
        """显式配了一个不存在的路径 → 判为不可用，不许悄悄用别的副本。"""
        self.agent.agent_config = {
            "tool_paths": {"flake8": os.path.join(self.tmp, "nope", "flake8")}
        }
        with patch.object(sys, "executable", self.fake_python):
            self.assertIsNone(self.agent._resolve_tool_path("flake8"))

    def test_extra_search_paths_from_config_are_honoured(self):
        """配置里额外追加的搜索目录也必须生效（工具既不在解释器目录、也不在 PATH 时用）。"""
        other = os.path.join(self.tmp, "extra_dir")
        fake = _make_executable(other, "flake8")
        self.agent.agent_config = {"tool_search_paths": [other]}
        with patch.object(sys, "executable", self.fake_python), \
                patch("shutil.which", side_effect=_which_without_system_path):
            resolved = self.agent._resolve_tool_path("flake8")
        self.assertEqual(os.path.normcase(os.path.abspath(resolved)),
                         os.path.normcase(os.path.abspath(fake)))

    def test_tool_config_is_read_live_not_snapshotted(self):
        """配置换了要立刻生效：不许用 __init__ 时的旧快照（静默失配类问题）。"""
        self.agent.agent_config = {"tool_paths": {"flake8": self.fake_flake8}}
        self.assertEqual(self.agent.configured_tool_paths, {"flake8": self.fake_flake8})
        self.agent.agent_config = {"tool_paths": {}}
        self.assertEqual(self.agent.configured_tool_paths, {})
        self.agent.agent_config = {"tool_search_paths": ["/tmp/a"]}
        self.assertEqual(self.agent.tool_search_paths, ["/tmp/a"])

    def test_availability_marks_python_dir_tool_available(self):
        """`_check_tool_availability` 必须把解释器同目录的工具判为可用（历史实现在这里判成"未安装"）。"""
        with patch.object(sys, "executable", self.fake_python), \
                patch("shutil.which", side_effect=_which_without_system_path), \
                patch("subprocess.run", return_value=_FakeCompleted(returncode=0)) as run:
            asyncio.run(self.agent._check_tool_availability())

        self.assertTrue(self.agent.available_tools.get("flake8"),
                        f"flake8 应被判为可用，实际 available_tools={self.agent.available_tools}")
        self.assertEqual(
            os.path.normcase(os.path.abspath(self.agent.resolved_tool_paths["flake8"])),
            os.path.normcase(os.path.abspath(self.fake_flake8)),
        )
        # 探测用的命令必须是绝对路径（不是裸名字）
        self.assertEqual(
            os.path.normcase(os.path.abspath(run.call_args[0][0][0])),
            os.path.normcase(os.path.abspath(self.fake_flake8)),
        )

    def test_runner_commands_use_resolved_absolute_path(self):
        """探测到的绝对路径要**真的用于调用**工具（否则探测成功也白搭）。"""
        self.agent.resolved_tool_paths = {"flake8": self.fake_flake8}
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = list(command)
            return _FakeCompleted(returncode=0, stdout="")

        with patch("subprocess.run", side_effect=fake_run), \
                patch("builtins.open", mock_open()), \
                patch("os.path.exists", return_value=False):
            issues = asyncio.run(self.agent._run_flake8("x = 1\n", ""))

        self.assertEqual(issues, [])
        self.assertEqual(
            os.path.normcase(os.path.abspath(captured["command"][0])),
            os.path.normcase(os.path.abspath(self.fake_flake8)),
            f"flake8 调用没用探测到的路径: {captured['command'][0]!r}",
        )

    def test_tool_cmd_falls_back_to_bare_name(self):
        """没探测到路径时保持历史行为（裸名字，交给 PATH）。"""
        self.assertEqual(self.agent._tool_cmd("flake8"), "flake8")


if __name__ == "__main__":
    unittest.main()
