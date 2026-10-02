# -*- coding: utf-8 -*-
"""回归测试：safety 运行器（《01》小事 16）。

历史状态：`safety` 被列进工具配置、也做了可用性检查，但**根本没有运行器** ——
于是 `enable_safety` / `safety_timeout` 永远不会生效（"看着像旋钮，其实是装饰"）。

safety 与其它工具不同：它扫的是**依赖声明**（requirements.txt 等），不是代码片段。
所以三条必须成立：① 没有清单要**跳过并写日志**（不静默）；② 老/新两种输出格式都要能解析；
③ 离线/需认证导致的失败**要写明原因**、不能假装成功。

测试不联网、不要求本机装 safety（用 mock 喂真实格式的输出）。
工作目录用**仓库内固定路径**：系统临时目录在本机沙箱里清理会抛 WinError 5（《03》坑 32）。
"""

from __future__ import annotations

import asyncio
import json
import shutil
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

SCRATCH = REPO_ROOT / "reports" / "_test_safety_runner"
SOURCE = REPO_ROOT / "core/agents/static_scan_agent.py"

V1 = [["requests", "2.19.0", "<2.20.0", "pyup.io-36107", "会话与 cookie 泄露", "CVE-2018-18074"],
      ["jinja2", "2.10", "<2.10.1", "pyup.io-36641", "沙箱逃逸", "CVE-2019-10906"]]
V3 = {"vulnerabilities": [{"package_name": "urllib3", "vulnerability_id": "GHSA-x",
                          "advisory": "请求走私"}]}


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestSafetyRunner(unittest.TestCase):
    def setUp(self):
        self.agent = StaticCodeScanAgent()
        self.work = SCRATCH / "case"
        if self.work.exists():
            shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def _manifest(self, text="requests==2.19.0\n"):
        (self.work / "requirements.txt").write_text(text, encoding="utf-8")

    # ---------- 结构 ----------
    def test_run_safety_exists_and_is_wired(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("async def _run_safety(", text, "运行器又不见了")
        body = _function_body(text, "_run_security_scans")
        self.assertIn('available_tools.get("safety")', body, "safety 没被接进安全扫描")
        self.assertIn("_run_safety", body)

    # ---------- ① 没有清单 ----------
    def test_skip_without_manifest(self):
        out = asyncio.run(self.agent._run_safety("x = 1\n", str(self.work)))
        self.assertEqual(out, [], "没有依赖清单时应当返回空")

    # ---------- ② 两种输出格式 ----------
    def test_parse_v1_list_format(self):
        self._manifest()
        with patch.object(subprocess, "run", return_value=_Completed(stdout=json.dumps(V1))):
            out = asyncio.run(self.agent._run_safety("x = 1\n", str(self.work)))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["package"], "requests")
        self.assertEqual(out[0]["vulnerability_id"], "pyup.io-36107")
        self.assertEqual(out[0]["manifest"], "requirements.txt")
        self.assertTrue(all(i["tool"] == "safety" for i in out))

    def test_parse_v3_dict_format(self):
        self._manifest("urllib3==1.24\n")
        with patch.object(subprocess, "run",
                          return_value=_Completed(returncode=64, stdout=json.dumps(V3))):
            out = asyncio.run(self.agent._run_safety("x = 1\n", str(self.work)))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["package"], "urllib3")
        self.assertEqual(out[0]["vulnerability_id"], "GHSA-x")

    # ---------- ③ 失败要写明原因、不能假成功 ----------
    def test_no_output_is_not_a_success(self):
        self._manifest()
        with patch.object(subprocess, "run",
                          return_value=_Completed(returncode=1, stdout="",
                                                  stderr="Safety requires an API key")):
            out = asyncio.run(self.agent._run_safety("x = 1\n", str(self.work)))
        self.assertEqual(out, [])

    def test_bad_json_does_not_raise(self):
        self._manifest()
        with patch.object(subprocess, "run", return_value=_Completed(stdout="not json")):
            out = asyncio.run(self.agent._run_safety("x = 1\n", str(self.work)))
        self.assertEqual(out, [])

    def test_timeout_does_not_raise(self):
        self._manifest()
        with patch.object(subprocess, "run",
                          side_effect=subprocess.TimeoutExpired(cmd="safety", timeout=1)):
            out = asyncio.run(self.agent._run_safety("x = 1\n", str(self.work)))
        self.assertEqual(out, [])

    # ---------- 变异对照 ----------
    def test_mutation_missing_manifest_guard_would_call_the_tool(self):
        """变异对照：若去掉"没有清单就跳过"的判断，就会真的去调 safety（本机没装 → 报错）。

        这条保证"没有清单要跳过"这个断言真的在管行为。
        """
        called = {}

        def _spy(*a, **k):
            called["yes"] = True
            raise FileNotFoundError("safety 不存在")

        with patch.object(subprocess, "run", side_effect=_spy):
            out = asyncio.run(self.agent._run_safety("x = 1\n", str(self.work)))
        self.assertEqual(out, [])
        self.assertFalse(called.get("yes"), "没有清单时**不应该**调用 safety")


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
