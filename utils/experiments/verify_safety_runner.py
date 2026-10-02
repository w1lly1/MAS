"""验证 safety 运行器三条路径：无清单跳过 / 老格式解析 / 新格式解析。

（用 mock 喂真实格式的输出，**不联网、不依赖本机装没装 safety**；
最后再用真 safety 跑一次可用性确认，没装就跳过。）
"""
import asyncio
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

# ⚠️ 不要用系统临时目录：在本机沙箱里连 `TemporaryDirectory(ignore_cleanup_errors=True)`
# 都会在清理时抛 WinError 5（实测）。改用**仓库内自管目录**（《03》坑 32 同类）。
SCRATCH = ROOT / "reports" / "_safety_verify"


@contextlib.contextmanager
def workdir(name: str = "case"):
    """固定的工作目录（每个用例进入前清空）。

    两个环境坑都踩过：① 系统临时目录在本机沙箱里**清理时**抛 WinError 5
    （连 `TemporaryDirectory(ignore_cleanup_errors=True)` 都不行）；
    ② 用 `mkdtemp` 造的随机名子目录**连写文件都被沙箱拒**（按临时目录规则挡掉）。
    所以改用**固定名**、且放在仓库内的 `reports/_safety_verify/` 下。
    """
    d = SCRATCH / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield str(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from core.agents.static_scan_agent import StaticCodeScanAgent  # noqa: E402

agent = StaticCodeScanAgent()
agent.available_tools["safety"] = True

print("=" * 88)
print("1) 没有依赖清单 → 跳过并写日志（不静默、不抛异常）")
print("=" * 88)
with workdir() as d:
    out = asyncio.run(agent._run_safety("x = 1\n", d))
print("  产出 %d 项（应为 0）" % len(out))

print()
print("=" * 88)
print("2) safety 1.x/2.x 的列表格式")
print("=" * 88)
V1 = [["requests", "2.19.0", "<2.20.0", "pyup.io-36107", "Requests 会话与 cookie 泄露", "CVE-2018-18074"],
      ["jinja2", "2.10", "<2.10.1", "pyup.io-36641", "沙箱逃逸", "CVE-2019-10906"]]


class _R1:
    returncode = 0
    stdout = json.dumps(V1)
    stderr = ""


with workdir() as d:
    Path(d, "requirements.txt").write_text("requests==2.19.0\n", encoding="utf-8")
    with patch.object(subprocess, "run", return_value=_R1()):
        out = asyncio.run(agent._run_safety("x = 1\n", d))
print("  产出 %d 项" % len(out))
for it in out[:3]:
    print("   pkg=%-10s vuln=%-16s sev=%s msg=%s"
          % (it["package"], it["vulnerability_id"], it["severity"], it["message"][:48]))

print()
print("=" * 88)
print("3) safety 3.x 的 dict 格式")
print("=" * 88)
V3 = {"vulnerabilities": [{"package_name": "urllib3", "vulnerability_id": "GHSA-x",
                          "advisory": "请求走私"}]}


class _R3:
    returncode = 64          # 3.x 命中漏洞时返回非 0
    stdout = json.dumps(V3)
    stderr = ""


with workdir() as d:
    Path(d, "requirements.txt").write_text("urllib3==1.24\n", encoding="utf-8")
    with patch.object(subprocess, "run", return_value=_R3()):
        out = asyncio.run(agent._run_safety("x = 1\n", d))
print("  产出 %d 项" % len(out))
for it in out[:3]:
    print("   pkg=%-10s vuln=%-16s msg=%s" % (it["package"], it["vulnerability_id"], it["message"][:44]))

print()
print("=" * 88)
print("4) 出错（离线/需认证）→ 写日志、返回空，不抛异常")
print("=" * 88)


class _RErr:
    returncode = 1
    stdout = ""
    stderr = "Safety requires an API key for this command"


with workdir() as d:
    Path(d, "requirements.txt").write_text("x\n", encoding="utf-8")
    with patch.object(subprocess, "run", return_value=_RErr()):
        out = asyncio.run(agent._run_safety("x = 1\n", d))
print("  产出 %d 项（应为 0），上面应有告警日志" % len(out))

print()
print("=" * 88)
print("5) 真实可用性（本机装没装）")
print("=" * 88)
asyncio.run(agent._check_tool_availability())
print("  safety 可用: %s  路径: %s" % (agent.available_tools.get("safety"),
                                      agent.resolved_tool_paths.get("safety")))
with workdir() as d:
    Path(d, "requirements.txt").write_text("requests==2.19.0\n", encoding="utf-8")
    real = asyncio.run(agent._run_safety("x = 1\n", d)) if agent.available_tools.get("safety") else None
print("  真跑（若无网络/需认证则会告警并返回空）: %s" % (("%d 项" % len(real)) if real is not None else "未安装，跳过"))
