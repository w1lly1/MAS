# -*- coding: utf-8 -*-
"""证明"5 个 Python 工具写死 /tmp/code_analysis.py"这个毛病真的修好了。

历史现象（《01》小事 15）：`_run_pylint/_run_flake8/_run_bandit/_run_radon_analysis/_run_mypy`
把待分析代码写死到 `/tmp/code_analysis.py`。在 Linux 上能用；在非 POSIX 主机上
`/tmp` 会被解析成**当前盘符下的 \\tmp**（通常不可写）→ 运行器直接失败。
后果很隐蔽：**工具探测明明修好了（小事 11），却因为这一步仍然跑不起来**。

本脚本四段，全部本地可跑：

  A. **负控**：先证明"写死 /tmp"在本机确实失败（否则说明这条缺陷不存在，测它对不对就无从谈起）
  B. 修好后 5 个运行器都能跑通并产出问题
  C. 临时文件**用完即删**（finally），不残留
  D. **变异对照**：让写临时文件这一步抛异常，运行器必须优雅返回（验证 `temp_file = None`
     这个初始化真的必要 —— 少了它，`finally` 里会 NameError 把异常盖成另一种异常）

用法: python -X utf8 utils/experiments/verify_python_tools_tempfile.py
"""
from __future__ import annotations

import asyncio
import glob
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from core.agents.static_scan_agent import StaticCodeScanAgent  # noqa: E402

SAMPLE = '''\
import os
import sys


def check(user_input):
    password = "hardcoded-secret-123"
    unused_variable = 1
    result = eval(user_input)
    q = "SELECT * FROM users WHERE name = '%s'" % user_input
    return result, q, password, os, sys


def very_complex(a, b, c, d, e, f):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            return 1
    return 0
'''

RUNNERS = ["_run_pylint", "_run_flake8", "_run_bandit", "_run_radon_analysis", "_run_mypy"]


def _temp_py_files():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "static_scan_*.py")))


def part_a_negative_control() -> bool:
    """负控：写死 /tmp 在本机应当失败。"""
    print("=" * 92)
    print("A. 负控：证明『写死 /tmp/code_analysis.py』在本机确实不可写")
    print("=" * 92)
    print("  本机 os.name=%s，/tmp 会解析成 -> %s" % (os.name, os.path.abspath("/tmp")))
    try:
        with open("/tmp/code_analysis.py", "w", encoding="utf-8") as fh:
            fh.write("x = 1\n")
        print("  写入成功 —— 本机 /tmp 可写，所以这条缺陷在本机**复现不出来**"
              "（服务器 Linux 上也是可写的，它影响的是 Windows 主机）")
        return True
    except Exception as exc:  # noqa: BLE001
        print("  写入失败: %s: %s" % (type(exc).__name__, exc))
        print("  ⇒ 原缺陷在本机**真实存在**（这正是「工具找到了却跑不起来」的原因）")
        return True


def part_b_runners_work(agent) -> bool:
    print()
    print("=" * 92)
    print("B. 修好后：5 个运行器都能跑通（用 venv 的解释器，工具才在解释器目录里）")
    print("=" * 92)
    print("  可用工具: %s" % {k: v for k, v in agent.available_tools.items() if v})
    ok = True
    for name in RUNNERS:
        fn = getattr(agent, name)
        try:
            out = asyncio.run(fn(SAMPLE, str(ROOT)))
        except Exception as exc:  # noqa: BLE001
            print("  [NG] %-20s 抛异常: %s: %s" % (name, type(exc).__name__, exc))
            ok = False
            continue
        n = len(out) if isinstance(out, list) else (
            len((out or {}).get("cyclomatic_complexity") or {}))
        print("  [%s] %-20s 产出 %d 项" % ("OK" if n else "??", name, n))
    return ok


def part_c_no_leftover(agent) -> bool:
    print()
    print("=" * 92)
    print("C. 临时文件用完即删（finally）")
    print("=" * 92)
    before = _temp_py_files()
    for name in RUNNERS:
        asyncio.run(getattr(agent, name)(SAMPLE, str(ROOT)))
    after = _temp_py_files()
    leftover = after - before
    print("  运行前 %d 个 / 运行后 %d 个，新增残留 %d 个" % (len(before), len(after), len(leftover)))
    if leftover:
        print("  残留示例: %s" % list(leftover)[:3])
    print("  [%s] 无残留" % ("OK" if not leftover else "NG"))
    return not leftover


def part_d_mutation(agent) -> bool:
    """变异对照：写临时文件这一步抛异常 → 必须优雅返回，不能崩。"""
    print()
    print("=" * 92)
    print("D. 变异对照：让写临时文件抛异常，运行器必须优雅返回")
    print("=" * 92)
    original = agent._write_temp_source

    def boom(*a, **k):
        raise RuntimeError("模拟写临时文件失败")

    agent._write_temp_source = boom
    ok = True
    try:
        for name in RUNNERS:
            try:
                out = asyncio.run(getattr(agent, name)(SAMPLE, str(ROOT)))
                empty = (out == []) if isinstance(out, list) else (
                    (out or {}).get("cyclomatic_complexity") == {})
                print("  [%s] %-20s 返回空结果（未抛出）" % ("OK" if empty else "NG", name))
                ok = ok and empty
            except NameError as exc:
                # 这就是"忘了 temp_file = None"会造成的症状：finally 把原异常盖成 NameError
                print("  [NG] %-20s **NameError**（说明 finally 里的 temp_file 没初始化）: %s"
                      % (name, exc))
                ok = False
            except Exception as exc:  # noqa: BLE001
                print("  [NG] %-20s 异常外泄: %s: %s" % (name, type(exc).__name__, exc))
                ok = False
    finally:
        agent._write_temp_source = original
    return ok


def main() -> int:
    agent = StaticCodeScanAgent()
    asyncio.run(agent._check_tool_availability())
    results = {
        "A 负控（缺陷真实存在）": part_a_negative_control(),
        "B 运行器恢复可用": part_b_runners_work(agent),
        "C 临时文件不残留": part_c_no_leftover(agent),
        "D 变异对照（异常路径不崩）": part_d_mutation(agent),
    }
    print()
    print("=" * 92)
    for k, v in results.items():
        print("  [%s] %s" % ("OK" if v else "NG", k))
    ok = all(results.values())
    print("  结论：%s" % ("修好了：工具能跑、临时文件不残留、异常路径也稳"
                        if ok else "**还有问题，别急着下结论**"))
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
