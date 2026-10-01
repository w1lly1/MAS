# -*- coding: utf-8 -*-
"""A/B 检验：改动有没有让任何测试从"通过"变成"失败"。

## 为什么需要它（而不是直接看"失败数"）

这台机器上**本来就有一批用例注定失败**：没有 GPU、没有本地 Weaviate、
部分 agent 需要模型服务。所以"失败数是不是 0"根本没有意义 ——
唯一有意义的比较是**失败集合的差集**：

* `fail_with - fail_without` = 我的改动**新弄坏**的用例（必须为空）
* `fail_without - fail_with` = 我的改动顺手修好的用例

## 做法

1. 备份要检验的文件并记哈希；
2. 带改动跑一遍 `pytest tests`，记下失败/报错集合；
3. `git stash push -u -- <这些文件>`，再跑一遍（基线）；
4. `git stash pop` 还原，并**用哈希断言还原成功**（不一致就用备份覆盖）；
5. 打印两个差集。

> 第 4 步的哈希断言不是多余的：`core.autocrlf` 打开时 `git stash pop` 写回的文件
> 行尾可能是 CRLF，与改动前的 LF 不同 —— 只看"命令没报错"会以为还原成功了。

## 用法

    python utils/experiments/ab_test_suite.py
    python utils/experiments/ab_test_suite.py --files api/main.py core/agents_integration.py
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
BACKUP = ROOT / "_ab_test_backup"

DEFAULT_FILES = [
    "api/main.py",
    "core/agents_integration.py",
    "core/agents/analysis_result_summary_agent.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_suite() -> tuple:
    proc = subprocess.run(
        [PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    lines = proc.stdout.splitlines()
    fails = set()
    for ln in lines:
        if ln.startswith(("FAILED", "ERROR")):
            parts = ln.split(" ")
            if len(parts) >= 2:
                fails.add(parts[1])
    summary = next((ln for ln in reversed(lines) if "passed" in ln or "failed" in ln), "")
    return fails, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None,
                    help="要检验的改动文件（默认 api/main.py 与两个 agent 文件）")
    ap.add_argument("--keep-backup", action="store_true", help="跑完保留备份目录")
    args = ap.parse_args()

    files = args.files or DEFAULT_FILES
    missing = [f for f in files if not (ROOT / f).exists()]
    if missing:
        print("这些文件不存在：", missing)
        return 2

    BACKUP.mkdir(exist_ok=True)
    for rel in files:
        shutil.copy2(ROOT / rel, BACKUP / Path(rel).name)
    before = {rel: sha(ROOT / rel) for rel in files}
    print("已备份 %d 个文件" % len(files))

    print("\n[1/2] 带改动跑测试 ...")
    fail_with, sum_with = run_suite()
    print("     ", sum_with)

    stashed = False
    fail_without = set()
    try:
        proc = subprocess.run(["git", "stash", "push", "-u", "--", *files],
                              cwd=str(ROOT), capture_output=True, text=True)
        stashed = proc.returncode == 0
        if not stashed:
            print("[stash] 失败：", (proc.stderr or proc.stdout).strip()[:200])
            return 3
        print("\n[2/2] 不带改动跑测试（基线）...")
        fail_without, sum_without = run_suite()
        print("     ", sum_without)
    finally:
        if stashed:
            pop = subprocess.run(["git", "stash", "pop"], cwd=str(ROOT),
                                 capture_output=True, text=True)
            print("\n[stash pop]", "OK" if pop.returncode == 0 else pop.stderr.strip()[:200])

    after = {rel: sha(ROOT / rel) for rel in files}
    if after != before:
        print("还原后哈希不一致（很可能是行尾差异），用备份覆盖 ...")
        for rel in files:
            shutil.copy2(BACKUP / Path(rel).name, ROOT / rel)
        after = {rel: sha(ROOT / rel) for rel in files}
    print("还原校验：", "OK" if after == before else "*** 仍不一致，请手动检查 ***")
    if not args.keep_backup and after == before:
        shutil.rmtree(BACKUP, ignore_errors=True)

    newly_broken = sorted(fail_with - fail_without)
    newly_fixed = sorted(fail_without - fail_with)
    print("\n" + "=" * 74)
    print("失败集合：带改动 %d 个，基线 %d 个" % (len(fail_with), len(fail_without)))
    print("我的改动**新弄坏**的用例：", newly_broken or "无（OK）")
    print("我的改动顺手修好的用例：", newly_fixed or "无")
    print("=" * 74)
    return 0 if not newly_broken else 1


if __name__ == "__main__":
    sys.exit(main())
