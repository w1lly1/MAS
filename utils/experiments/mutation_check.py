# -*- coding: utf-8 -*-
"""变异检验：确认一条回归测试**真的会红**。

## 为什么必须有这个工具

这个项目栽过两次同一类跟头：**写了一个永远会通过的测试**，
于是拿"全绿"当"已验证"去汇报（见《03_踩过的坑》里"变异测试假通过"那一条）。
所以凡是为某个修复新增回归测试，都应先回答一句：
**把修复点改回错误写法，这条测试会不会失败？**

## 做法（按字节替换，不是按字符串拼接）

1. 读出文件**字节**，断言目标片段**唯一出现**；
2. 替换后断言**内容确实变了**（防止"改了但没生效"，行尾差异就踩过这个坑）；
3. 跑指定的测试节点，期望**非 0 退出码**；
4. `finally` 里按原字节还原，并断言还原成功。

## 用法

    python utils/experiments/mutation_check.py \
        --file api/main.py \
        --find "'status': 'partial'" \
        --replace "'status': 'done'" \
        --test tests/test_batch_partial_status.py::TestBatchFlowMarksPartial::test_timed_out_item_is_partial_not_done
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PY = str(ROOT / "venv" / "Scripts" / "python.exe")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="被变异的源文件（相对仓库根）")
    ap.add_argument("--find", required=True, help="要被替换掉的原文（唯一出现）")
    ap.add_argument("--replace", required=True, help="替换成什么（应为错误写法）")
    ap.add_argument("--test", required=True, help="测试节点 id")
    args = ap.parse_args()

    target = ROOT / args.file
    if not target.exists():
        print("文件不存在：", target)
        return 2

    original = target.read_bytes()
    find = args.find.encode("utf-8")
    replace = args.replace.encode("utf-8")

    count = original.count(find)
    if count != 1:
        print("目标片段出现 %d 次（必须是 1 次），本次检验无效" % count)
        return 2

    mutated = original.replace(find, replace)
    if mutated == original:
        print("替换没有生效，本次检验无效")
        return 2

    target.write_bytes(mutated)
    print("已注入变异：%s" % args.find)
    try:
        proc = subprocess.run(
            [PY, "-m", "pytest", args.test, "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-3:]
        print("变异后 pytest 退出码 =", proc.returncode)
        for ln in tail:
            print("   ", ln)
        ok = proc.returncode != 0
        print("\n结论：", "回归测试能抓住这个缺陷（OK）" if ok else "*** 测试没红，这条回归测试是假的 ***")
        return 0 if ok else 1
    finally:
        target.write_bytes(original)
        restored = target.read_bytes() == original
        print("已还原原文件：", "OK" if restored else "*** 还原失败，请手动检查 ***")
        if not restored:
            sys.exit(3)


if __name__ == "__main__":
    sys.exit(main())
