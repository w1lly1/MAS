"""库外评测里"同一个文件"的判据测试。

**为什么需要**：这里踩过一次度量 bug —— 知识库里的 `file_pattern` 用真斜杠
（`include/freerdp/codec/nsc.h`），而运行产物里的文件路径是**压平的**
（`.../d1112c27/include__freerdp__codec__nsc.h`）。第一版按 `/` 取末两段，
于是三臂 35/36/37 条放行**全被判成"跨文件"**，与已知结果矛盾（《03》坑 33）。

下面每个断言都对应一个具体写法，改坏归一化就会红。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT / "local_libs"), str(REPO_ROOT)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from utils.experiments.eval_held_runs import classify, file_identity  # noqa: E402

KB_FILE = "include/freerdp/codec/nsc.h"
FLAT_SAME = ("/root/autodl-tmp/MAS/tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/"
             "source_code_restructured/before/CVE-2018-8788/d1112c27/"
             "include__freerdp__codec__nsc.h")
FLAT_OTHER = ("/root/autodl-tmp/MAS/tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/"
              "source_code_restructured/before/CVE-2018-8788/d1112c27/"
              "libfreerdp__codec__nsc_encode.h")
WIN_PATH = (r"E:\MyOwn\ProgramStudy\MAS\tests\BigVul\MSR_20_Code_vulnerability_CSV_Dataset"
            r"\source_code_restructured\before\CVE-2018-8788\d1112c27"
            r"\include__freerdp__codec__nsc.h")


def test_flattened_and_slashed_paths_are_the_same_file():
    """压平写法与真斜杠写法必须归一化到同一个身份（这就是当年踩的坑）。"""
    assert file_identity(KB_FILE) == file_identity(FLAT_SAME)
    assert file_identity(KB_FILE) == file_identity(WIN_PATH)


def test_different_file_in_same_directory_is_not_the_same_file():
    """同目录下的**不同文件**不能被判成同一个文件。"""
    assert file_identity(KB_FILE) != file_identity(FLAT_OTHER)


def test_empty_path_has_empty_identity():
    assert file_identity("") == ""
    assert file_identity(None) == ""


def test_classify_same_file():
    info = classify({"sqlite_id": 54, "file": FLAT_SAME}, {FLAT_SAME}, {54: KB_FILE})
    assert info["kind"] == "same_file"


def test_classify_cross_file_when_kb_file_does_not_match():
    info = classify({"sqlite_id": 54, "file": FLAT_OTHER}, {FLAT_OTHER}, {54: KB_FILE})
    assert info["kind"] == "cross_file"


def test_classify_uses_run_files_not_only_finding_file():
    """判定要看"这一跑被分析的文件"，不能只看这条 finding 自己带的路径。"""
    info = classify({"sqlite_id": 54, "file": ""}, {FLAT_SAME}, {54: KB_FILE})
    assert info["kind"] == "same_file"


def test_mutation_breaking_normalization_flips_the_verdict():
    """变异对照：把"身份"改成只取裸文件名，负例就会误判成同文件。

    这条保证上面的断言**真的在管归一化**，而不是写了一堆恒真的断言。
    """
    def naive_identity(path: str) -> str:
        return str(path or "").replace("\\", "/").rsplit("/", 1)[-1].lower()

    # 朴素写法下，压平的路径取末段仍是 `include__freerdp__codec__nsc.h`，
    # 与 KB 的 `nsc.h` 对不上 → 会把同文件误判成跨文件（正是当年的现象）
    assert naive_identity(KB_FILE) != naive_identity(FLAT_SAME)
    assert file_identity(KB_FILE) == file_identity(FLAT_SAME)


if __name__ == "__main__":  # pragma: no cover
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
