# -*- coding: utf-8 -*-
"""知识库分类规则（`utils/bigvul_ingest/rules.py`）的回归测试。

## 为什么要有这一组

`error_type` 不只是个标签：索引侧 `semantic` 层写着 `[error_type] <值>`，
`code_pattern` 层那句"模式描述"（`problematic_pattern`）也是**按它选的**。
所以分类错了，那一层的文本会跟着变弱 —— 它直接决定检索质量。

## 被这些用例锁住的具体事故

1. **连字符漏匹配**：原规则在原文里找 `"out of bounds"`（空格），
   而 CVE 摘要写的是 `"out-of-bounds"`（连字符）→ 一条明确的越界读被归到 `general`。
   实测 `CVE-2018-20854` 就是这样判错的，而它也直接导致那一层的模式描述变成泛泛的话。
2. **影响盖过机制**：摘要里常有 `denial of service`，但根因是内存越界。
   分类应该描述**机制**，否则模式描述会写成与代码不符的话。
   所以 CWE 优先，且"内存/校验/竞态/权限"排在 `dos` 前面。
3. **过宽的关键词**：原规则用裸 `"input"` 判输入校验，会把 "reads the input file"
   这类无关描述也吞进去。
4. **重写模块时漏删/漏留函数**：本轮重写 `rules.py` 时把 `score_to_severity` 弄丢了，
   整个 ingest 直接 import 失败。这里用一个"公开接口仍在"的用例把它钉住。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.bigvul_ingest.build_structured_ingest import _build_pattern  # noqa: E402
from utils.bigvul_ingest.rules import (  # noqa: E402
    VALID_ERROR_TYPES, derive_error_type, derive_problematic_pattern, normalize_for_match,
    score_to_severity,
)


def test_normalize_collapses_hyphens_underscores_and_case():
    """归一化：连字符 / 下划线 / 斜杠 / 多空格 / 大小写都必须落到同一写法。"""
    want = "out of bounds"
    for raw in ("out-of-bounds", "out_of_bounds", "out of  bounds", "OUT-OF-BOUNDS",
                "out/of/bounds", "  out - of - bounds  "):
        assert normalize_for_match(raw) == want, raw


def test_hyphenated_out_of_bounds_is_memory_overflow():
    """**核心事故回归**：带连字符的越界描述必须判成内存越界，而不是 general。

    这条摘要就是 CVE-2018-20854 的原文（此前被判成 general）。
    """
    summary = ("An issue was discovered in the Linux kernel before 4.20. "
               "drivers/phy/mscc/phy-ocelot-serdes.c has an off-by-one error with a "
               "resultant ctrl->phys out-of-bounds read.")
    assert derive_error_type("", "", summary) == "memory_overflow"
    # 连字符换成下划线、空格、大写，结论必须一样
    for variant in (summary.replace("out-of-bounds", "out_of_bounds"),
                    summary.replace("out-of-bounds", "OUT OF BOUNDS")):
        assert derive_error_type("", "", variant) == "memory_overflow", variant


def test_cwe_is_authoritative_over_keywords():
    """CWE 能对上就直接采信，不再看关键词。"""
    assert derive_error_type("CWE-362", "", "a denial of service in the parser") == "race_condition"
    assert derive_error_type("CWE-787", "", "denial of service") == "memory_overflow"
    assert derive_error_type("CWE-20", "", "denial of service") == "input_validation"


def test_mechanism_beats_impact():
    """摘要写的是**影响**（DoS），CWE 或代码指向的是**机制** → 以机制为准。

    实测：`dos` 在旧规则下占 33%，修好后降到 14% —— 大部分是这类"影响盖机制"的误标。
    """
    assert derive_error_type("CWE-125", "", "allows local users to cause a denial of service") \
        == "memory_overflow"
    assert derive_error_type("", "", "a buffer overflow leads to a denial of service") \
        == "memory_overflow"


def test_bare_input_word_does_not_trigger_input_validation():
    """裸 "input" 不再算输入校验（它太宽，会误吞大量条目）。"""
    assert derive_error_type("", "", "the driver reads the input file at startup") == "general"
    assert derive_error_type("", "", "input validation is missing for the packet length") \
        == "input_validation"
    assert derive_error_type("", "", "the packet length is not validated") == "input_validation"


def test_resource_exhaustion_needs_leak_or_exhaustion_words():
    assert derive_error_type("", "", "a memory leak in the connection handler") \
        == "resource_exhaustion"
    assert derive_error_type("", "", "fails to release the allocated buffer") \
        == "resource_exhaustion"
    # 只提到 "resource" 不算
    assert derive_error_type("", "", "the resource manager is initialized") == "general"


def test_both_derived_fields_use_the_same_family():
    """分类与模式描述必须**同源**：模式描述是按家族选句子的。"""
    for fam in VALID_ERROR_TYPES:
        sentence = derive_problematic_pattern(fam, "some evidence")
        assert sentence.endswith("Evidence: some evidence")
        assert len(sentence) > len("Evidence: some evidence")
    assert derive_problematic_pattern("memory_overflow", "e") != \
        derive_problematic_pattern("general", "e")


def test_llm_family_overrides_and_records_provenance():
    """旁挂的模型家族优先，并在 tags 里留痕（便于审计哪些条目被模型改过分类）。"""
    meta = {"cve_id": "CVE-X", "cwe_id": "CWE-476", "summary": "a denial of service",
            "score": "5.5", "lang": "C", "project": "linux"}

    by_rules = _build_pattern(meta, file_pattern="a/b.c")
    assert by_rules["error_type"] == "dos"
    assert "error_type_source=rules" in by_rules["tags"]
    assert "rule_family=dos" in by_rules["tags"]

    by_llm = _build_pattern(meta, file_pattern="a/b.c",
                            llm_semantic="It walks a list. Missing limits may loop forever.",
                            llm_family="resource_exhaustion")
    assert by_llm["error_type"] == "resource_exhaustion"
    assert "error_type_source=llm" in by_llm["tags"]
    # 关键：模式描述句必须跟着换成新家族那句，不能还是旧家族的话
    assert by_llm["problematic_pattern"].startswith(
        derive_problematic_pattern("resource_exhaustion", "").split(" Evidence:")[0])
    assert by_llm["llm_semantic"].startswith("It walks a list")

    # 非法家族必须回退到规则值，不能把脏值写进库
    bad = _build_pattern(meta, file_pattern="a/b.c", llm_family="totally_made_up")
    assert bad["error_type"] == "dos"
    assert "error_type_source=rules" in bad["tags"]


def test_public_api_still_exported():
    """公开接口仍在（重写模块时漏删函数会直接让 ingest import 失败）。"""
    assert callable(score_to_severity)
    assert score_to_severity("9.5") == "critical"
    assert score_to_severity("7.2") == "high"
    assert score_to_severity("5.0") == "medium"
    assert score_to_severity("2.0") == "low"
    assert score_to_severity("not a number") == "medium"
    assert len(VALID_ERROR_TYPES) == 7
