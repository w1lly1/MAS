# -*- coding: utf-8 -*-
"""语义描述契约（`utils/semantic_contract.py`）与它在两处的接线测试。

## 这一组锁住的事故

历史上首轮安全分析的产出是 `生成威胁文本中提及关键词: inject, leak` ——
**没有行号、没有描述**，既不能定位、也不能用来检索（而且它是中文，进不了英文编码器的空间）。
实测 8 个样本**全部**落到这个关键词兜底：模型其实**生成过**一段威胁分析文本，
但 JSON 解析失败后就把它丢掉了。

离线实验量化过代价：索引侧是英文散文、查询侧是"原始代码 + 中文标签"时，
正确条目只有 **47%** 进 top-5；换成同语言同语域的英文描述后到 **81%**
（见《02》第十六节）。所以"契约解析"这一步值得单独钉住。

## 三件事必须成立

1. **契约解析**：两行格式（`error_type: <7类>` + 两句英文）能被正确解析，标签被剥掉；
2. **接线**：安全代理解析出 `llm_semantic`/`llm_family` 并挂到 issue 上；
   检索查询文本**优先**用它，且**没有时行为与以前完全一致**（fail-open）；
3. **两份实现不许跑偏**：契约逻辑在 `utils/semantic_contract.py` 与
   `utils/experiments/gen_llm_semantic.py` 各有一份（脚本是独立可跑的），
   这里用同一个输入集合交叉比对——**任一处分歧就失败**。
   本项目已有先例：层文本构造器两份实现跑偏，导致 file/class 字段恒为空。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from utils.semantic_contract import (  # noqa: E402
    FAMILIES, FAMILY_LIST, clean_description, compliance, is_acceptable, parse_contract,
    split_family,
)

RAW_REPLIES = [
    # 标准契约
    "error_type: memory_overflow\nThe function f2fs_balance_fs indexes the segment array "
    "with an unchecked value. A negative index may read out of bounds.",
    # 句中标了标签（实测 14 条里 4 条会这样）
    "This code initializes a mux table. Sentence B: Unchecked access to the table may "
    "cause an out-of-bounds write.",
    # 开头标了标签
    "Sentence 1: The driver parses the header. Sentence 2: Missing length validation may "
    "allow a buffer overread.",
    # 家族非法 → 应判为空、但正文仍可用
    "error_type: not_a_real_family\nThe parser reads a length. An unchecked length may "
    "cause an out-of-bounds read.",
    # 含中文 → 必须被拒
    "error_type: dos\nThe function loops forever for a未知的 input. It may hang the thread.",
    # 家族行缺失
    "The routine copies the packet. Missing bounds checks may overflow the destination buffer.",
]


def test_parse_contract_extracts_family_and_strips_labels():
    for raw in RAW_REPLIES:
        fam, desc = parse_contract(raw)
        assert fam in FAMILIES or fam == "", raw
        assert "sentence" not in desc.lower(), desc
        assert "error_type" not in desc.lower(), desc


def test_family_must_be_one_of_seven():
    """非法家族**不采纳**，但那一行仍然要被剥掉。

    为什么不把非法值留在正文里：它是个格式错误的标签行，留在正文里只会污染待入库的文本
    （而这段文本会被编码进向量）。所以处理是"**行必删、值可不认**"。
    """
    fam, desc = split_family("error_type: totally_made_up\nSome text here.")
    assert fam == ""                        # 非法值不采纳
    assert "totally_made_up" not in desc    # 标签行也不能留在正文里
    assert "Some text here." in desc        # 正文本身不能丢
    for f in FAMILIES:
        assert split_family("error_type: %s\nSome text." % f)[0] == f


def test_clean_description_keeps_the_risk_sentence():
    """**关键**：模型常把前两句都写成"在做什么"，风险句被推到第三句。

    直接截前两句会把风险信息整个丢掉（这是第一版的真实 bug）。这里构造一个
    "前两句都在描述功能、第三句才是风险"的回复，要求保留下来的必须含风险句。
    """
    raw = ("error_type: general\n"
           "The function walks the list of segments. It also updates the counters for each "
           "entry. Missing bounds checks may cause an out-of-bounds access.")
    _fam, desc = parse_contract(raw)
    assert "out-of-bounds" in desc, desc
    assert len([s for s in desc.split(".") if s.strip()]) <= 2


def test_cjk_and_meta_leaks_are_rejected():
    assert not is_acceptable("The function loops for a未知 input. It may hang.")
    assert not is_acceptable("A bug in CVE-2018-20854 caused an overflow. It may crash.")
    assert not is_acceptable("Fixed in kernel before 4.20. It may crash the driver.")
    assert not is_acceptable("See drivers/phy/mscc/phy-ocelot-serdes.c for details. It may crash.")
    assert compliance("ok short text")["words"] < 20


def test_risk_cue_present_in_accepted_text():
    _fam, desc = parse_contract(RAW_REPLIES[0])
    assert compliance(desc)["risk_cue"] is True
    assert is_acceptable(desc, "x.c")


def test_two_implementations_agree():
    """契约逻辑不允许有两套行为：脚本里那份与共享模块必须逐例一致。"""
    from utils.experiments import gen_llm_semantic as gen  # noqa: PLC0415

    for raw in RAW_REPLIES:
        assert gen.split_family(raw) == split_family(raw), raw
        assert gen.clean_reply(raw) == clean_description(raw), raw
    # 家族清单也必须同源（模型看到的是同一套分类，否则两侧对不上号）
    assert set(FAMILIES) == set(gen.FAMILIES), (set(FAMILIES) ^ set(gen.FAMILIES))
    assert FAMILY_LIST.strip() == gen.FAMILY_LIST.strip()


def test_security_agent_returns_semantic_fields_instead_of_keyword_tag():
    """安全代理：契约形状的回复必须产出 `llm_semantic`，而不是关键词标签。"""
    from core.agents.ai_driven_security_agent import AIDrivenSecurityAgent

    agent = AIDrivenSecurityAgent()
    reply = [{"generated_text": RAW_REPLIES[0]}]
    out = asyncio.run(agent._extract_vulnerability_details(reply, "int f(void){}", 0))
    assert out is not None
    assert out["source"] == "semantic_contract"
    assert out["llm_family"] == "memory_overflow"
    assert "unchecked" in out["llm_semantic"].lower()
    assert "提及关键词" not in out["description"]


def test_security_agent_keeps_keyword_fallback_for_legacy_text():
    """向后兼容：老的、没有契约形状的回复仍走关键词兜底，不能直接报错。"""
    from core.agents.ai_driven_security_agent import AIDrivenSecurityAgent

    agent = AIDrivenSecurityAgent()
    legacy = [{"generated_text": "This code may have an sql injection and a memory leak."}]
    out = asyncio.run(agent._extract_vulnerability_details(legacy, "x", 1))
    assert out is not None
    assert out["type"] == "generated_threat_indicator"
    assert out.get("llm_semantic") in (None, "")


def _fresh_agent():
    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    a = AIDrivenSecondPassAnalysisAgent()
    a._debug_log = lambda *args, **kwargs: None
    return a


def test_query_text_prefers_semantic_description():
    """检索查询文本优先用语义描述：带上家族、不再塞原始代码片段。"""
    agent = _fresh_agent()
    issue = {
        "source": "source_code_chunk", "file": "drivers/phy/x.c", "severity": "medium",
        "description": "source_code_chunk L1-40: <raw code here>",
        "code_snippet": "for (i = 0; i <= MAX; i++) { arr[i] = 0; }",
        "llm_semantic": "The function indexes the array with an unchecked bound. A large "
                        "value may write out of bounds.",
        "llm_family": "memory_overflow",
    }
    text = agent._build_query_text(issue, issue["file"])
    assert "error_type: memory_overflow" in text
    assert "unchecked bound" in text
    assert "snippet:" not in text, "用了语义描述时不该再塞原始代码片段"


def test_query_text_unchanged_when_semantic_absent():
    """没有语义描述时，查询文本必须与历史实现**逐字节相同**（fail-open）。"""
    agent = _fresh_agent()
    issue = {
        "source": "source_code_chunk", "file": "a/b.c", "severity": "medium",
        "description": "source_code_chunk L1-10: code", "code_snippet": "int x;",
        "line_number": 3,
    }
    expected = " | ".join([
        "[source_code_chunk] source_code_chunk L1-10: code",
        "line_number:3", "severity:medium", "file:b.c", "ext:c",
        "snippet:int x;", "sig:%s" % agent._semantic_signature(issue),
    ])
    assert agent._build_query_text(issue, issue["file"]) == expected
