"""门控"加法融合"（A5b 结论的落地形态）测试。

被测的逻辑只有一句话：

    score(x) = s(x) + λ·s_sem(x)，   admit ⇔ F(x) ∧ score(x) ≥ θ

三条必须成立的硬性质（本文件的骨架）：

1. **默认关闭时判定逐条不变** —— 整块逻辑被 `gate_fusion_enabled` 包住，
   关闭时既不产生新放行，也不往报告里塞新字段、也不发额外查询；
2. **开关是有效的**（同一条候选，开/关得到不同判定）—— 否则"开关"只是个摆设
   （变异检验会专门打这一点）；
3. **F(x) 的硬守卫不因融合而失效** —— 跨文件、已修复、弱结构无锚点这三道守卫
   都必须先用开关打开也拦得住（用户明确要求：不放开跨文件闸门）。
"""
from __future__ import annotations

from core.agents.ai_driven_second_pass_analysis_agent import (
    AIDrivenSecondPassAnalysisAgent,
)


def _make_agent(fusion: bool = False, lam: float = 0.5, theta: float = 0.7):
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *args, **kwargs: None
    agent.gate_fusion_enabled = fusion
    agent.gate_fusion_lambda = lam
    agent.gate_fusion_theta = theta
    agent.gate_fusion_z_scale = 4.0
    return agent


def _cand(**kw):
    """同文件、非泛化 error_type 的 weaviate 候选（避开跨文件守卫的干扰）。"""
    base = {
        "channel": "weaviate",
        "vector_layer": "solution",
        "error_type": "out_of_bounds",
        "structured_score": 0.0,
        "semantic_score": 0.3,
        "context_score": 0.0,
        "anchor_score": 0.0,
        "matched_fields": [],
        "file_pattern": "net/core/scm.c",
        "_analysis_file": "net/core/scm.c",
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# 1) 默认关闭：判定与历史实现一致（按规格手写期望，不是抄实现）
# --------------------------------------------------------------------------- #
def test_disabled_matches_historical_dnf_decisions():
    cases = [
        # (matched_fields, semantic, anchor, 期望判定)
        (["error_code_clone"], 0.30, 0.00, "discarded_hit"),            # s=0.50 < 0.65
        (["error_code_clone", "class_pattern_in_code"], 0.30, 0.00, "formal_hit"),  # s=0.75
        (["file_basename_anchor"], 0.90, 0.40, "explanatory_hit"),      # s=0.20≥θ_w 且 v≥τ 且 a≥θ_a
        ([], 0.90, 0.40, "low_confidence_hit"),                         # s=0 弱结构 + 高语义
        (["class_pattern_in_code"], 0.50, 0.10, "discarded_hit"),       # 什么都不够
    ]
    agent = _make_agent(fusion=False)
    for fields, sem, anchor, expect in cases:
        cand = _cand(matched_fields=list(fields), semantic_score=sem, anchor_score=anchor)
        agent._gate_candidate(cand)
        assert cand["gating_decision"] == expect, (fields, sem, anchor, cand["gating_decision"])


def test_disabled_adds_no_fields_and_no_extra_query():
    agent = _make_agent(fusion=False)
    cand = _cand(matched_fields=["error_code_clone"])
    agent._gate_candidate(cand)
    assert "fusion_score" not in cand and "fusion_semantic_term" not in cand
    assert "lambda" not in cand["gate_params"]
    assert cand["gate_formula"] == (
        "admit = F(x) & ( s(x)>=theta_s | ( v(x)>=tau & a(x)>=theta_a & s(x)>=theta_w ) )"
    )

    calls = {"n": 0}

    class _Stub:
        def search_knowledge_items(self, **kwargs):  # pragma: no cover - 不应被调用
            calls["n"] += 1
            return []

    agent.vector_service = _Stub()
    cand2 = _cand(matched_fields=["error_code_clone"], semantic_score=0.3)
    agent._gate_candidate(cand2)
    assert calls["n"] == 0, "关闭融合时不该为分布统计发任何查询"


# --------------------------------------------------------------------------- #
# 2) 开关有效：同一条候选，开/关结论不同（否则开关是摆设）
# --------------------------------------------------------------------------- #
def test_enabled_semantic_term_can_promote_candidate():
    # s=0.50（只有 error_code_clone 一项），语义项 0.5 ⇒ 0.50+0.5*0.5=0.75 ≥ 0.70
    cand = _cand(matched_fields=["error_code_clone"], semantic_score=0.30)
    agent = _make_agent(fusion=True, lam=0.5, theta=0.7)
    agent._fusion_layer_stats = {"solution": {"mu": 0.20, "sigma": 0.05, "n": 200.0}}
    agent._gate_candidate(cand)
    assert cand["fusion_semantic_term"] == 0.5      # z=(0.30-0.20)/0.05=2 → 2/4
    assert cand["fusion_score"] == 0.75
    assert cand["gating_decision"] == "formal_hit"
    assert cand["gate_branch"] == "fused_semantic"

    # 同一条候选 + 关闭开关 ⇒ 回到"不放行"
    off = _cand(matched_fields=["error_code_clone"], semantic_score=0.30)
    _make_agent(fusion=False)._gate_candidate(off)
    assert off["gating_decision"] == "discarded_hit"


def test_threshold_and_lambda_are_load_bearing():
    for theta, expect in ((0.7, "formal_hit"), (0.76, "discarded_hit")):
        cand = _cand(matched_fields=["error_code_clone"], semantic_score=0.30)
        agent = _make_agent(fusion=True, lam=0.5, theta=theta)
        agent._fusion_layer_stats = {"solution": {"mu": 0.20, "sigma": 0.05, "n": 200.0}}
        agent._gate_candidate(cand)
        assert cand["gating_decision"] == expect, theta
    # λ=0 ⇒ 语义项不参与（融合退化成"纯阈值"，且比 θ_s 更松时也不该放行）
    cand = _cand(matched_fields=["error_code_clone"], semantic_score=0.30)
    agent = _make_agent(fusion=True, lam=0.0, theta=0.7)
    agent._fusion_layer_stats = {"solution": {"mu": 0.20, "sigma": 0.05, "n": 200.0}}
    agent._gate_candidate(cand)
    assert cand["gating_decision"] == "discarded_hit"


def test_semantic_term_is_relative_and_only_for_weaviate():
    c = AIDrivenSecondPassAnalysisAgent._fuse_semantic_term
    assert abs(c(0.30, 0.20, 0.05, 4.0) - 0.5) < 1e-9   # z=2
    assert c(0.40, 0.20, 0.05, 4.0) == 1.0               # z=4 → 封顶
    assert c(0.60, 0.20, 0.05, 4.0) == 1.0               # 超过也封顶
    assert c(0.10, 0.20, 0.05, 4.0) == 0.0               # 低于均值 → 0（不出现负分）
    assert c(0.30, 0.20, 0.0, 4.0) == 0.0                # σ 退化 → 不给分
    assert c(0.30, None, 0.05, 4.0) == 0.0               # 缺统计 → 不给分

    agent = _make_agent(fusion=True)
    agent._fusion_layer_stats = {"solution": {"mu": 0.20, "sigma": 0.05, "n": 200.0}}
    # 非 weaviate 通道没有可比的相似度（其 semantic_score 恒为 0）⇒ 语义项恒为 0
    assert agent._fusion_semantic_term_of(_cand(channel="curated_issue")) == 0.0
    # 层对不上（没有该层的分布）⇒ 也返回 0，不猜
    assert agent._fusion_semantic_term_of(_cand(vector_layer="full")) == 0.0


# --------------------------------------------------------------------------- #
# 3) F(x) 的硬守卫不因融合而失效
# --------------------------------------------------------------------------- #
def test_fusion_cannot_open_cross_file_gate():
    """跨文件 + 无代码锚点的候选，即使语义项拉满也必须被拦下。"""
    agent = _make_agent(fusion=True, lam=3.0, theta=0.5)   # 极端参数
    agent._fusion_layer_stats = {"solution": {"mu": 0.0, "sigma": 0.01, "n": 200.0}}
    cand = _cand(
        matched_fields=["error_code_clone"],                # s=0.5
        semantic_score=1.0,                                 # 语义项会被封顶到 1.0
        file_pattern="net/netlabel/netlabel_cipso_v4.c",    # 知识条目属于别的文件
        _analysis_file="arch/powerpc/kernel/traps.c",
    )
    agent._gate_candidate(cand)
    assert cand["gating_decision"] == "discarded_hit"
    assert cand["rejection_reason"] == "cross_file_mismatch"


def test_fusion_cannot_revive_weak_structure_no_anchor():
    """weaviate 通道"弱结构 + 无文件锚点"的守卫同样不被融合绕过。"""
    agent = _make_agent(fusion=True, lam=3.0, theta=0.3)
    agent._fusion_layer_stats = {"solution": {"mu": 0.0, "sigma": 0.01, "n": 200.0}}
    cand = _cand(
        matched_fields=[],                 # s=0 ⇒ unified_s < gate_weak_structure_threshold
        semantic_score=1.0,
        file_pattern="net/netlabel/netlabel_cipso_v4.c",
        _analysis_file="net/netlabel/netlabel_cipso_v4.c",   # 同文件，避开上一道守卫
    )
    agent._gate_candidate(cand)
    assert cand["gating_decision"] in {"discarded_hit", "low_confidence_hit"}
    assert cand["rejection_reason"] in {"low_confidence_or_generic", "weak_structure_no_file_anchor"}


def test_fusion_stats_helper_computes_mu_sigma_and_degrades_safely():
    agent = _make_agent(fusion=True)
    agent.gate_fusion_stats_limit = 200

    class _Stub:
        def __init__(self, rows):
            self.rows = rows
            self.calls = []

        def search_knowledge_items(self, **kwargs):
            self.calls.append(kwargs)
            return self.rows

    rows = [{"_additional": {"distance": d}} for d in (1.6, 1.5, 1.4, 1.0)]
    stub = _Stub(rows)
    agent.vector_service = stub
    stats = agent._fusion_layer_similarity_stats([0.1] * 8, "full")
    assert stub.calls and stub.calls[0]["layer"] == "full"
    assert stub.calls[0]["limit"] == 200
    assert stats["n"] == 4.0
    # similarity = 1 - distance/2 → [0.2, 0.25, 0.3, 0.5]，均值 0.3125
    assert abs(stats["mu"] - 0.3125) < 1e-9
    assert stats["sigma"] > 0

    # 少于两条 / 查询失败 ⇒ 返回空，绝不抛异常（融合是加分项，不该拖垮链路）
    agent.vector_service = _Stub([{"_additional": {"distance": 1.0}}])
    assert agent._fusion_layer_similarity_stats([0.0], "full") == {}
    agent.vector_service = _Stub([{"_additional": {}}])       # 没有 distance 字段
    assert agent._fusion_layer_similarity_stats([0.0], "full") == {}

    class _Boom:
        def search_knowledge_items(self, **kwargs):
            raise RuntimeError("weaviate down")

    agent.vector_service = _Boom()
    assert agent._fusion_layer_similarity_stats([0.0], "full") == {}
