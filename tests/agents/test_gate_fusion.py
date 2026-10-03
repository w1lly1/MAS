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


def _make_agent(fusion: bool = False, lam: float = 0.5, theta: float = 0.7,
                veto: str = "none"):
    """构造被测 agent。

    `veto` 默认给 "none"：本文件前半部分测的是**融合公式本身**（加不加否决是另一组用例），
    生产默认是 "same_file"（配置里的值），专门由下面那组"否决条件"用例覆盖。
    """
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *args, **kwargs: None
    agent.gate_fusion_enabled = fusion
    agent.gate_fusion_lambda = lam
    agent.gate_fusion_theta = theta
    agent.gate_fusion_z_scale = 4.0
    agent.gate_fusion_veto = veto
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


def test_semantic_term_uses_layer_table_for_lexical_channels():
    """**线上踩到的坑，钉成测试**：词法通道的候选自己没有相似度，必须靠"整层余弦表"补上。

    真实故障（2026-10-03 真机）：走到融合分支的 5089 条候选全是 `curated_issue`(5037) 与
    `sqlite`(52)，它们的 `semantic_score` 恒为 0 ⇒ 语义项恒为 0 ⇒ `fusion_score == s(x)`，
    融合**整个空转**。修法：整层检索（limit=stats_limit）本来就带距离，把它按 `sqlite_id`
    存成表，任何通道都能查到"它那条知识跟本次查询有多像"。
    """
    agent = _make_agent(fusion=True, lam=1.5, theta=0.7, veto="none")
    agent._fusion_layer_stats = {
        "code_pattern": {"mu": 0.20, "sigma": 0.05, "n": 200.0,
                         "sims_by_id": {77: 0.30}},          # z=2 → 语义项 0.5
    }
    # curated 通道、自己没有相似度，但表里有它那条**模式** ⇒ 语义项算得出来（修好的地方）
    c = _cand(channel="curated_issue", vector_layer="code_pattern", sqlite_id=11,
              kb_pattern_id=77, semantic_score=0.0, matched_fields=["basename_match"])
    assert abs(agent._fusion_semantic_term_of(c) - 0.5) < 1e-9
    # 表里没有它、自己也是 0 ⇒ 没有语义原料，记 0
    c2 = _cand(channel="curated_issue", vector_layer="code_pattern", sqlite_id=999,
               kb_pattern_id=999, semantic_score=0.0, matched_fields=["basename_match"])
    assert agent._fusion_semantic_term_of(c2) == 0.0
    # 层对不上（该候选的图层没有表）⇒ 仍能从"本次查过的其它层"拿到分（跨层取最大，与离线同口径）
    c3 = _cand(channel="curated_issue", vector_layer="full", sqlite_id=11,
               kb_pattern_id=77, semantic_score=0.0, matched_fields=["basename_match"])
    assert abs(agent._fusion_semantic_term_of(c3) - 0.5) < 1e-9
    # 表里查不到时，退回候选自己的 semantic_score（向量通道的老路径不能坏）
    c4 = _cand(channel="weaviate", vector_layer="code_pattern", sqlite_id=999,
               semantic_score=0.30, matched_fields=["basename_match"])
    assert abs(agent._fusion_semantic_term_of(c4) - 0.5) < 1e-9


def test_semantic_term_takes_max_across_queried_layers():
    """跨层取最大：离线 `sims[sid] = 跨层、跨块取最大`，线上要与它同口径。

    这条同时钉住"没层标签也能拿分"：结构化通道的候选本来就没有图层标签，
    若要求"必须有层"，等于把纯语义动作绑死在词法字段上（实现造成的假依赖）。
    """
    agent = _make_agent(fusion=True, lam=1.5, theta=0.7, veto="none")
    agent._fusion_layer_stats = {
        # 同一层：μ/σ 相同，77 的相似度 0.30 → z=2 → 0.5
        "semantic": {"mu": 0.20, "sigma": 0.05, "n": 200.0, "sims_by_id": {77: 0.30}},
        # 另一层：σ 更小 ⇒ 同一个 0.30 的相对分离更高（z=5 → 封顶 1.0）
        "solution": {"mu": 0.20, "sigma": 0.02, "n": 200.0, "sims_by_id": {77: 0.30}},
        # 第三层没有这条知识 ⇒ 不参与
        "full": {"mu": 0.20, "sigma": 0.05, "n": 200.0, "sims_by_id": {99: 0.25}},
    }
    cand = _cand(channel="curated_issue", vector_layer=None, sqlite_id=11,
                 kb_pattern_id=77, semantic_score=0.0, matched_fields=["basename_match"])
    assert agent._fusion_semantic_term_of(cand) == 1.0, "应取各层里最大的相对分"
    # 只有一层的表里有它 ⇒ 取那一层的分
    agent._fusion_layer_stats = {
        "semantic": {"mu": 0.20, "sigma": 0.05, "n": 200.0, "sims_by_id": {77: 0.30}},
        "full": {"mu": 0.20, "sigma": 0.05, "n": 200.0, "sims_by_id": {99: 0.25}},
    }
    assert abs(agent._fusion_semantic_term_of(cand) - 0.5) < 1e-9


def test_curated_candidate_must_use_pattern_id_not_instance_id():
    """**id 空间必须分开**：`curated_issues.id`（实例）≠ `issue_patterns.id`（模式）。

    真实事故（2026-10-03）：融合查相似度表时用了候选的 `sqlite_id`，而 `curated_issues` 通道的
    `sqlite_id` 是**实例 id**；索引与余弦表都按**模式 id** 编号，两张表 id 又从 1 开始
    ⇒ **静默地把另一条模式的相似度安在这条候选上**（不报错、数字看起来还挺合理）。
    修法：`curated_issue` 通道只认 `kb_pattern_id`；拿不到就**不给分**，绝不拿实例 id 顶替。
    """
    agent = _make_agent(fusion=True, lam=1.5, theta=0.7, veto="none")
    # 表按**模式 id** 编号：77 那条很像（sim 0.30 → z=2 → 语义项 0.5），11 那条不像（sim 0.20）
    agent._fusion_layer_stats = {
        "code_pattern": {"mu": 0.20, "sigma": 0.05, "n": 200.0,
                         "sims_by_id": {77: 0.30, 11: 0.20}},
    }
    # 实例 id=11、模式 id=77：必须拿到 **77** 的相似度（0.5），而不是 11 的（0.0）
    ok = _cand(channel="curated_issue", vector_layer="code_pattern",
               sqlite_id=11, kb_pattern_id=77, semantic_score=0.0,
               matched_fields=["basename_match"])
    assert abs(agent._fusion_semantic_term_of(ok) - 0.5) < 1e-9, "必须用模式 id 查表"
    # 反向对照：只有实例 id 在表里（模式 id 拿不到）⇒ **不给分**，不许拿 11 顶替
    bad = _cand(channel="curated_issue", vector_layer="code_pattern",
                sqlite_id=11, kb_pattern_id=None, semantic_score=0.0,
                matched_fields=["basename_match"])
    assert agent._fusion_semantic_term_of(bad) == 0.0, "实例 id 不得当作模式 id 使用"
    # issue_patterns 通道的 sqlite_id 本来就是模式 id ⇒ 照常可用
    pat = _cand(channel="sqlite", vector_layer="code_pattern", sqlite_id=77,
                semantic_score=0.0, matched_fields=["basename_match"])
    assert abs(agent._fusion_semantic_term_of(pat) - 0.5) < 1e-9


def test_fusion_stats_helper_returns_id_to_similarity_table():
    """整层检索的结果要留下 `sims_by_id`（融合能不能起作用全看它）。"""
    agent = _make_agent(fusion=True)

    class _Stub:
        def search_knowledge_items(self, **kwargs):
            return [{"sqlite_id": 11, "_additional": {"distance": 1.6}},   # sim 0.2
                    {"sqlite_id": 22, "_additional": {"distance": 1.0}},   # sim 0.5
                    {"sqlite_id": 33, "_additional": {"distance": 1.4}}]   # sim 0.3

    agent.vector_service = _Stub()
    stats = agent._fusion_layer_similarity_stats([0.1] * 8, "solution")
    assert stats["n"] == 3.0
    assert abs(stats["sims_by_id"][11] - 0.2) < 1e-9
    assert abs(stats["sims_by_id"][22] - 0.5) < 1e-9
    assert abs(stats["mu"] - 1.0 / 3) < 1e-9


def test_semantic_term_is_relative_and_never_negative():
    c = AIDrivenSecondPassAnalysisAgent._fuse_semantic_term
    assert abs(c(0.30, 0.20, 0.05, 4.0) - 0.5) < 1e-9   # z=2
    assert c(0.40, 0.20, 0.05, 4.0) == 1.0               # z=4 → 封顶
    assert c(0.60, 0.20, 0.05, 4.0) == 1.0               # 超过也封顶
    assert c(0.10, 0.20, 0.05, 4.0) == 0.0               # 低于均值 → 0（不出现负分）
    assert c(0.30, 0.20, 0.0, 4.0) == 0.0                # σ 退化 → 不给分
    assert c(0.30, None, 0.05, 4.0) == 0.0               # 缺统计 → 不给分


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


# --------------------------------------------------------------------------- #
# 3.5) 语义项的否决条件（安全阀）—— 这是本轮离线研究最重要的结论
# --------------------------------------------------------------------------- #
def test_veto_same_file_blocks_semantic_boost_without_same_file_evidence():
    """同文件否决：没有"同文件身份"证据的候选，语义项一律记 0。

    离线实测（`a5c_llm_veto_design.py`，45 样本）：没有否决时 λ≥1.0 会让跨文件误报 7~39 条；
    加上否决后 λ=1.5 仍然跨文件 0，正命中 25/30 → 29~30/30。
    """
    agent = _make_agent(fusion=True, lam=1.5, theta=0.7)
    agent.gate_fusion_veto = "same_file"
    agent._fusion_layer_stats = {"solution": {"mu": 0.0, "sigma": 0.01, "n": 200.0}}

    # 无锚点字段 ⇒ 否决
    assert agent._fusion_semantic_term_of(_cand(matched_fields=[])) == 0.0
    # 只有类名/函数名在码里（锚点但不是"同文件身份"）⇒ 也被同文件否决挡掉
    assert agent._fusion_semantic_term_of(
        _cand(matched_fields=["class_pattern_in_code"])) == 0.0
    # 有同文件身份 ⇒ 放行语义项（sim=1.0、μ=0、σ=0.01 → z 极大 → 封顶 1.0）
    assert agent._fusion_semantic_term_of(
        _cand(matched_fields=["file_basename_anchor"], semantic_score=1.0)) == 1.0


def test_veto_anchor_allows_anchored_but_not_bare_candidates():
    agent = _make_agent(fusion=True, lam=1.5, theta=0.7)
    agent.gate_fusion_veto = "anchor"
    agent._fusion_layer_stats = {"solution": {"mu": 0.0, "sigma": 0.01, "n": 200.0}}
    assert agent._fusion_semantic_term_of(
        _cand(matched_fields=["function_name_in_code"], semantic_score=1.0)) == 1.0
    assert agent._fusion_semantic_term_of(_cand(matched_fields=[], semantic_score=1.0)) == 0.0


def test_veto_unknown_value_falls_back_to_strictest():
    """配置写错值时按最严处理（不允许"写错了就悄悄放开"）。"""
    agent = _make_agent(fusion=True, lam=1.5, theta=0.7)
    agent.gate_fusion_veto = "typo_value"
    agent._fusion_layer_stats = {"solution": {"mu": 0.0, "sigma": 0.01, "n": 200.0}}
    assert agent._fusion_semantic_term_of(_cand(matched_fields=[], semantic_score=1.0)) == 0.0
    assert agent._fusion_semantic_term_of(
        _cand(matched_fields=["file_basename_anchor"], semantic_score=1.0)) == 1.0


def test_veto_none_is_the_unsafe_control():
    """`none` 只用于实验对照：此时"有锚点但不是同文件"的候选也能吃到满语义加分。

    这正是"不否决就危险"的地方 —— 同一个候选，换个 veto 就从放行变成拒掉。
    （注：完全无锚点的候选在更前面的 weaviate 守卫就被拦了，所以这里用"有函数名锚点、
    但不是同文件身份"的候选来演示差异。）
    """
    kw = dict(matched_fields=["function_name_in_code"], semantic_score=1.0)
    stats = {"solution": {"mu": 0.0, "sigma": 0.01, "n": 200.0}}

    unsafe = _make_agent(fusion=True, lam=1.5, theta=0.7, veto="none")
    unsafe._fusion_layer_stats = dict(stats)
    c1 = _cand(**kw)
    unsafe._gate_candidate(c1)
    assert c1["gating_decision"] == "formal_hit"        # 语义单飞（必须避免的用法）
    assert c1["gate_branch"] == "fused_semantic"

    safe = _make_agent(fusion=True, lam=1.5, theta=0.7, veto="same_file")
    safe._fusion_layer_stats = dict(stats)
    c2 = _cand(**kw)
    safe._gate_candidate(c2)
    assert c2["gating_decision"] != "formal_hit"        # 被否决 → 回到它本该有的判定


def test_enabling_fusion_never_downgrades_a_decision():
    """开启开关只能**多放行**，不能把原本放行的候选降级（结构性不变量）。

    为什么要有它：③ 的预登记断言是"s(x)+λ·s_sem(x) ≥ θ 这条**辅助**分支只加召回"。
    如果实现上不小心把旧的 explanatory 分支挪到融合分支后面、或让融合分支抢先 return，
    就可能出现"开了开关反而少放行"这种极难在批次结果里察觉的倒退 —— 这里用穷举钉住。
    """
    fields_pool = [[], ["error_code_clone"], ["class_pattern_in_code"],
                   ["error_code_clone", "class_pattern_in_code"], ["file_basename_anchor"]]
    stats = {"solution": {"mu": 0.20, "sigma": 0.05, "n": 200.0}}
    for ld in (0.5, 1.0, 3.0):
        on_agent = _make_agent(fusion=True, lam=ld, theta=0.7)
        off_agent = _make_agent(fusion=False)
        for fields in fields_pool:
            for sem in (0.0, 0.3, 0.9):
                for anchor in (0.0, 0.4):
                    kw = dict(matched_fields=list(fields), semantic_score=sem, anchor_score=anchor)
                    on = _cand(**kw)
                    off = _cand(**kw)
                    on_agent._fusion_layer_stats = dict(stats)
                    on_agent._gate_candidate(on)
                    off_agent._gate_candidate(off)
                    admit = {"formal_hit", "explanatory_hit"}
                    assert not (off["gating_decision"] in admit
                                and on["gating_decision"] not in admit), (
                        fields, sem, anchor, ld, off["gating_decision"], on["gating_decision"])


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
