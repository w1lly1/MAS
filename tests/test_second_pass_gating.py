from __future__ import annotations

import os

from core.agents.ai_driven_second_pass_analysis_agent import (
    AIDrivenSecondPassAnalysisAgent,
)


def _make_agent() -> AIDrivenSecondPassAnalysisAgent:
    """构造被测 agent。

    直接走真实 __init__（只读配置 + 建 DAO，不连接 Weaviate），
    以免手工罗列属性与实现漂移——历史上此处用 __new__ 手工赋值，
    漏掉统一门控引入的 gate_structured/anchor/weak_structure 三个阈值，
    导致 7 个门控用例在基线就是 AttributeError（测试网长期失效）。
    """
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *args, **kwargs: None
    return agent


def test_weaviate_semantic_anchor_promotes_hit():
    agent = _make_agent()

    candidate = {
        "channel": "weaviate",
        "vector_layer": "code_pattern",
        "error_type": "dos",
        "structured_score": 0.25,
        "semantic_score": 0.92,
        "context_score": 0.1,
        "anchor_score": 0.4,
        "matched_fields": ["class_pattern_in_code"],
        "_analysis_file": "arch/powerpc/kernel/traps.c",
        "file_pattern": "arch/powerpc/kernel/traps.c",
    }

    agent._gate_candidate(candidate)

    assert candidate["gating_decision"] == "explanatory_hit"
    assert candidate["layer_bonus"] == 0.03


def test_cross_file_weaviate_hit_discarded():
    agent = _make_agent()
    candidate = {
        "channel": "weaviate",
        "vector_layer": "semantic",
        "error_type": "memory_overflow",
        "structured_score": 0.0,
        "semantic_score": 1.0,
        "context_score": 0.1,
        "anchor_score": 0.55,
        "matched_fields": [],
        "file_pattern": "net/netlabel/netlabel_cipso_v4.c",
        "error_description": "off-by-one bug in net/netlabel/netlabel_cipso_v4.c",
        "_analysis_file": "arch/powerpc/kernel/traps.c",
    }
    agent._gate_candidate(candidate)
    assert candidate["gating_decision"] == "discarded_hit"
    assert candidate["rejection_reason"] == "cross_file_mismatch"


def test_bigvul_flattened_path_not_cross_file():
    """BigVul 扁平文件名应与知识库路径 basename 对齐，不得误判 cross_file。"""
    agent = _make_agent()
    candidate = {
        "channel": "weaviate",
        "vector_layer": "solution",
        "error_type": "dos",
        "structured_score": 0.0,
        "semantic_score": 1.0,
        "context_score": 0.1,
        "anchor_score": 0.55,
        "matched_fields": [],
        "file_pattern": "src/kadmin/server/schpw.c",
        "error_description": "schpw.c in the kpasswd service ...",
        "_analysis_file": r"E:\tests\BigVul\before\CVE-2002-2443\cf1a0c41\src__kadmin__server__schpw.c",
    }
    agent._apply_file_function_anchors(
        candidate,
        {
            "description": "source_code_chunk L38-80: goto chpwfail",
            "code_snippet": "goto chpwfail;",
        },
        candidate["_analysis_file"],
    )
    agent._gate_candidate(candidate)
    assert candidate.get("rejection_reason") != "cross_file_mismatch"
    assert "file_basename_anchor" in candidate["matched_fields"]
    assert candidate["gating_decision"] in {"explanatory_hit", "formal_hit", "low_confidence_hit"}


def test_curated_line_coincidence_alone_does_not_promote():
    """行号落在 gap chunk 区间内【不得】作为晋升依据。

    历史上 _match_curated_issue 对 line_in_curated_range 给 +0.4 并据此晋升，
    等价于"同文件同行号自查"（循环论证，见 认知记录 问题1）。
    现主匹配键已改为错误代码克隆：无克隆命中时 structured_score 封顶 0.4 < 0.45，
    因此行号重合单独出现时必须拒绝。本用例即锁定该不变量。
    """
    agent = _make_agent()
    issue = {
        "file": r"before\CVE-2002-2443\cf1a0c41\src__kadmin__server__schpw.c",
        "line": 38,
        "chunk_start_line": 38,
        "chunk_end_line": 80,
        "description": "source_code_chunk L38-80: goto chpwfail UDP packet",
    }
    curated = {
        "file_path": "src/kadmin/server/schpw.c",
        "start_line": 55,
        "end_line": 55,
        "problem_phenomenon": "schpw.c UDP packet triggers communication loop",
        "root_cause": "schpw.c improper validation of UDP packets",
    }
    match = agent._match_curated_issue(curated, issue, issue["file"])
    # 不再以行号区间作为证据字段
    assert "line_in_curated_range" not in match["matched_fields"]
    # basename 相同仍记录为辅助证据，但无克隆时不足以晋升
    assert "basename_match" in match["matched_fields"]
    assert match["structured_score"] < 0.45
    assert match["matched"] is False


def test_normalize_source_basename():
    agent = _make_agent()
    assert agent._normalize_source_basename("src__kadmin__server__schpw.c") == "schpw.c"
    assert agent._normalize_source_basename("src/kadmin/server/schpw.c") == "schpw.c"
    assert agent._normalize_source_basename("arch__powerpc__kernel__traps.c") == "traps.c"
    assert agent._normalize_source_basename("include__asm-ia64__ptrace.h") == "ptrace.h"


def test_weak_structure_semantic_only_is_low_confidence():
    agent = _make_agent()
    candidate = {
        "channel": "weaviate",
        "vector_layer": "semantic",
        "error_type": "input_validation",
        "structured_score": 0.0,
        "semantic_score": 1.0,
        "context_score": 0.1,
        "anchor_score": 0.55,
        "matched_fields": [],
        "error_description": "Irssi before 0.8.15 does not verify hostname",
        "_analysis_file": "arch/powerpc/kernel/traps.c",
    }
    agent._gate_candidate(candidate)
    assert candidate["gating_decision"] == "low_confidence_hit"
    assert candidate["rejection_reason"] == "weak_structure_no_file_anchor"


def test_curated_without_error_code_clone_rejected():
    """跨文件的 curated 命中：无错误代码克隆时不得晋升。

    拒绝原因字面量随"克隆为主匹配键"的重构由 curated_missing_basename
    改为 curated_no_error_code_clone；本用例锁定"不晋升"这一不变量。
    """
    agent = _make_agent()
    issue = {
        "file": "arch/powerpc/kernel/traps.c",
        "line": 332,
        "description": "source_code_chunk about machine check",
    }
    curated = {
        "file_path": "drivers/media/video/v4l2-ioctl.c",
        "start_line": 300,
        "end_line": 400,
        "problem_phenomenon": "The video_usercopy function mishandles buffers",
        "root_cause": "The video_usercopy function mishandles buffers",
    }
    match = agent._match_curated_issue(curated, issue, issue["file"])
    assert match["matched"] is False
    assert match.get("rejection_reason") == "curated_no_error_code_clone"


def test_demote_unanchored_severity():
    agent = _make_agent()
    hit = {"structured_score": 0.0, "matched_fields": []}
    assert agent._demote_unanchored_severity("high", hit) == "info"
    anchored = {"structured_score": 0.5, "matched_fields": ["class_pattern_in_code"]}
    assert agent._demote_unanchored_severity("high", anchored) == "high"


def test_semantic_layer_bonus_higher_than_full_at_same_similarity():
    agent = _make_agent()
    common = {
        "channel": "weaviate",
        "error_type": "dos",
        "structured_score": 0.0,
        "semantic_score": 1.0,
        "context_score": 0.1,
        "anchor_score": 0.55,
        "matched_fields": [],
    }

    semantic = {**common, "vector_layer": "semantic"}
    full = {**common, "vector_layer": "full"}
    agent._gate_candidate(semantic)
    agent._gate_candidate(full)

    assert semantic["layer_bonus"] == 0.08
    assert full["layer_bonus"] == 0.01
    assert semantic["total_score"] > full["total_score"]


def test_layer_bonus_zero_when_below_similarity_gate():
    agent = _make_agent()
    candidate = {
        "channel": "weaviate",
        "vector_layer": "semantic",
        "error_type": "dos",
        "structured_score": 0.0,
        "semantic_score": 0.5,
        "context_score": 0.1,
        "anchor_score": 0.55,
        "matched_fields": [],
    }
    agent._gate_candidate(candidate)
    assert candidate["layer_bonus"] == 0.0


def test_merge_weaviate_candidates_prefers_sparse_layer_identity():
    agent = _make_agent()
    merged = agent._merge_weaviate_candidates_by_sqlite_id(
        [
            {
                "channel": "weaviate",
                "sqlite_id": 5,
                "vector_layer": "full",
                "semantic_score": 0.9,
                "context_score": 0.1,
                "error_type": "dos",
                "reasoning": "weaviate_full_match",
            },
            {
                "channel": "weaviate",
                "sqlite_id": 5,
                "vector_layer": "semantic",
                "semantic_score": 0.88,
                "context_score": 0.05,
                "error_type": "dos",
                "reasoning": "weaviate_semantic_match",
            },
        ]
    )

    assert len(merged) == 1
    item = merged[0]
    assert item["sqlite_id"] == 5
    assert item["vector_layer"] == "semantic"
    assert item["semantic_score"] == 0.9
    assert set(item["matched_layers"]) == {"full", "semantic"}
    assert item["reasoning"] == "weaviate_semantic_match"
    details = {d["layer"]: d["bonus"] for d in item["matched_layer_details"]}
    assert details["semantic"] == 0.08
    assert details["full"] == 0.01


def test_merge_backfills_solution_from_solution_layer():
    agent = _make_agent()
    merged = agent._merge_weaviate_candidates_by_sqlite_id(
        [
            {
                "channel": "weaviate",
                "sqlite_id": 7,
                "vector_layer": "semantic",
                "semantic_score": 0.95,
                "context_score": 0.1,
                "error_type": "memory_overflow",
                "solution": "",
                "reasoning": "weaviate_semantic_match",
            },
            {
                "channel": "weaviate",
                "sqlite_id": 7,
                "vector_layer": "solution",
                "semantic_score": 0.8,
                "context_score": 0.1,
                "error_type": "memory_overflow",
                "solution": "Add bounds checks before buffer writes",
                "reasoning": "weaviate_solution_match",
            },
        ]
    )
    assert len(merged) == 1
    assert merged[0]["vector_layer"] == "semantic"
    assert merged[0]["solution"] == "Add bounds checks before buffer writes"


def test_backfill_solution_from_sqlite_patterns():
    agent = _make_agent()
    agent.enable_weaviate_query = False
    agent.vector_service = type("VS", (), {"is_connected": lambda self: False})()
    candidate = {
        "channel": "weaviate",
        "sqlite_id": 11,
        "vector_layer": "semantic",
        "solution": "",
    }
    agent._backfill_weaviate_candidate_solution(
        candidate,
        weaviate_hits=[{"sqlite_id": 11, "vector_layer": "semantic", "solution": ""}],
        sqlite_patterns=[{"id": 11, "solution": "Throttle abusive request sequences"}],
    )
    assert candidate["solution"] == "Throttle abusive request sequences"


def test_apply_file_function_anchors_boosts_matching_file():
    agent = _make_agent()
    candidate = {
        "channel": "weaviate",
        "sqlite_id": 3,
        "vector_layer": "semantic",
        "error_description": "The altivec_unavailable_exception function in arch/powerpc/kernel/traps.c ...",
        "class_pattern": "altivec_unavailable_exception",
        "file_pattern": "arch/powerpc/kernel/traps.c",
        "structured_score": 0.0,
        "matched_fields": [],
    }
    issue = {
        "description": "source_code_chunk L901-920: void altivec_unavailable_exception",
        "code_snippet": "void altivec_unavailable_exception(struct pt_regs *regs)\n{\n#if !defined(CONFIG_ALTIVEC)\n",
    }
    agent._apply_file_function_anchors(candidate, issue, "arch/powerpc/kernel/traps.c")
    assert candidate["structured_score"] > 0.2
    assert "file_basename_anchor" in candidate["matched_fields"]
    assert "class_pattern_in_code" in candidate["matched_fields"] or "function_name_in_code" in candidate["matched_fields"]


def test_resolve_gap_finding_line_prefers_function_in_chunk():
    agent = _make_agent()
    chunk = {
        "start_line": 900,
        "end_line": 920,
        "text": "void foo(void) {}\n\nvoid altivec_unavailable_exception(struct pt_regs *regs)\n{\n#if !defined(CONFIG_ALTIVEC)\n",
    }
    finding = {
        "line": 900,
        "evidence": {
            "class_pattern": "altivec_unavailable_exception",
            "error_description": "The altivec_unavailable_exception function in traps.c",
        },
    }
    line = agent._resolve_gap_finding_line(chunk, finding)
    assert line is not None
    assert line > 900


def test_agent_loads_inverse_density_layer_bonus_from_config():
    agent = AIDrivenSecondPassAnalysisAgent()
    assert agent.layer_bonus_map["semantic"] == 0.08
    assert agent.layer_bonus_map["full"] == 0.01
    assert agent.layer_bonus_require_similarity_gate is True
    assert agent.layer_bonus_map["semantic"] > agent.layer_bonus_map["full"]


# --------------------------------------------------------------------------- #
# 问题 5：弱证据不入计分（不变量测试）
# --------------------------------------------------------------------------- #
def test_weak_evidence_bonus_cannot_flip_any_gate_decision():
    """弱证据的 +0.1 **不可能**改变任何门控判定 —— 于是它只是噪声，应当从计分里拿掉。

    这是把问题 5 的"改法"锁成一条**可验证的不变量**，而不是靠"看起来合理"：
    把强证据字段的每一种子集都拿出来，比较"加 0.1"与"不加 0.1"两种算法下
    θ_s=0.65 / θ_w=0.20 两道门槛的通过情况。若哪天有人调了权重或阈值导致
    这条不再成立，本用例会失败，提醒"这 0.1 现在真的有影响了，需要重新论证"。
    """
    from itertools import combinations

    strong = AIDrivenSecondPassAnalysisAgent._UNIFIED_STRUCT_FIELDS
    weak = AIDrivenSecondPassAnalysisAgent._UNIFIED_ANNOTATION_FIELDS
    theta_s, theta_w = 0.65, 0.20

    def score(fields, add_weak):
        s = sum(w for f, w in strong.items() if f in fields)
        if add_weak and (set(fields) & weak):
            s += 0.1
        return min(1.0, s)

    names = list(strong)
    changed = []
    for r in range(len(names) + 1):
        for combo in combinations(names, r):
            for has_weak in (False, True):
                fields = set(combo)
                if has_weak:
                    fields.add("phenomenon_in_description")
                for theta in (theta_s, theta_w):
                    if (score(fields, False) >= theta) != (score(fields, True) >= theta):
                        changed.append((sorted(fields), theta))
    assert changed == [], "弱证据 +0.1 竟然改变了判定：%r —— 需要重新论证该不该保留" % changed[:5]


def test_weak_evidence_fields_are_not_scored():
    agent = _make_agent()
    assert agent._unified_structured_score(["phenomenon_in_description"]) == 0.0
    assert agent._unified_structured_score(["error_code_clone"]) == 0.5
    assert agent._unified_structured_score(["error_code_clone", "class_pattern_in_code"]) == 0.75
    assert agent._unified_structured_score(
        ["file_basename_anchor", "basename_match", "class_pattern_in_code"]) == 0.65


# --------------------------------------------------------------------------- #
# 问题 4：code_already_fixed 必须限定在"同一个文件"
# --------------------------------------------------------------------------- #
_FIX_SOLUTION = (
    "Remove incorrect logic: if (len > 0) memset(buf, 0, len);; ret = do_work(buf);. "
    "Ensure corrected path: add a bounds check before the write."
)


def test_cross_project_rejection_is_labelled_cross_file_not_already_fixed():
    """跨项目候选被拦下时，理由必须是"不是这个文件"，而不是"已修好"。

    场景即问题 4 的原文：库里那条记录讲的是**别的文件**，它的错误代码当然不在本文件里。
    历史实现据此判"已修好"并丢弃，占全部否决的 82.7%，而理由是误导性的。

    注意本用例同时锁定**取舍不变**：谓词没改，所以这条候选改前改后都被拦下，
    变的只有拒绝理由 —— 这正是"晋升数不变"这条预期的依据。
    """
    agent = _make_agent()
    candidate = {
        "channel": "weaviate",
        "vector_layer": "semantic",
        "error_type": "buffer_overflow",
        "solution": _FIX_SOLUTION,
        "semantic_score": 0.70,
        "context_score": 0.1,
        "anchor_score": 0.4,
        "structured_score": 0.2,
        "matched_fields": ["file_basename_anchor"],
        "file_pattern": "crypto/x509/x509_vpm.c",
        "error_description": "off-by-one in x509_vpm.c",
        "_analysis_file": "net/ipv4/ip_forward.c",
        "_current_code": "static int ip_forward(struct sk_buff *skb) { return 0; }\n",
    }
    assert agent._same_analysis_target(candidate) is False
    agent._gate_candidate(candidate)
    assert candidate["rejection_reason"] == "cross_file_mismatch"
    assert candidate["code_fixed_scope"] == "different_file"
    assert candidate["gating_decision"] == "discarded_hit"


def test_code_already_fixed_only_ever_appears_for_same_file():
    """`code_already_fixed` 只允许出现在"同一个文件"上 —— 这是它值得信任的前提。"""
    agent = _make_agent()
    for analysis_file, knowledge_file, expect in (
        ("net/ipv4/ip_forward.c", "net/ipv4/ip_forward.c", "code_already_fixed"),
        ("net/ipv4/ip_forward.c", "crypto/x509/x509_vpm.c", "cross_file_mismatch"),
    ):
        candidate = {
            "channel": "weaviate",
            "solution": _FIX_SOLUTION,
            "semantic_score": 0.70, "context_score": 0.1, "anchor_score": 0.4,
            "structured_score": 0.2, "matched_fields": ["file_basename_anchor"],
            "file_pattern": knowledge_file, "_analysis_file": analysis_file,
            "_current_code": "static int ip_forward(struct sk_buff *skb) { return 0; }\n",
        }
        agent._gate_candidate(candidate)
        assert candidate["rejection_reason"] == expect


def test_code_already_fixed_still_fires_on_same_file():
    """同一个文件、且错误代码确实已从文件中消失 → 仍应判"已修好"（这条判据要保留价值）。"""
    agent = _make_agent()
    candidate = {
        "channel": "weaviate",
        "vector_layer": "semantic",
        "error_type": "buffer_overflow",
        "solution": _FIX_SOLUTION,
        "semantic_score": 0.80,
        "context_score": 0.1,
        "anchor_score": 0.4,
        "structured_score": 0.2,
        "matched_fields": ["file_basename_anchor"],
        "file_pattern": "net/ipv4/ip_forward.c",
        "_analysis_file": "net/ipv4/ip_forward.c",
        "_current_code": "static int ip_forward(struct sk_buff *skb) { return 0; }\n",
    }
    assert agent._same_analysis_target(candidate) is True
    agent._gate_candidate(candidate)
    assert candidate["rejection_reason"] == "code_already_fixed"
    assert candidate["code_fixed_scope"] == "same_file"


def test_same_analysis_target_uses_two_level_path_not_basename():
    """同名不同目录不算同一个文件（inode.c 假阳性的教训）。"""
    agent = _make_agent()
    assert agent._same_analysis_target({
        "file_pattern": "fs/overlayfs/inode.c",
        "_analysis_file": "fs/udf/inode.c",
    }) is False
    assert agent._same_analysis_target({
        "file_pattern": "fs/udf/inode.c",
        "_analysis_file": "/root/autodl-tmp/MAS/tests/BigVul/before/CVE-1/abc/fs__udf__inode.c",
    }) is True
    # 取不到任一侧 → 不猜，禁止下"已修好"结论
    assert agent._same_analysis_target({"file_pattern": "a/b.c"}) is False
    assert agent._same_analysis_target({"_analysis_file": "a/b.c"}) is False


# --------------------------------------------------------------------------- #
# 问题 6/7：一把尺子（整文件）+ 最强证据在所有通道可用
# --------------------------------------------------------------------------- #
def _seed_current_code(agent, path: str, text: str) -> str:
    """把"被分析文件的完整内容"直接灌进 agent 的代码缓存，返回该路径。

    为什么不用临时文件：本仓库在受限环境（沙箱 / 中文用户名）下新建目录会被拒，
    写临时文件会让**整组测试**在 setup 阶段就挂掉。这里改为灌缓存——
    `_resolve_current_code` 正是按路径缓存的，灌进去和真的读一个文件对被测逻辑等价，
    而且顺带把测试变成纯内存、更快。
    """
    agent._current_code_cache[os.path.normcase(os.path.abspath(path))] = text
    return path


def test_error_code_clone_is_computed_for_weaviate_channel():
    """向量通道也要算 error_code_clone —— 历史实现里它压根没算（最强的一把尺子被废掉）。"""
    agent = _make_agent()
    src = _seed_current_code(agent, "net/ipv4/ip_forward.c",
                             "static int ip_forward(struct sk_buff *skb)\n{\n"
                             "    if (len > 0) memset(buf, 0, len);\n"
                             "    ret = do_work(buf);\n"
                             "    return ret;\n}\n")
    candidate = {
        "channel": "weaviate",
        "vector_layer": "solution",
        "solution": _FIX_SOLUTION,
        "matched_fields": [],
        "structured_score": 0.0,
    }
    agent._apply_error_code_clone_evidence(candidate, src, {"code_snippet": "return ret;"})
    assert "error_code_clone" in candidate["matched_fields"]
    assert agent._unified_structured_score(candidate["matched_fields"]) == 0.5


def test_error_code_clone_spans_whole_file_not_snippet():
    """错误代码落在"分片片段"之外、但在整个文件里 → 必须算命中（问题 6 的核心）。"""
    agent = _make_agent()
    src = _seed_current_code(
        agent, "big.c",
        "void unrelated_top(void) { }\n" * 200
        + "    if (len > 0) memset(buf, 0, len);\n"
        + "    ret = do_work(buf);\n"
        + "void unrelated_bottom(void) { }\n" * 200,
    )
    snippet = "void unrelated_top(void) { }"   # 历史实现只用这个 5 行片段当搜索范围
    assert agent._error_code_clone_matched(_FIX_SOLUTION, snippet) is False
    candidate = {"channel": "weaviate", "solution": _FIX_SOLUTION,
                 "matched_fields": [], "structured_score": 0.0}
    agent._apply_error_code_clone_evidence(candidate, src, {"code_snippet": snippet})
    assert "error_code_clone" in candidate["matched_fields"]


def test_curated_recall_uses_whole_file_as_haystack():
    """召回判定与"已修复"判定必须共用同一个 haystack（问题 7）。"""
    agent = _make_agent()
    whole_file = ("void other(void) { }\n" * 100
                  + "    if (len > 0) memset(buf, 0, len);\n    ret = do_work(buf);\n")
    src = _seed_current_code(agent, "pkt.c", whole_file)
    curated = {
        "file_path": src,
        "solution": _FIX_SOLUTION,
        "problem_phenomenon": "unrelated words here",
        "root_cause": "unrelated words here",
    }
    issue = {"file": src, "line": 1, "description": "source_code_chunk L1-5: other()",
             "code_snippet": "void other(void) { }"}
    match = agent._match_curated_issue(curated, issue, src)
    assert "error_code_clone" in match["matched_fields"]
    assert match["matched"] is True
    # 同一 haystack 下，"已修复"判据给出相反结论 —— 两把尺子刻度一致
    assert agent._candidate_code_fixed(
        {"solution": _FIX_SOLUTION, "_current_code": whole_file}) is False
