from __future__ import annotations

from core.agents.ai_driven_second_pass_analysis_agent import (
    AIDrivenSecondPassAnalysisAgent,
)


def _make_agent() -> AIDrivenSecondPassAnalysisAgent:
    agent = AIDrivenSecondPassAnalysisAgent.__new__(AIDrivenSecondPassAnalysisAgent)
    agent.similarity_threshold = 0.78
    agent.layer_bonus_map = {
        "semantic": 0.08,
        "solution": 0.05,
        "code_pattern": 0.03,
        "full": 0.01,
    }
    agent.layer_bonus_require_similarity_gate = True
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


def test_curated_chunk_covers_vuln_line():
    """curated 行落在 gap chunk 区间内应给 line_in_curated_range，即使 issue.line 是 chunk 首行。"""
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
    assert match["matched"] is True
    assert "line_in_curated_range" in match["matched_fields"]
    assert "basename_match" in match["matched_fields"]
    assert match["structured_score"] >= 0.45


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


def test_curated_without_basename_rejected():
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
    assert match.get("rejection_reason") == "curated_missing_basename"


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
