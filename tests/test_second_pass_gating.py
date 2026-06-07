from __future__ import annotations

from core.agents.ai_driven_second_pass_analysis_agent import (
    AIDrivenSecondPassAnalysisAgent,
)


def test_weaviate_semantic_anchor_promotes_hit():
    agent = AIDrivenSecondPassAnalysisAgent.__new__(AIDrivenSecondPassAnalysisAgent)
    agent.similarity_threshold = 0.78
    agent._debug_log = lambda *args, **kwargs: None

    candidate = {
        "channel": "weaviate",
        "vector_layer": "code_pattern",
        "error_type": "dos",
        "structured_score": 0.0,
        "semantic_score": 0.92,
        "context_score": 0.1,
        "anchor_score": 0.4,
        "matched_fields": [],
    }

    agent._gate_candidate(candidate)

    assert candidate["gating_decision"] == "explanatory_hit"
    assert candidate["layer_bonus"] > 0
