"""可读性报告：置信度公式内嵌分项分 + vectorDebug hit_index 联动。"""

from __future__ import annotations

import unittest

from tests.base import AgentTestCase
from core.agents.ai_driven_readability_enhancement_agent import AIDrivenReadabilityEnhancementAgent


class TestMarkdownFormulaAndVectorDebugLink(AgentTestCase):
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenReadabilityEnhancementAgent()

    def test_format_confidence_formula_embeds_scores(self):
        comps = {
            "structured": 0.0,
            "semantic": 1.0,
            "context": 0.0,
            "anchor": 0.0,
            "anchor_bonus": 0.0,
            "layer_bonus": 0.0,
            "penalty": 0.0,
        }
        lines = self.agent._format_confidence_formula_lines(comps, "weaviate", indent="      ")
        text = "\n".join(lines)
        self.assertIn("structured(0.0)*0.5", text)
        self.assertIn("semantic(1.0)*0.35", text)
        self.assertIn("context(0.0)*0.1", text)
        self.assertIn("anchor(0.0)*0.05", text)
        self.assertIn("anchor_bonus(0.0)", text)
        self.assertIn("layer_bonus(0.0)", text)
        self.assertIn("penalty(0.0)", text)

    def test_render_evidence_includes_formula_and_vector_debug_ref(self):
        issue = {
            "description": "历史知识命中: input_validation",
            "line": 1,
        }
        evidence = {
            "channel": "weaviate",
            "total_score": 0.3875,
            "sqlite_id": 9,
            "reasoning": "weaviate_semantic_match",
            "matched_layers": ["semantic", "full"],
            "matched_layer_details": [
                {"layer": "semantic", "bonus": 0.08},
                {"layer": "full", "bonus": 0.01},
            ],
            "error_description": "The altivec_unavailable_exception function in arch/powerpc/kernel/traps.c causes denial of service.",
            "class_pattern": "altivec_unavailable_exception",
            "confidence_components": {
                "structured": 0.0,
                "semantic": 1.0,
                "context": 0.0,
                "anchor": 0.0,
                "anchor_bonus": 0.0,
                "layer_bonus": 0.08,
                "penalty": 0.0,
            },
            "vector_debug_hit_index": 13,
            "vector_debug_hit_indexes": [13, 45, 102],
            "vector_debug_query_pass": "gap_from_original_analysis",
        }
        rendered = self.agent._render_evidence_item(issue, evidence)
        self.assertIn("置信度计算公式:", rendered)
        self.assertIn("semantic(1.0)*0.35", rendered)
        self.assertNotIn("置信度分项:", rendered)
        self.assertIn("vectorDebug节点: #13 (gap_from_original_analysis, sqlite_id=9)", rendered)
        self.assertIn("相关节点: #13, #45, #102", rendered)
        self.assertIn("命中分层: semantic(+0.08), full(+0.01)", rendered)
        self.assertIn("知识摘要: The altivec_unavailable_exception function", rendered)
        self.assertIn("锚定函数: altivec_unavailable_exception", rendered)
        self.assertIn("命中原因: weaviate_semantic_match", rendered)

    def test_attach_vector_debug_refs_prefers_gap_and_error_type(self):
        report_data = {
            "issues": [
                {
                    "description": "历史知识命中: dos",
                    "line": 1,
                    "source": "db_supplemented",
                    "severity": "medium",
                    "evidence": {"sqlite_id": 3, "channel": "weaviate"},
                }
            ]
        }
        payload = {
            "hits": [
                {
                    "hit_index": 2,
                    "sqlite_id": 3,
                    "error_type": "dos",
                    "query_pass": "validation_from_consolidated",
                    "produces_valid_output": True,
                },
                {
                    "hit_index": 40,
                    "sqlite_id": 3,
                    "error_type": "dos",
                    "query_pass": "gap_from_original_analysis",
                    "produces_valid_output": True,
                },
                {
                    "hit_index": 99,
                    "sqlite_id": 3,
                    "error_type": "memory_overflow",
                    "query_pass": "gap_from_original_analysis",
                    "produces_valid_output": True,
                },
            ]
        }
        self.agent._attach_vector_debug_refs(report_data, payload)
        ev = report_data["issues"][0]["evidence"]
        self.assertEqual(ev["vector_debug_hit_index"], 40)
        self.assertEqual(ev["vector_debug_query_pass"], "gap_from_original_analysis")
        self.assertEqual(ev["vector_debug_hit_indexes"], [40, 2])


if __name__ == "__main__":
    unittest.main()
