"""vectorDebug 双查询命中来源标注测试。"""

from __future__ import annotations

import unittest

from tests.base import AgentTestCase
from core.agents.ai_driven_readability_enhancement_agent import AIDrivenReadabilityEnhancementAgent


class TestVectorDebugDualQuery(AgentTestCase):
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenReadabilityEnhancementAgent()

    def test_build_vector_debug_separates_validation_and_gap(self):
        report_data = {
            "issues": [
                {
                    "source": "db_supplemented",
                    "severity": "high",
                    "line": 10,
                    "description": "历史知识命中: dos",
                    "evidence": {"sqlite_id": 1, "channel": "weaviate"},
                }
            ],
            "retrieval_evidence": [
                {
                    "query_channel": "validation_from_consolidated",
                    "query_pass_label": "一轮LLM/consolidated分析命中数据库",
                    "issue_description": "consolidated issue",
                    "issue_file": "a.c",
                    "weaviate_hits": [
                        {
                            "sqlite_id": 1,
                            "vector_layer": "full",
                            "distance": 0.0,
                            "similarity": 1.0,
                            "error_type": "dos",
                            "severity": "medium",
                        }
                    ],
                    "evidence_hits": [{"sqlite_id": 1, "gating_decision": "formal_hit"}],
                }
            ],
            "gap_retrieval_evidence": [
                {
                    "query_channel": "gap_from_original_analysis",
                    "query_pass_label": "二轮原始源代码分片命中数据库",
                    "issue_description": "source_code_chunk L42-60: free(p)",
                    "issue_file": "a.c",
                    "code_chunk": {
                        "start_line": 42,
                        "end_line": 60,
                        "chunk_index": 0,
                        "text": "free(p); use(p);",
                    },
                    "weaviate_hits": [
                        {
                            "sqlite_id": 9,
                            "vector_layer": "full",
                            "distance": 0.1,
                            "similarity": 0.95,
                            "error_type": "uaf",
                            "severity": "high",
                        }
                    ],
                    "evidence_hits": [],
                    # 【小事 12】被门控拒掉的候选也在 candidates 里，带真实判定与拒绝原因
                    "candidates": [
                        {
                            "sqlite_id": 9,
                            "gating_decision": "discarded_hit",
                            "rejection_reason": "cross_file_mismatch",
                        }
                    ],
                }
            ],
        }

        payload = self.agent._build_vector_debug_payload(report_data)
        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["summary"]["validation_from_consolidated_count"], 1)
        self.assertEqual(payload["summary"]["gap_from_original_analysis_count"], 1)
        self.assertEqual(payload["summary"]["matched_as_evidence_validation"], 1)
        self.assertEqual(payload["summary"]["matched_as_evidence_gap"], 0)
        self.assertEqual(payload["summary"]["produces_valid_output_validation"], 1)
        self.assertEqual(payload["summary"]["produces_valid_output_gap"], 0)
        self.assertEqual(payload["effective_hit_indexes"], [0])

        channels = {h["query_pass"] for h in payload["hits"]}
        self.assertEqual(channels, {"validation_from_consolidated", "gap_from_original_analysis"})

        gap_hit = next(h for h in payload["hits"] if h["query_pass"] == "gap_from_original_analysis")
        self.assertEqual(gap_hit["code_chunk_start_line"], 42)
        self.assertEqual(gap_hit["code_chunk_end_line"], 60)
        self.assertIn("源代码分片", gap_hit["query_pass_label"])
        self.assertFalse(gap_hit["produces_valid_output"])
        self.assertEqual(gap_hit["hit_index"], 1)
        # 【小事 12】未晋升的记录必须写明判定与"为什么被拒"（历史实现两个字段都是 null）
        self.assertFalse(gap_hit["matched_as_evidence"])
        self.assertEqual(gap_hit["gating_decision"], "discarded_hit")
        self.assertEqual(gap_hit["rejection_reason"], "cross_file_mismatch")

        validation_hit = next(h for h in payload["hits"] if h["query_pass"] == "validation_from_consolidated")
        self.assertTrue(validation_hit["matched_as_evidence"])
        self.assertTrue(validation_hit["produces_valid_output"])
        self.assertEqual(validation_hit["output_issue_count"], 1)
        self.assertEqual(validation_hit["output_issues"][0]["line"], 10)
        self.assertIn("一轮LLM", validation_hit["query_pass_label"])
        # 晋升的记录：判定写明、拒绝原因为空（不是"没记"）
        self.assertEqual(validation_hit["gating_decision"], "formal_hit")
        self.assertIsNone(validation_hit["rejection_reason"])

    def test_unadmitted_hit_without_candidate_record_still_writes_both_keys(self):
        """连门控候选记录都没有时，两个字段也要存在（值为 None），不要让读取方猜。"""
        report_data = {
            "issues": [],
            "retrieval_evidence": [
                {
                    "query_channel": "validation_from_consolidated",
                    "issue_description": "x",
                    "issue_file": "a.c",
                    "weaviate_hits": [{"sqlite_id": 5, "vector_layer": "full", "similarity": 0.7}],
                    "evidence_hits": [],
                }
            ],
            "gap_retrieval_evidence": [],
        }
        payload = self.agent._build_vector_debug_payload(report_data)
        hit = payload["hits"][0]
        self.assertIn("rejection_reason", hit)
        self.assertIn("gating_decision", hit)
        self.assertIsNone(hit["gating_decision"])
        self.assertIsNone(hit["rejection_reason"])


if __name__ == "__main__":
    unittest.main()
