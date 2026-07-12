"""
二次分析查询2：原始源代码按上下文分片后查库补漏。
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from tests.base import AgentTestCase
from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent


class TestSecondPassGapCodeChunks(AgentTestCase):
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenSecondPassAnalysisAgent()
        self.agent.enable_weaviate_query = False
        self.agent.enable_llm_second_pass = False
        self.agent.max_new_findings = 5
        self.agent.gap_code_chunk_chars = 80
        self.agent.gap_chunk_overlap_lines = 1
        self.agent.max_gap_code_chunks = 10
        self.agent._debug_log = lambda *args, **kwargs: None

    def test_split_file_into_context_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.c"
            path.write_text(
                "line1\n" * 5 + "line2_block\n" * 10 + "line3_tail\n" * 5,
                encoding="utf-8",
            )
            chunks = self.agent._split_file_into_context_chunks(str(path))
            self.assertGreaterEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["start_line"], 1)
            self.assertLessEqual(chunks[0]["end_line"], chunks[1]["start_line"] + 1)
            self.assertTrue(chunks[0]["text"])

    def test_code_chunk_as_issue(self):
        chunk = {
            "file": "a.c",
            "start_line": 10,
            "end_line": 20,
            "text": "int foo() { return 1; }",
            "chunk_index": 0,
        }
        issue = self.agent._code_chunk_as_issue(chunk)
        self.assertEqual(issue["source"], "source_code_chunk")
        self.assertEqual(issue["line"], 10)
        self.assertIn("source_code_chunk", issue["description"])
        self.assertEqual(issue["code_snippet"], chunk["text"])

    def test_collect_gap_evidence_from_code_chunks(self):
        called = []

        async def fake_collect(issue, sqlite_patterns, layer_mode=None):
            called.append(issue.get("line"))
            return {
                "issue_description": issue.get("description"),
                "issue_file": issue.get("file"),
                "candidates": [],
                "weaviate_hits": [],
            }

        self.agent._collect_evidence = fake_collect
        chunks = [
            {"file": "a.c", "start_line": 1, "end_line": 5, "text": "aaa", "chunk_index": 0},
            {"file": "a.c", "start_line": 6, "end_line": 10, "text": "bbb", "chunk_index": 1},
        ]
        gap_evidence = asyncio.run(
            self.agent._collect_gap_evidence_from_code_chunks(
                code_chunks=chunks,
                sqlite_patterns=[],
                layer_mode="all_only",
                run_id="run-1",
            )
        )
        self.assertEqual(called, [1, 6])
        self.assertEqual(len(gap_evidence), 2)
        self.assertEqual(gap_evidence[0]["query_channel"], "gap_from_original_analysis")
        self.assertEqual(gap_evidence[0]["code_chunk"]["start_line"], 1)
        self.assertIn("源代码分片", gap_evidence[0]["query_pass_label"])

    def test_derive_new_findings_prefers_code_chunk_evidence(self):
        gap_evidence = [
            {
                "code_chunk": {
                    "file": "src/b.c",
                    "start_line": 42,
                    "end_line": 60,
                    "text": "free(p); use(p);",
                    "chunk_index": 0,
                },
                "candidates": [
                    {
                        "gating_decision": "explanatory_hit",
                        "error_type": "use_after_free",
                        "severity": "high",
                        "channel": "weaviate",
                        "sqlite_id": 7,
                        "semantic_score": 0.9,
                        "structured_score": 0.5,
                        "total_score": 0.9,
                        "matched_fields": ["file_basename_anchor", "class_pattern_in_code"],
                        "file_pattern": "src/b.c",
                        "class_pattern": "free",
                        "solution": "null after free",
                        "reasoning": "pattern hit",
                        "rejection_reason": None,
                    }
                ],
            }
        ]
        findings = self.agent._derive_new_findings_from_gap_evidence(
            gap_retrieval_evidence=gap_evidence,
            fallback_issues=[{"description": "should not be used", "line": 1}],
            fallback_evidence=[{"candidates": []}],
            run_id="run-1",
            requirement_id=1,
            file_path="src/b.c",
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("use_after_free", findings[0]["description"])
        self.assertEqual(findings[0]["file"], "src/b.c")
        self.assertEqual(findings[0]["line"], 42)
        self.assertEqual(findings[0]["evidence"]["primary_channel"], "weaviate")

    def test_curated_finding_uses_vuln_line_and_primary_channel(self):
        gap_evidence = [
            {
                "code_chunk": {
                    "file": r"cf1a0c41\src__kadmin__server__schpw.c",
                    "start_line": 38,
                    "end_line": 80,
                    "text": "if (req->length < 4) {\n        goto chpwfail;\n    }\n",
                    "chunk_index": 0,
                },
                "candidates": [
                    {
                        "gating_decision": "explanatory_hit",
                        "error_type": "dos",
                        "severity": "medium",
                        "channel": "curated_issue",
                        "sqlite_id": 1,
                        "semantic_score": 0.0,
                        "structured_score": 0.7,
                        "total_score": 0.5,
                        "matched_fields": ["basename_match", "line_in_curated_range"],
                        "start_line": 55,
                        "solution": "goto bailout",
                        "reasoning": "curated_issue_structural_match",
                        "rejection_reason": None,
                    }
                ],
            }
        ]
        findings = self.agent._derive_new_findings_from_gap_evidence(
            gap_retrieval_evidence=gap_evidence,
            fallback_issues=[],
            fallback_evidence=[{"candidates": []}],
            run_id="run-1",
            requirement_id=1,
            file_path=r"cf1a0c41\src__kadmin__server__schpw.c",
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["line"], 55)
        self.assertEqual(findings[0]["evidence"]["primary_channel"], "curated")

    def test_derive_new_findings_drops_unanchored_weaviate(self):
        """无文件/函数锚定的 weaviate 命中不得进入最终 findings。"""
        gap_evidence = [
            {
                "code_chunk": {
                    "file": "src/b.c",
                    "start_line": 42,
                    "end_line": 60,
                    "text": "free(p); use(p);",
                    "chunk_index": 0,
                },
                "candidates": [
                    {
                        "gating_decision": "explanatory_hit",
                        "error_type": "memory_overflow",
                        "severity": "high",
                        "channel": "weaviate",
                        "sqlite_id": 5,
                        "semantic_score": 1.0,
                        "structured_score": 0.0,
                        "total_score": 0.95,
                        "matched_fields": [],
                        "file_pattern": "net/netlabel/netlabel_cipso_v4.c",
                        "solution": "unrelated",
                        "reasoning": "semantic only",
                        "rejection_reason": None,
                    }
                ],
            }
        ]
        findings = self.agent._derive_new_findings_from_gap_evidence(
            gap_retrieval_evidence=gap_evidence,
            fallback_issues=[{"description": "should not be used", "line": 1}],
            fallback_evidence=[{"candidates": []}],
            run_id="run-1",
            requirement_id=1,
            file_path="src/b.c",
        )
        self.assertEqual(len(findings), 0)


if __name__ == "__main__":
    unittest.main()
