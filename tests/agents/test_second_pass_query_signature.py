"""
Second-pass query signature regression tests.
"""

import hashlib
import unittest

from tests.base import AgentTestCase
from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent


class TestSecondPassQuerySignature(AgentTestCase):
    """确保二次分析语义签名包含结构化锚点"""

    def setUp(self):
        super().setUp()
        self.agent = AIDrivenSecondPassAnalysisAgent()

    def test_semantic_signature_includes_anchors(self):
        issue = {
            "file": "src/net/handler.c",
            "source": "security_ai",
            "severity": "high",
            "tool": "second_pass_analysis",
            "requirement_id": 101,
            "issue_type": "input_validation",
            "function_name": "handle_packet",
            "location": "第42行",
            "line_number": 42,
            "code_snippet": "if (len < 0) return;",
        }
        signature = self.agent._semantic_signature(issue)
        parts = signature.split("|")

        self.assertGreaterEqual(len(parts), 8)
        self.assertIn("src/net/handler.c", parts[0])
        self.assertIn("security_ai", parts[1])
        self.assertIn("high", parts[2])
        self.assertIn("input_validation", signature)
        self.assertIn("handle_packet", signature)
        self.assertIn("42", signature)

        expected_hash = hashlib.sha256(issue["code_snippet"].lower().encode("utf-8", errors="ignore")).hexdigest()[:12]
        self.assertEqual(parts[-1], expected_hash)

    def test_curated_issue_match_promotes_structural_hit(self):
        issue = {
            "file": "src/kadmin/server/schpw.c",
            "source": "performance_bottleneck",
            "severity": "info",
            "line": 120,
            "line_number": 120,
            "code_snippet": "/* read, check ap-req length */",
        }
        curated = {
            "id": 1,
            "pattern_id": 1,
            "severity": "medium",
            "solution": "Add guards",
            "project_path": "krb5",
            "file_path": "src/kadmin/server/schpw.c",
            "start_line": 55,
            "end_line": 150,
            "problem_phenomenon": "UDP packet triggers infinite loop",
            "root_cause": "Improper validation of UDP packets",
            "status": "resolved",
        }

        match_info = self.agent._match_curated_issue(curated, issue, issue.get("file"))
        self.assertTrue(match_info.get("structured_score", 0.0) >= 0.45)

        curated_hit = dict(curated)
        curated_hit.update(match_info)
        candidate = self.agent._build_candidate_from_curated_issue(
            curated_hit,
            issue_desc="Possible UDP packet issue",
            file_path=issue.get("file"),
            issue=issue,
        )
        self.agent._gate_candidate(candidate)

        self.assertIn(candidate.get("gating_decision"), {"formal_hit", "explanatory_hit"})


if __name__ == "__main__":
    unittest.main()
