"""Tests for LLM-first security agent strategy helpers."""

import unittest

from core.agents.ai_driven_security_agent import AIDrivenSecurityAgent


class TestAISecurityAgentLLMStrategy(unittest.TestCase):
    def setUp(self):
        self.agent = AIDrivenSecurityAgent()

    def test_validate_security_analysis_result_schema(self):
        raw = {
            "ai_security_analysis": {
                "overall_security_rating": {"security_score": 7.2, "rating_level": "Good"},
                "vulnerabilities_detected": [
                    {
                        "vulnerability_id": "A",
                        "type": "sql_injection",
                        "description": "SQL injection risk",
                        "severity": "high",
                        "location": "L10",
                        "ai_confidence": 0.9,
                    }
                ],
                "remediation_plan": {
                    "immediate_actions": [
                        {
                            "priority": "high",
                            "vulnerability_id": "A",
                            "type": "sql_injection",
                            "fix": "Use parameterized queries",
                        }
                    ],
                    "short_term_fixes": [],
                    "long_term_improvements": [],
                    "estimated_effort": "1-2 days",
                },
                "hardening_recommendations": [
                    {
                        "category": "database",
                        "priority": "high",
                        "recommendation": "Enable least privilege",
                        "implementation": "Create read/write split users",
                    }
                ],
                "failed_steps": [],
            },
            "analysis_status": "completed",
        }

        result = self.agent._validate_security_analysis_result(raw)
        self.assertIn("ai_security_analysis", result)
        self.assertIn("vulnerabilities_detected", result["ai_security_analysis"])
        self.assertIn("remediation_plan", result["ai_security_analysis"])
        self.assertIn("hardening_recommendations", result["ai_security_analysis"])

    def test_parse_llm_security_rating_json(self):
        generated = [
            {
                "generated_text": '{"llm_security_score": 8.1, "confidence": 0.78, "primary_risks": ["xss"], "explanation": "风险可控"}'
            }
        ]
        parsed = self.agent._parse_llm_security_rating(generated)
        self.assertTrue(parsed.get("available"))
        self.assertGreaterEqual(parsed.get("score", 0), 0)
        self.assertLessEqual(parsed.get("score", 10), 10)


if __name__ == "__main__":
    unittest.main()
