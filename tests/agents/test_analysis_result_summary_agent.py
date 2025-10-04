"""
分析结果汇总Agent测试
Tests for Analysis Result Summary Agent
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from tests.base import AgentTestCase
from core.agents.analysis_result_summary_agent import SummaryAgent


class TestAnalysisResultSummaryAgent(AgentTestCase):
    """分析结果汇总Agent测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = SummaryAgent()
    
    def test_agent_initialization(self):
        """测试agent初始化"""
        self.assert_agent_initialized()
        self.assertTrue(bool(self.agent.name))
    
    def test_aggregate_analysis_results(self):
        """测试分析结果聚合"""
        # 模拟多个agent的分析结果
        analysis_results = {
            "code_quality": {
                "agent": "AI Code Quality Agent",
                "score": 85,
                "issues": [
                    {"type": "naming", "severity": "medium", "count": 3},
                    {"type": "complexity", "severity": "high", "count": 1}
                ],
                "timestamp": datetime.now().isoformat()
            },
            "security": {
                "agent": "AI Security Agent", 
                "score": 78,
                "vulnerabilities": [
                    {"type": "sql_injection", "severity": "high", "count": 1},
                    {"type": "hardcoded_secrets", "severity": "medium", "count": 2}
                ],
                "timestamp": datetime.now().isoformat()
            },
            "performance": {
                "agent": "AI Performance Agent",
                "score": 72,
                "bottlenecks": [
                    {"type": "algorithm", "severity": "high", "count": 2},
                    {"type": "memory", "severity": "low", "count": 3}
                ],
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # 执行聚合
        summary = self.agent.aggregate_results(analysis_results)
        
        # 验证聚合结果
        self.assertIsInstance(summary, dict)
        self.assertIn("overall_score", summary)
        self.assertIn("summary_by_category", summary)
        self.assertIn("priority_issues", summary)
        self.assertIn("recommendations", summary)
        
        # 验证总体评分
        self.assertIsInstance(summary["overall_score"], (int, float))
        self.assertGreaterEqual(summary["overall_score"], 0)
        self.assertLessEqual(summary["overall_score"], 100)
    
    def test_prioritize_issues(self):
        """测试问题优先级排序"""
        issues = [
            {"type": "security", "severity": "critical", "description": "SQL注入漏洞"},
            {"type": "performance", "severity": "high", "description": "性能瓶颈"},
            {"type": "code_quality", "severity": "medium", "description": "代码重复"},
            {"type": "security", "severity": "low", "description": "信息泄露"},
            {"type": "performance", "severity": "critical", "description": "内存泄露"}
        ]
        
        prioritized = self.agent.prioritize_issues(issues)
        
        self.assertIsInstance(prioritized, list)
        self.assertEqual(len(prioritized), len(issues))
        
        # 验证排序：critical > high > medium > low
        severities = [issue["severity"] for issue in prioritized]
        expected_order = ["critical", "critical", "high", "medium", "low"]
        self.assertEqual(severities, expected_order)
    
    def test_generate_executive_summary(self):
        """测试执行摘要生成"""
        aggregated_data = {
            "overall_score": 78,
            "total_issues": 15,
            "critical_issues": 2,
            "high_issues": 5,
            "categories": {
                "code_quality": {"score": 85, "issues": 6},
                "security": {"score": 75, "issues": 4},
                "performance": {"score": 74, "issues": 5}
            }
        }
        
        summary = self.agent.generate_executive_summary(aggregated_data)
        
        self.assertIsInstance(summary, dict)
        self.assertIn("headline", summary)
        self.assertIn("key_metrics", summary)
        self.assertIn("main_concerns", summary)
        self.assertIn("success_areas", summary)
    
    def test_calculate_trend_analysis(self):
        """测试趋势分析"""
        historical_data = [
            {"date": "2024-01-01", "overall_score": 70, "issues": 20},
            {"date": "2024-01-15", "overall_score": 75, "issues": 18},
            {"date": "2024-02-01", "overall_score": 78, "issues": 15},
            {"date": "2024-02-15", "overall_score": 82, "issues": 12}
        ]
        
        trends = self.agent.calculate_trends(historical_data)
        
        self.assertIsInstance(trends, dict)
        self.assertIn("score_trend", trends)
        self.assertIn("issues_trend", trends)
        self.assertIn("improvement_rate", trends)
        
        # 验证趋势方向
        self.assertEqual(trends["score_trend"], "improving")  # 分数上升
        self.assertEqual(trends["issues_trend"], "decreasing")  # 问题减少
    
    def test_cross_category_correlation(self):
        """测试跨类别关联分析"""
        category_data = {
            "code_quality": {
                "complexity_issues": 5,
                "maintainability_score": 75
            },
            "performance": {
                "slow_functions": 3,
                "memory_usage": "high"
            },
            "security": {
                "vulnerabilities": 2,
                "secure_coding_score": 80
            }
        }
        
        correlations = self.agent.analyze_correlations(category_data)
        
        self.assertIsInstance(correlations, list)
        
        # 验证关联分析结果结构
        for correlation in correlations:
            self.assertIn("categories", correlation)
            self.assertIn("relationship", correlation)
            self.assertIn("impact", correlation)
    
    def test_generate_action_plan(self):
        """测试行动计划生成"""
        prioritized_issues = [
            {
                "type": "security",
                "severity": "critical", 
                "description": "SQL注入漏洞",
                "estimated_effort": "high",
                "impact": "critical"
            },
            {
                "type": "performance",
                "severity": "high",
                "description": "算法优化",
                "estimated_effort": "medium",
                "impact": "high"
            }
        ]
        
        action_plan = self.agent.generate_action_plan(prioritized_issues)
        
        self.assertIsInstance(action_plan, dict)
        self.assertIn("immediate_actions", action_plan)
        self.assertIn("short_term_goals", action_plan)
        self.assertIn("long_term_improvements", action_plan)
        self.assertIn("timeline", action_plan)


class TestSummaryIntegration(AgentTestCase):
    """汇总集成测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = SummaryAgent()
    
    def test_comprehensive_summary_workflow(self):
        """测试综合汇总工作流"""
        # 模拟完整的分析流程结果
        comprehensive_results = {
            "project_info": {
                "name": "test_project",
                "language": "python",
                "files_analyzed": 25,
                "lines_of_code": 5000
            },
            "analysis_results": {
                "code_quality": {
                    "score": 82,
                    "issues": 8,
                    "top_issues": ["naming_convention", "code_duplication"]
                },
                "security": {
                    "score": 76,
                    "vulnerabilities": 5,
                    "critical_vulns": 1
                },
                "performance": {
                    "score": 79,
                    "bottlenecks": 6,
                    "memory_issues": 2
                },
                "static_analysis": {
                    "score": 85,
                    "warnings": 12,
                    "errors": 0
                }
            }
        }
        
        # 生成综合报告
        comprehensive_summary = self.agent.create_comprehensive_report(
            comprehensive_results
        )
        
        # 验证综合报告
        self.assertIsInstance(comprehensive_summary, dict)
        self.assertIn("project_overview", comprehensive_summary)
        self.assertIn("executive_summary", comprehensive_summary)
        self.assertIn("detailed_analysis", comprehensive_summary)
        self.assertIn("recommendations", comprehensive_summary)
        self.assertIn("next_steps", comprehensive_summary)
    
    def test_report_export_formats(self):
        """测试报告导出格式"""
        summary_data = {
            "overall_score": 80,
            "issues": 10,
            "recommendations": ["优化算法", "修复安全漏洞"]
        }
        
        # 测试JSON格式
        json_report = self.agent.export_report(summary_data, format="json")
        self.assertIsInstance(json_report, str)
        
        # 测试Markdown格式
        markdown_report = self.agent.export_report(summary_data, format="markdown")
        self.assertIsInstance(markdown_report, str)
        self.assertIn("#", markdown_report)  # Markdown标题
        
        # 测试HTML格式
        html_report = self.agent.export_report(summary_data, format="html")
        self.assertIsInstance(html_report, str)
        self.assertIn("<html>", html_report)


if __name__ == "__main__":
    unittest.main()
