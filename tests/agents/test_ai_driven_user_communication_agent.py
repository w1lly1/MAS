"""
AI驱动用户沟通Agent测试
Tests for AI-driven User Communication Agent
"""

import unittest
from unittest.mock import Mock, patch

from tests.base import AgentTestCase
from core.agents.ai_driven_user_communication_agent import AIDrivenUserCommunicationAgent


class TestAIUserCommunicationAgent(AgentTestCase):
    """AI用户沟通Agent测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenUserCommunicationAgent()
    
    def test_agent_initialization(self):
        """测试agent初始化"""
        self.assert_agent_initialized()
        self.assertTrue(bool(self.agent.name))
    
    @patch('core.agents.ai_driven_user_communication_agent.AutoTokenizer')
    @patch('core.agents.ai_driven_user_communication_agent.AutoModelForCausalLM')
    def test_generate_user_report(self, mock_model, mock_tokenizer):
        """测试用户报告生成"""
        # 设置mock
        mock_tokenizer.from_pretrained.return_value = self.mock_model
        mock_model.from_pretrained.return_value = self.mock_model
        self.mock_model.generate.return_value = [Mock()]
        self.mock_model.decode.return_value = "代码分析报告：您的代码质量良好..."
        
        # 测试数据
        analysis_data = {
            "code_quality": {
                "score": 85,
                "issues": ["变量命名需要改进", "缺少文档注释"]
            },
            "security": {
                "score": 90,
                "vulnerabilities": ["硬编码密钥"]
            },
            "performance": {
                "score": 75,
                "bottlenecks": ["循环优化机会"]
            }
        }
        
        # 生成报告
        report = self.agent.generate_user_report(analysis_data)
        
        # 验证结果
        self.assertIsInstance(report, dict)
        self.assertIn("summary", report)
        self.assertIn("detailed_analysis", report)
        self.assertIn("recommendations", report)
        self.assertIn("next_steps", report)
    
    def test_explain_technical_concept(self):
        """测试技术概念解释"""
        technical_term = "循环复杂度"
        explanation = self.agent.explain_concept(technical_term)
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 50)
        self.assertIn("复杂度", explanation.lower())
    
    def test_format_code_suggestions(self):
        """测试代码建议格式化"""
        suggestions = [
            {
                "type": "naming",
                "line": 5,
                "message": "变量名应该使用小写和下划线",
                "suggestion": "将 'userName' 改为 'user_name'"
            },
            {
                "type": "documentation",
                "line": 1,
                "message": "函数缺少文档字符串",
                "suggestion": "添加详细的docstring说明函数功能"
            }
        ]
        
        formatted = self.agent.format_suggestions(suggestions)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("建议", formatted)
        self.assertIn("第5行", formatted)
    
    def test_generate_learning_path(self):
        """测试学习路径生成"""
        skill_gaps = ["代码重构", "单元测试", "设计模式"]
        
        learning_path = self.agent.generate_learning_path(skill_gaps)
        
        self.assertIsInstance(learning_path, list)
        self.assertGreater(len(learning_path), 0)
        
        # 验证学习路径结构
        for step in learning_path:
            self.assertIsInstance(step, dict)
            self.assertIn("topic", step)
            self.assertIn("description", step)
            self.assertIn("resources", step)
    
    def test_customize_report_for_user_level(self):
        """测试根据用户水平定制报告"""
        base_report = {
            "issues": ["复杂度过高", "缺少异常处理"],
            "technical_details": "圈复杂度为15，超过建议阈值10"
        }
        
        # 测试初级用户报告
        beginner_report = self.agent.customize_for_level(base_report, "beginner")
        self.assertIn("简单", beginner_report["explanation"])
        
        # 测试高级用户报告
        advanced_report = self.agent.customize_for_level(base_report, "advanced")
        self.assertIn("圈复杂度", advanced_report["explanation"])
    
    def test_interactive_q_and_a(self):
        """测试交互式问答"""
        questions = [
            "什么是代码重构？",
            "如何提高代码性能？",
            "为什么要写单元测试？"
        ]
        
        for question in questions:
            answer = self.agent.answer_question(question)
            
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 20)
            self.assert_valid_response(answer)
    
    def test_progress_tracking(self):
        """测试进度跟踪"""
        user_id = "test_user_123"
        
        # 记录初始状态
        initial_state = {
            "code_quality_score": 70,
            "security_score": 80,
            "performance_score": 65
        }
        self.agent.record_progress(user_id, initial_state)
        
        # 记录改进后状态
        improved_state = {
            "code_quality_score": 85,
            "security_score": 90,
            "performance_score": 75
        }
        self.agent.record_progress(user_id, improved_state)
        
        # 获取进度报告
        progress = self.agent.get_progress_report(user_id)
        
        self.assertIsInstance(progress, dict)
        self.assertIn("improvement", progress)
        self.assertIn("trends", progress)
        self.assertGreater(progress["improvement"]["code_quality"], 0)


class TestCommunicationIntegration(AgentTestCase):
    """用户沟通集成测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenUserCommunicationAgent()
    
    def test_end_to_end_communication_flow(self):
        """测试端到端沟通流程"""
        # 模拟完整的用户沟通场景
        user_profile = {
            "id": "user_001",
            "skill_level": "intermediate",
            "preferences": {
                "language": "chinese",
                "detail_level": "medium",
                "focus_areas": ["代码质量", "性能优化"]
            }
        }
        
        analysis_results = {
            "code_quality": {"score": 75, "issues": 5},
            "security": {"score": 85, "vulnerabilities": 2},
            "performance": {"score": 70, "bottlenecks": 3}
        }
        
        # 生成个性化报告
        personalized_report = self.agent.generate_personalized_report(
            user_profile, analysis_results
        )
        
        # 验证个性化报告
        self.assertIsInstance(personalized_report, dict)
        self.assertIn("greeting", personalized_report)
        self.assertIn("summary", personalized_report)
        self.assertIn("detailed_findings", personalized_report)
        self.assertIn("action_items", personalized_report)
        self.assertIn("learning_resources", personalized_report)
    
    def test_multi_language_support(self):
        """测试多语言支持"""
        test_message = "代码质量分析完成"
        
        # 测试英文翻译
        english_message = self.agent.translate_message(test_message, "english")
        self.assertIsInstance(english_message, str)
        self.assertNotEqual(english_message, test_message)
        
        # 测试保持中文
        chinese_message = self.agent.translate_message(test_message, "chinese")
        self.assertEqual(chinese_message, test_message)


if __name__ == "__main__":
    unittest.main()
