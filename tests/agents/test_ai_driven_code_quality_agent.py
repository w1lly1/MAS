"""
AI驱动代码质量分析Agent测试
Tests for AI-driven Code Quality Analysis Agent
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from tests.base import AgentTestCase
from core.agents.ai_driven_code_quality_agent import AIDrivenCodeQualityAgent


class TestAIDrivenCodeQualityAgent(AgentTestCase):
    """AI代码质量分析Agent测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenCodeQualityAgent()
    
    def test_agent_initialization(self):
        """测试agent初始化"""
        self.assert_agent_initialized()
        self.assertEqual(self.agent.name, "AI Code Quality Agent")
    
    @patch('core.agents.ai_driven_code_quality_agent.AutoTokenizer')
    @patch('core.agents.ai_driven_code_quality_agent.AutoModelForCausalLM')
    def test_analyze_code_quality(self, mock_model, mock_tokenizer):
        """测试代码质量分析功能"""
        # 设置mock
        mock_tokenizer.from_pretrained.return_value = self.mock_model
        mock_model.from_pretrained.return_value = self.mock_model
        self.mock_model.generate.return_value = [Mock()]
        self.mock_model.decode.return_value = "代码质量良好，建议进行以下优化：..."
        
        # 测试代码
        test_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
        
        # 执行分析
        result = self.agent.analyze_code_quality(test_code)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("analysis", result)
        self.assertIn("recommendations", result)
        self.assertIn("quality_score", result)
    
    def test_analyze_empty_code(self):
        """测试空代码分析"""
        with self.assertRaises(ValueError):
            self.agent.analyze_code_quality("")
    
    def test_analyze_invalid_code(self):
        """测试无效代码分析"""
        invalid_code = "this is not python code!!!"
        result = self.agent.analyze_code_quality(invalid_code)
        
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
    
    @patch('core.agents.ai_driven_code_quality_agent.AutoTokenizer')
    def test_model_loading_error(self, mock_tokenizer):
        """测试模型加载错误"""
        mock_tokenizer.from_pretrained.side_effect = Exception("Model loading failed")
        
        with self.assertRaises(Exception):
            self.agent.load_model()
    
    def test_generate_recommendations(self):
        """测试生成建议功能"""
        analysis_result = {
            "issues": ["变量命名不规范", "缺少注释"],
            "complexity": "medium"
        }
        
        recommendations = self.agent.generate_recommendations(analysis_result)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_calculate_quality_score(self):
        """测试质量评分计算"""
        metrics = {
            "cyclomatic_complexity": 5,
            "code_duplication": 0.1,
            "test_coverage": 0.8,
            "maintainability_index": 75
        }
        
        score = self.agent.calculate_quality_score(metrics)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


class TestCodeQualityIntegration(AgentTestCase):
    """代码质量分析集成测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = AICodeQualityAgent()
    
    def test_full_analysis_workflow(self):
        """测试完整分析工作流"""
        # 准备测试文件
        test_file = self.temp_dir / "test_code.py"
        test_file.write_text("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        result = 0
        for i in range(b):
            result += a
        return result
""")
        
        # 执行分析
        result = self.agent.analyze_file(str(test_file))
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("file_path", result)
        self.assertIn("analysis", result)
        self.assertIn("metrics", result)
        self.assertEqual(result["file_path"], str(test_file))


if __name__ == "__main__":
    unittest.main()
