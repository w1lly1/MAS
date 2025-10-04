"""
AI驱动性能分析Agent测试
Tests for AI-driven Performance Analysis Agent
"""

import unittest
from unittest.mock import Mock, patch
import time

from tests.base import AgentTestCase, PerformanceTestCase
from core.agents.ai_driven_performance_agent import AIDrivenPerformanceAgent


class TestAIDrivenPerformanceAgent(AgentTestCase):
    """AI性能分析Agent测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenPerformanceAgent()
    
    def test_agent_initialization(self):
        """测试agent初始化"""
        self.assert_agent_initialized()
        self.assertTrue(bool(self.agent.name))
    
    @patch('core.agents.ai_driven_performance_agent.AutoTokenizer')
    @patch('core.agents.ai_driven_performance_agent.AutoModelForCausalLM')
    def test_analyze_performance(self, mock_model, mock_tokenizer):
        """测试性能分析功能"""
        # 设置mock
        mock_tokenizer.from_pretrained.return_value = self.mock_model
        mock_model.from_pretrained.return_value = self.mock_model
        self.mock_model.generate.return_value = [Mock()]
        self.mock_model.decode.return_value = "性能分析：发现以下性能问题..."
        
        # 测试代码
        test_code = """
def slow_function():
    result = []
    for i in range(10000):
        for j in range(1000):
            result.append(i * j)
    return result
"""
        
        # 执行分析
        result = self.agent.analyze_performance(test_code)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("performance_issues", result)
        self.assertIn("optimization_suggestions", result)
        self.assertIn("complexity_analysis", result)
    
    def test_profile_code_execution(self):
        """测试代码执行性能分析"""
        def test_function():
            time.sleep(0.01)  # 模拟执行时间
            return sum(range(1000))
        
        profile_result = self.agent.profile_execution(test_function)
        
        self.assertIsInstance(profile_result, dict)
        self.assertIn("execution_time", profile_result)
        self.assertIn("memory_usage", profile_result)
        self.assertGreater(profile_result["execution_time"], 0)
    
    def test_memory_analysis(self):
        """测试内存使用分析"""
        test_code = """
def memory_intensive():
    big_list = [i for i in range(1000000)]
    return len(big_list)
"""
        
        memory_result = self.agent.analyze_memory_usage(test_code)
        
        self.assertIsInstance(memory_result, dict)
        self.assertIn("memory_peak", memory_result)
        self.assertIn("memory_recommendations", memory_result)
    
    def test_bottleneck_detection(self):
        """测试性能瓶颈检测"""
        performance_data = {
            "functions": [
                {"name": "func1", "execution_time": 0.001, "calls": 100},
                {"name": "func2", "execution_time": 0.5, "calls": 10},
                {"name": "func3", "execution_time": 0.1, "calls": 50}
            ]
        }
        
        bottlenecks = self.agent.detect_bottlenecks(performance_data)
        
        self.assertIsInstance(bottlenecks, list)
        self.assertGreater(len(bottlenecks), 0)
        # func2应该被识别为瓶颈（最高总执行时间）
        self.assertEqual(bottlenecks[0]["name"], "func2")


class TestPerformanceOptimization(PerformanceTestCase):
    """性能优化测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenPerformanceAgent()
    
    def test_optimization_suggestions(self):
        """测试优化建议生成"""
        def test_optimization():
            return self.agent.generate_optimization_suggestions({
                "slow_functions": ["func1", "func2"],
                "memory_issues": ["large_allocations"],
                "complexity_issues": ["nested_loops"]
            })
        
        suggestions, exec_time = self.measure_execution_time(test_optimization)
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        self.assert_performance_threshold("test_optimization", 1.0)
    
    def test_code_complexity_analysis(self):
        """测试代码复杂度分析"""
        complex_code = """
def complex_function(data):
    result = []
    for item in data:
        if item > 0:
            for i in range(item):
                if i % 2 == 0:
                    for j in range(i):
                        if j % 3 == 0:
                            result.append(i * j)
    return result
"""
        
        def test_complexity():
            return self.agent.analyze_complexity(complex_code)
        
        complexity, exec_time = self.measure_execution_time(test_complexity)
        
        self.assertIsInstance(complexity, dict)
        self.assertIn("cyclomatic_complexity", complexity)
        self.assertIn("nesting_depth", complexity)
        self.assertGreater(complexity["cyclomatic_complexity"], 1)


if __name__ == "__main__":
    unittest.main()
