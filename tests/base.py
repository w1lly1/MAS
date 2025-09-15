"""
MAS测试基类
Base classes for MAS testing framework
"""

import unittest
import time
import logging
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

from tests import TEST_CONFIG, setup_test_environment, cleanup_test_environment


class MASTestCase(unittest.TestCase):
    """MAS系统测试基类"""
    
    def setUp(self):
        """测试前置设置"""
        self.start_time = time.time()
        self.temp_dir = setup_test_environment()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mock_data = {}
        
    def tearDown(self):
        """测试后置清理"""
        duration = time.time() - self.start_time
        self.logger.info(f"Test {self._testMethodName} completed in {duration:.2f}s")
        cleanup_test_environment()
    
    def create_mock_model(self, model_name: str = "test-model") -> Mock:
        """创建模拟模型"""
        mock_model = Mock()
        mock_model.name = model_name
        mock_model.generate.return_value = "Mock response"
        return mock_model
    
    def create_mock_agent_config(self) -> Dict[str, Any]:
        """创建模拟agent配置"""
        return {
            "model_name": "Qwen/Qwen1.5-7B-Chat",
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 30,
        }


class AgentTestCase(MASTestCase):
    """Agent测试基类"""
    
    def setUp(self):
        super().setUp()
        self.agent = None
        self.mock_model = self.create_mock_model()
        self.agent_config = self.create_mock_agent_config()
    
    def assert_agent_initialized(self):
        """断言agent正确初始化"""
        self.assertIsNotNone(self.agent)
        self.assertTrue(hasattr(self.agent, 'process'))
        self.assertTrue(hasattr(self.agent, 'model_name'))
    
    def assert_valid_response(self, response: str):
        """断言有效的响应"""
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertNotIn("error", response.lower())


class IntegrationTestCase(MASTestCase):
    """集成测试基类"""
    
    def setUp(self):
        super().setUp()
        self.agents = {}
        self.test_data = {}
    
    def setup_test_agents(self):
        """设置测试用的agents"""
        # 子类实现具体的agent设置
        pass
    
    def simulate_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟工作流程"""
        # 子类实现具体的工作流程
        return {"status": "completed", "result": "mock_result"}


class PerformanceTestCase(MASTestCase):
    """性能测试基类"""
    
    def setUp(self):
        super().setUp()
        self.performance_metrics = {}
    
    def measure_execution_time(self, func, *args, **kwargs):
        """测量执行时间"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        func_name = getattr(func, '__name__', str(func))
        self.performance_metrics[func_name] = {
            'execution_time': execution_time,
            'timestamp': start_time
        }
        
        return result, execution_time
    
    def assert_performance_threshold(self, func_name: str, max_time: float):
        """断言性能阈值"""
        if func_name in self.performance_metrics:
            actual_time = self.performance_metrics[func_name]['execution_time']
            self.assertLessEqual(
                actual_time, 
                max_time, 
                f"{func_name} took {actual_time:.2f}s, expected <= {max_time:.2f}s"
            )
