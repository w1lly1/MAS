"""
基础Agent测试示例
用于验证测试框架工作正常
"""

import unittest
from tests.base import AgentTestCase


class TestBasicAgentFunctionality(AgentTestCase):
    """基础Agent功能测试"""
    
    def setUp(self):
        super().setUp()
        # 创建一个简单的mock agent
        self.agent = type('MockAgent', (), {
            'name': 'Test Agent',
            'process': lambda self, data: {"result": "processed"},
            'model_name': 'test-model'
        })()
    
    def test_agent_initialization(self):
        """测试agent初始化"""
        self.assert_agent_initialized()
        self.assertEqual(self.agent.name, "Test Agent")
    
    def test_agent_processing(self):
        """测试agent数据处理"""
        result = self.agent.process("test data")
        self.assertIsInstance(result, dict)
        self.assertIn("result", result)
        self.assertEqual(result["result"], "processed")
    
    def test_mock_model_creation(self):
        """测试mock模型创建"""
        mock_model = self.create_mock_model("test-model")
        self.assertEqual(mock_model.name, "test-model")
        self.assertEqual(mock_model.generate.return_value, "Mock response")
    
    def test_agent_config_creation(self):
        """测试agent配置创建"""
        config = self.create_mock_agent_config()
        self.assertIsInstance(config, dict)
        self.assertIn("model_name", config)
        self.assertIn("temperature", config)


class TestFrameworkUtilities(AgentTestCase):
    """测试框架工具测试"""
    
    def test_temp_directory_creation(self):
        """测试临时目录创建"""
        self.assertTrue(self.temp_dir.exists())
        self.assertTrue(self.temp_dir.is_dir())
    
    def test_valid_response_assertion(self):
        """测试有效响应断言"""
        valid_response = "这是一个有效的响应"
        self.assert_valid_response(valid_response)
        
        # 测试无效响应会抛出异常
        with self.assertRaises(AssertionError):
            self.assert_valid_response("")
        
        with self.assertRaises(AssertionError):
            self.assert_valid_response("发生了error")


if __name__ == "__main__":
    unittest.main()
