"""
MAS系统集成测试
Integration tests for Multi-Agent System
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

from tests.base import IntegrationTestCase
from core.agents_integration import AgentIntegration


class TestMASIntegration(IntegrationTestCase):
    """MAS系统集成测试"""
    
    def setUp(self):
        super().setUp()
        self.integration = AgentIntegration()
        self.setup_test_agents()
    
    def setup_test_agents(self):
        """设置测试用的agents"""
        # 创建mock agents
        self.agents = {
            "code_quality": Mock(name="CodeQualityAgent"),
            "security": Mock(name="SecurityAgent"),
            "performance": Mock(name="PerformanceAgent"),
            "user_communication": Mock(name="UserCommunicationAgent"),
            "summary": Mock(name="SummaryAgent")
        }
        
        # 配置mock返回值
        self.agents["code_quality"].analyze.return_value = {
            "score": 85,
            "issues": ["naming_convention", "code_complexity"],
            "recommendations": ["改进变量命名", "简化复杂函数"]
        }
        
        self.agents["security"].analyze.return_value = {
            "score": 78,
            "vulnerabilities": ["sql_injection", "hardcoded_secrets"],
            "recommendations": ["使用参数化查询", "移除硬编码密钥"]
        }
        
        self.agents["performance"].analyze.return_value = {
            "score": 72,
            "bottlenecks": ["nested_loops", "inefficient_algorithms"],
            "recommendations": ["优化循环结构", "改进算法效率"]
        }
    
    def test_full_analysis_pipeline(self):
        """测试完整分析流程"""
        # 创建测试代码文件
        test_file = self.temp_dir / "sample_code.py"
        test_file.write_text("""
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def database_query(user_id):
    import sqlite3
    conn = sqlite3.connect('test.db')
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return conn.execute(query).fetchone()

class DataProcessor:
    def __init__(self):
        self.api_key = "hardcoded-api-key-123"
    
    def process_data(self, data):
        result = []
        for item in data:
            for i in range(len(item)):
                for j in range(len(item[i])):
                    result.append(item[i][j] * 2)
        return result
""")
        
        # 执行完整分析
        with patch.object(self.integration, 'agents', self.agents):
            results = self.integration.analyze_file(str(test_file))
        
        # 验证结果
        self.assertIsInstance(results, dict)
        self.assertIn("file_path", results)
        self.assertIn("analysis_results", results)
        
        # 验证各个agent都被调用
        for agent_name, agent in self.agents.items():
            if agent_name != "summary":  # summary agent单独处理
                agent.analyze.assert_called()
    
    def test_agent_coordination(self):
        """测试agent协调机制"""
        # 模拟agent之间的数据传递
        input_data = {
            "code": "def test(): pass",
            "file_path": "/test/file.py"
        }
        
        with patch.object(self.integration, 'agents', self.agents):
            # 执行协调分析
            coordination_result = self.integration.coordinate_analysis(input_data)
        
        self.assertIsInstance(coordination_result, dict)
        self.assertIn("coordinated_results", coordination_result)
        self.assertIn("cross_agent_insights", coordination_result)
    
    def test_parallel_analysis(self):
        """测试并行分析能力"""
        test_files = []
        for i in range(3):
            test_file = self.temp_dir / f"test_file_{i}.py"
            test_file.write_text(f"""
def function_{i}():
    return {i} * 2
""")
            test_files.append(str(test_file))
        
        with patch.object(self.integration, 'agents', self.agents):
            # 执行并行分析
            parallel_results = self.integration.analyze_multiple_files(test_files)
        
        self.assertIsInstance(parallel_results, list)
        self.assertEqual(len(parallel_results), 3)
        
        # 验证每个文件都得到了分析结果
        for result in parallel_results:
            self.assertIn("file_path", result)
            self.assertIn("analysis_results", result)
    
    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复机制"""
        # 模拟agent错误
        self.agents["security"].analyze.side_effect = Exception("Security analysis failed")
        
        test_file = self.temp_dir / "error_test.py"
        test_file.write_text("def test(): pass")
        
        with patch.object(self.integration, 'agents', self.agents):
            # 即使某个agent失败，系统应该继续运行
            results = self.integration.analyze_file(str(test_file))
        
        # 验证系统继续运行并记录错误
        self.assertIsInstance(results, dict)
        self.assertIn("errors", results)
        self.assertIn("security", results["errors"])
    
    def test_configuration_management(self):
        """测试配置管理"""
        # 测试配置加载
        config = self.integration.load_configuration()
        
        self.assertIsInstance(config, dict)
        self.assertIn("models", config)
        self.assertIn("agents", config)
        
        # 测试配置更新
        new_config = {
            "models": {"default": "Qwen/Qwen1.5-7B-Chat"},
            "timeout": 60
        }
        
        self.integration.update_configuration(new_config)
        updated_config = self.integration.get_current_configuration()
        
        self.assertEqual(updated_config["timeout"], 60)
    
    def test_result_aggregation(self):
        """测试结果聚合"""
        # 模拟多个分析结果
        individual_results = {
            "code_quality": {
                "score": 85,
                "issues": 5,
                "recommendations": ["改进命名", "减少复杂度"]
            },
            "security": {
                "score": 78,
                "vulnerabilities": 3,
                "recommendations": ["修复SQL注入", "加密敏感数据"]
            },
            "performance": {
                "score": 72,
                "bottlenecks": 4,
                "recommendations": ["优化算法", "减少内存使用"]
            }
        }
        
        # 执行聚合
        aggregated = self.integration.aggregate_results(individual_results)
        
        self.assertIsInstance(aggregated, dict)
        self.assertIn("overall_score", aggregated)
        self.assertIn("summary", aggregated)
        self.assertIn("prioritized_recommendations", aggregated)


class TestWorkflowIntegration(IntegrationTestCase):
    """工作流集成测试"""
    
    def setUp(self):
        super().setUp()
        self.integration = AgentIntegration()
    
    def test_complete_analysis_workflow(self):
        """测试完整分析工作流"""
        # 创建测试项目结构
        project_dir = self.temp_dir / "test_project"
        project_dir.mkdir()
        
        # 创建多个测试文件
        (project_dir / "main.py").write_text("""
import os
import sqlite3

API_KEY = "secret-key-123"

def get_user_data(user_id):
    conn = sqlite3.connect('users.db')
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return conn.execute(query).fetchone()

def inefficient_sort(data):
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i] < data[j]:
                data[i], data[j] = data[j], data[i]
    return data
""")
        
        (project_dir / "utils.py").write_text("""
def helper_function(x, y):
    return x + y

class UtilityClass:
    def method1(self):
        pass
    
    def method2(self):
        pass
""")
        
        # 执行完整工作流
        workflow_result = self.integration.run_complete_workflow(str(project_dir))
        
        # 验证工作流结果
        self.assertIsInstance(workflow_result, dict)
        self.assertIn("project_analysis", workflow_result)
        self.assertIn("summary_report", workflow_result)
        self.assertIn("user_report", workflow_result)
        self.assertIn("execution_metadata", workflow_result)
    
    def test_custom_workflow_definition(self):
        """测试自定义工作流定义"""
        # 定义自定义工作流
        custom_workflow = {
            "name": "security_focused_analysis",
            "steps": [
                {"agent": "security", "priority": "high"},
                {"agent": "code_quality", "priority": "medium"},
                {"agent": "summary", "priority": "high"}
            ],
            "configuration": {
                "security": {"deep_scan": True},
                "code_quality": {"focus_areas": ["security_patterns"]}
            }
        }
        
        test_file = self.temp_dir / "security_test.py"
        test_file.write_text("def vulnerable_function(): pass")
        
        # 执行自定义工作流
        result = self.integration.execute_custom_workflow(
            custom_workflow, str(test_file)
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("workflow_name", result)
        self.assertEqual(result["workflow_name"], "security_focused_analysis")


if __name__ == "__main__":
    unittest.main()
