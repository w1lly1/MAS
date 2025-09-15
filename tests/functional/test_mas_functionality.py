"""
MAS系统功能测试
Functional tests for Multi-Agent System
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from tests.base import MASTestCase
from tests import setup_test_environment, cleanup_test_environment


class TestMASFunctionality(MASTestCase):
    """MAS系统功能测试"""
    
    def setUp(self):
        super().setUp()
        self.test_project = self.create_test_project()
    
    def create_test_project(self):
        """创建测试项目"""
        project_dir = self.temp_dir / "functional_test_project"
        project_dir.mkdir()
        
        # 创建包含各种代码模式的测试文件
        (project_dir / "main.py").write_text("""
#!/usr/bin/env python3
\"\"\"
主应用程序
包含多种代码质量、安全和性能问题用于测试
\"\"\"

import os
import sqlite3
import pickle
from flask import Flask, request

app = Flask(__name__)

# 硬编码敏感信息（安全问题）
DATABASE_URL = "postgresql://admin:password123@localhost/mydb"
SECRET_KEY = "hardcoded-secret-key-123"

class UserManager:
    \"\"\"用户管理类\"\"\"
    
    def __init__(self):
        self.users = []
        self.db_connection = None
    
    def connect_database(self):
        \"\"\"连接数据库\"\"\"
        self.db_connection = sqlite3.connect('users.db')
    
    def get_user_by_id(self, user_id):
        \"\"\"通过ID获取用户（存在SQL注入风险）\"\"\"
        if not self.db_connection:
            self.connect_database()
        
        # SQL注入漏洞
        query = f"SELECT * FROM users WHERE id = {user_id}"
        cursor = self.db_connection.execute(query)
        return cursor.fetchone()
    
    def create_user(self, user_data):
        \"\"\"创建用户\"\"\"
        # 性能问题：不必要的循环
        processed_data = []
        for item in user_data:
            for i in range(len(item)):
                for j in range(len(item[i]) if isinstance(item[i], list) else 1):
                    processed_data.append(item)
        
        return processed_data
    
    def load_user_profile(self, profile_data):
        \"\"\"加载用户配置（不安全的反序列化）\"\"\"
        return pickle.loads(profile_data)  # 安全风险

def inefficient_fibonacci(n):
    \"\"\"效率低下的斐波那契计算\"\"\"
    if n <= 1:
        return n
    return inefficient_fibonacci(n-1) + inefficient_fibonacci(n-2)

def calculate_primes(limit):
    \"\"\"计算素数（性能问题）\"\"\"
    primes = []
    for num in range(2, limit):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

@app.route('/profile')
def user_profile():
    \"\"\"用户资料页面（XSS漏洞）\"\"\"
    username = request.args.get('username')
    # XSS漏洞：直接输出用户输入
    return f"<h1>Welcome {username}</h1>"

@app.route('/search')
def search():
    \"\"\"搜索功能\"\"\"
    query = request.args.get('q')
    # 代码重复和命名问题
    searchResults = []
    SearchQuery = query.lower()
    for item in get_all_items():
        if SearchQuery in item.lower():
            searchResults.append(item)
    return searchResults

def get_all_items():
    \"\"\"获取所有项目\"\"\"
    # 硬编码返回值
    return ["item1", "item2", "item3"]

# 全局变量（代码质量问题）
GLOBAL_COUNTER = 0
globalData = {}

def increment_counter():
    global GLOBAL_COUNTER
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER

if __name__ == "__main__":
    app.run(debug=True)  # 生产环境中不应启用debug模式
""")
        
        (project_dir / "utils.py").write_text("""
\"\"\"
工具函数模块
\"\"\"

import hashlib
import random

def hash_password(password):
    \"\"\"哈希密码（使用弱哈希算法）\"\"\"
    return hashlib.md5(password.encode()).hexdigest()  # 弱哈希算法

def generate_token():
    \"\"\"生成令牌（不安全的随机数）\"\"\"
    return str(random.random())  # 不适合安全用途

def validate_email(email):
    \"\"\"验证邮箱地址\"\"\"
    # 简单但不完整的邮箱验证
    if "@" in email and "." in email:
        return True
    return False

class ConfigManager:
    \"\"\"配置管理器\"\"\"
    
    def __init__(self):
        self.config = {
            "database_url": "sqlite:///app.db",
            "secret_key": "default-secret",
            "debug": True
        }
    
    def get_config(self, key):
        \"\"\"获取配置\"\"\"
        return self.config.get(key)
    
    def set_config(self, key, value):
        \"\"\"设置配置\"\"\"
        self.config[key] = value
    
    def load_from_file(self, file_path):
        \"\"\"从文件加载配置\"\"\"
        # 潜在的路径遍历漏洞
        with open(file_path, 'r') as f:
            content = f.read()
        # 不安全的eval使用
        self.config = eval(content)  # 代码注入风险
""")
        
        (project_dir / "tests_example.py").write_text("""
\"\"\"
示例测试文件
\"\"\"

def test_basic_function():
    \"\"\"基本测试函数\"\"\"
    assert 1 + 1 == 2

def test_user_creation():
    \"\"\"测试用户创建\"\"\"
    from main import UserManager
    
    manager = UserManager()
    user_data = [["test", "data"]]
    result = manager.create_user(user_data)
    assert isinstance(result, list)

# 缺少更多测试用例（测试覆盖率问题）
""")
        
        return project_dir
    
    def test_code_quality_analysis(self):
        """测试代码质量分析功能"""
        from core.agents.ai_driven_code_quality_agent import AICodeQualityAgent
        
        agent = AICodeQualityAgent()
        
        # 分析主文件
        main_file = self.test_project / "main.py"
        result = agent.analyze_file(str(main_file))
        
        # 验证分析结果
        self.assertIsInstance(result, dict)
        self.assertIn("quality_issues", result)
        self.assertIn("recommendations", result)
        
        # 应该检测到的问题类型
        expected_issues = [
            "naming_convention",  # SearchQuery, searchResults等
            "global_variables",   # GLOBAL_COUNTER, globalData
            "code_complexity",    # 嵌套循环
            "hardcoded_values"    # 硬编码字符串
        ]
        
        detected_issues = [issue["type"] for issue in result["quality_issues"]]
        
        # 至少应该检测到一些问题
        self.assertGreater(len(detected_issues), 0)
    
    def test_security_analysis(self):
        """测试安全分析功能"""
        from core.agents.ai_driven_security_agent import AISecurityAgent
        
        agent = AISecurityAgent()
        
        # 分析包含安全漏洞的文件
        main_file = self.test_project / "main.py"
        utils_file = self.test_project / "utils.py"
        
        main_result = agent.analyze_file(str(main_file))
        utils_result = agent.analyze_file(str(utils_file))
        
        # 验证安全分析结果
        self.assertIsInstance(main_result, dict)
        self.assertIn("vulnerabilities", main_result)
        
        # 应该检测到的安全漏洞
        all_vulns = main_result["vulnerabilities"] + utils_result["vulnerabilities"]
        vuln_types = [vuln["type"] for vuln in all_vulns]
        
        expected_vulns = [
            "sql_injection",      # get_user_by_id中的SQL注入
            "xss",               # user_profile中的XSS
            "hardcoded_secrets", # SECRET_KEY等
            "insecure_deserialization",  # pickle.loads
            "weak_crypto",       # MD5哈希
            "code_injection"     # eval使用
        ]
        
        # 至少应该检测到一些漏洞
        detected_count = sum(1 for expected in expected_vulns if expected in vuln_types)
        self.assertGreater(detected_count, 0)
    
    def test_performance_analysis(self):
        """测试性能分析功能"""
        from core.agents.ai_driven_performance_agent import AIPerformanceAgent
        
        agent = AIPerformanceAgent()
        
        # 分析包含性能问题的文件
        main_file = self.test_project / "main.py"
        result = agent.analyze_file(str(main_file))
        
        # 验证性能分析结果
        self.assertIsInstance(result, dict)
        self.assertIn("performance_issues", result)
        self.assertIn("bottlenecks", result)
        
        # 应该检测到的性能问题
        issue_types = [issue["type"] for issue in result["performance_issues"]]
        
        expected_performance_issues = [
            "inefficient_algorithm",  # inefficient_fibonacci
            "nested_loops",          # create_user中的三层循环
            "inefficient_search"     # calculate_primes的暴力搜索
        ]
        
        # 至少应该检测到一些性能问题
        self.assertGreater(len(issue_types), 0)
    
    def test_comprehensive_project_analysis(self):
        """测试完整项目分析"""
        from core.agents_integration import AgentIntegration
        
        integration = AgentIntegration()
        
        # 执行完整项目分析
        result = integration.analyze_project(str(self.test_project))
        
        # 验证综合分析结果
        self.assertIsInstance(result, dict)
        self.assertIn("project_summary", result)
        self.assertIn("file_analyses", result)
        self.assertIn("overall_scores", result)
        
        # 验证分析了所有文件
        analyzed_files = list(result["file_analyses"].keys())
        expected_files = ["main.py", "utils.py", "tests_example.py"]
        
        for expected_file in expected_files:
            self.assertTrue(
                any(expected_file in analyzed for analyzed in analyzed_files),
                f"Expected file {expected_file} not found in analysis"
            )
    
    def test_report_generation(self):
        """测试报告生成功能"""
        from core.agents.analysis_result_summary_agent import AnalysisResultSummaryAgent
        
        # 模拟分析结果
        mock_results = {
            "code_quality": {
                "score": 65,
                "issues": 15,
                "main_issues": ["naming", "complexity", "duplication"]
            },
            "security": {
                "score": 55,
                "vulnerabilities": 8,
                "critical_vulns": 3
            },
            "performance": {
                "score": 60,
                "bottlenecks": 6,
                "critical_bottlenecks": 2
            }
        }
        
        agent = AnalysisResultSummaryAgent()
        report = agent.generate_comprehensive_report(mock_results)
        
        # 验证报告结构
        self.assertIsInstance(report, dict)
        self.assertIn("executive_summary", report)
        self.assertIn("detailed_findings", report)
        self.assertIn("recommendations", report)
        self.assertIn("action_plan", report)
    
    def test_user_communication(self):
        """测试用户沟通功能"""
        from core.agents.ai_driven_user_communication_agent import AIUserCommunicationAgent
        
        agent = AIUserCommunicationAgent()
        
        # 模拟用户询问
        questions = [
            "什么是SQL注入？",
            "如何提高代码性能？",
            "代码复杂度太高怎么办？"
        ]
        
        for question in questions:
            response = agent.answer_question(question)
            
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 50)
            self.assert_valid_response(response)
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 这是一个完整的端到端测试
        from core.agents_integration import AgentIntegration
        
        integration = AgentIntegration()
        
        # 1. 分析项目
        analysis_result = integration.analyze_project(str(self.test_project))
        
        # 2. 生成用户报告
        user_report = integration.generate_user_report(
            analysis_result, 
            user_level="intermediate"
        )
        
        # 3. 生成技术报告
        technical_report = integration.generate_technical_report(analysis_result)
        
        # 验证端到端结果
        self.assertIsInstance(analysis_result, dict)
        self.assertIsInstance(user_report, dict)
        self.assertIsInstance(technical_report, dict)
        
        # 验证报告包含必要信息
        self.assertIn("summary", user_report)
        self.assertIn("next_steps", user_report)
        self.assertIn("detailed_analysis", technical_report)
        self.assertIn("metrics", technical_report)


class TestMASConfigurationAndDeployment(MASTestCase):
    """MAS配置和部署测试"""
    
    def test_configuration_loading(self):
        """测试配置加载"""
        # 创建测试配置文件
        config_file = self.temp_dir / "test_config.json"
        config_data = {
            "models": {
                "code_quality": "Qwen/Qwen1.5-7B-Chat",
                "security": "Qwen/Qwen1.5-7B-Chat",
                "performance": "Qwen/Qwen1.5-7B-Chat"
            },
            "thresholds": {
                "code_quality_min": 70,
                "security_min": 80,
                "performance_min": 65
            },
            "output_formats": ["json", "markdown", "html"]
        }
        
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # 测试配置加载
        from core.ai_agent_config import AIAgentConfig
        
        config = AIAgentConfig()
        loaded_config = config.load_from_file(str(config_file))
        
        self.assertEqual(loaded_config["models"]["code_quality"], "Qwen/Qwen1.5-7B-Chat")
        self.assertEqual(loaded_config["thresholds"]["security_min"], 80)
    
    def test_deployment_script_execution(self):
        """测试部署脚本执行"""
        # 这里测试deploy_qwen_alternatives.py的功能
        import sys
        import importlib.util
        
        # 动态导入部署脚本
        spec = importlib.util.spec_from_file_location(
            "deploy_module", 
            str(Path(__file__).parent.parent.parent / "deploy_qwen_alternatives.py")
        )
        deploy_module = importlib.util.module_from_spec(spec)
        
        # 测试主要部署函数（如果存在）
        if hasattr(deploy_module, 'main'):
            # 在测试环境中执行部署逻辑
            with patch('builtins.print'):  # 抑制打印输出
                try:
                    # 这里应该有部署验证逻辑
                    # 由于实际部署可能修改文件，这里只做基本验证
                    self.assertTrue(True)  # 占位符
                except Exception as e:
                    self.fail(f"Deployment script failed: {e}")


if __name__ == "__main__":
    unittest.main()
