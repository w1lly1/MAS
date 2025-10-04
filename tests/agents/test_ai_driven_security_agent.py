"""
AI驱动安全分析Agent测试
Tests for AI-driven Security Analysis Agent
"""

import unittest
from unittest.mock import Mock, patch

from tests.base import AgentTestCase
from core.agents.ai_driven_security_agent import AIDrivenSecurityAgent


class TestAISecurityAgent(AgentTestCase):
    """AI安全分析Agent测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenSecurityAgent()
    
    def test_agent_initialization(self):
        """测试agent初始化"""
        self.assert_agent_initialized()
        self.assertTrue(bool(self.agent.name))
    
    @patch('core.agents.ai_driven_security_agent.AutoTokenizer')
    @patch('core.agents.ai_driven_security_agent.AutoModelForCausalLM')
    def test_security_vulnerability_scan(self, mock_model, mock_tokenizer):
        """测试安全漏洞扫描"""
        # 设置mock
        mock_tokenizer.from_pretrained.return_value = self.mock_model
        mock_model.from_pretrained.return_value = self.mock_model
        self.mock_model.generate.return_value = [Mock()]
        self.mock_model.decode.return_value = "发现安全漏洞：SQL注入风险..."
        
        # 测试代码（包含潜在安全漏洞）
        vulnerable_code = """
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()

def login(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    # 直接执行用户输入，存在SQL注入风险
"""
        
        # 执行扫描
        result = self.agent.scan_vulnerabilities(vulnerable_code)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("vulnerabilities", result)
        self.assertIn("security_score", result)
        self.assertIn("recommendations", result)
    
    def test_sql_injection_detection(self):
        """测试SQL注入检测"""
        sql_injection_code = """
def unsafe_query(user_input):
    query = f"SELECT * FROM table WHERE column = '{user_input}'"
    cursor.execute(query)
"""
        
        vulnerabilities = self.agent.detect_sql_injection(sql_injection_code)
        
        self.assertIsInstance(vulnerabilities, list)
        self.assertGreater(len(vulnerabilities), 0)
        self.assertIn("sql_injection", vulnerabilities[0]["type"])
    
    def test_xss_vulnerability_detection(self):
        """测试XSS漏洞检测"""
        xss_code = """
from flask import Flask, request, render_template_string

@app.route('/user')
def user_profile():
    username = request.args.get('username')
    return f"<h1>Welcome {username}</h1>"  # 直接输出用户输入
"""
        
        vulnerabilities = self.agent.detect_xss(xss_code)
        
        self.assertIsInstance(vulnerabilities, list)
        if vulnerabilities:  # 如果检测到XSS漏洞
            self.assertIn("xss", vulnerabilities[0]["type"])
    
    def test_insecure_deserialization_detection(self):
        """测试不安全反序列化检测"""
        deserialization_code = """
import pickle

def load_data(data):
    return pickle.loads(data)  # 不安全的反序列化
"""
        
        vulnerabilities = self.agent.detect_insecure_deserialization(deserialization_code)
        
        self.assertIsInstance(vulnerabilities, list)
        if vulnerabilities:
            self.assertIn("deserialization", vulnerabilities[0]["type"])
    
    def test_hardcoded_secrets_detection(self):
        """测试硬编码密钥检测"""
        secrets_code = """
API_KEY = "sk-1234567890abcdef"
DATABASE_PASSWORD = "password123"
JWT_SECRET = "my-secret-key"

def connect_to_api():
    headers = {"Authorization": f"Bearer {API_KEY}"}
"""
        
        secrets = self.agent.detect_hardcoded_secrets(secrets_code)
        
        self.assertIsInstance(secrets, list)
        self.assertGreater(len(secrets), 0)
    
    def test_security_score_calculation(self):
        """测试安全评分计算"""
        vulnerability_data = {
            "critical": 1,
            "high": 2,
            "medium": 3,
            "low": 5,
            "info": 10
        }
        
        score = self.agent.calculate_security_score(vulnerability_data)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_generate_security_report(self):
        """测试安全报告生成"""
        scan_results = {
            "vulnerabilities": [
                {
                    "type": "sql_injection",
                    "severity": "high",
                    "line": 5,
                    "description": "SQL注入漏洞"
                }
            ],
            "security_score": 65,
            "scan_time": "2024-01-01 12:00:00"
        }
        
        report = self.agent.generate_security_report(scan_results)
        
        self.assertIsInstance(report, dict)
        self.assertIn("summary", report)
        self.assertIn("detailed_findings", report)
        self.assertIn("recommendations", report)


class TestSecurityIntegration(AgentTestCase):
    """安全分析集成测试"""
    
    def setUp(self):
        super().setUp()
        self.agent = AIDrivenSecurityAgent()
    
    def test_comprehensive_security_analysis(self):
        """测试综合安全分析"""
        # 创建包含多种安全问题的测试文件
        test_file = self.temp_dir / "vulnerable_app.py"
        test_file.write_text("""
import pickle
import sqlite3
from flask import Flask, request

app = Flask(__name__)

# 硬编码密钥
SECRET_KEY = "hardcoded-secret-123"
DATABASE_URL = "postgresql://user:password@localhost/db"

def unsafe_query(user_id):
    # SQL注入漏洞
    conn = sqlite3.connect('app.db')
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return conn.execute(query).fetchone()

@app.route('/profile')
def profile():
    # XSS漏洞
    name = request.args.get('name')
    return f"<h1>Hello {name}</h1>"

def load_user_data(data):
    # 不安全反序列化
    return pickle.loads(data)
""")
        
        # 执行综合分析
        result = self.agent.analyze_file(str(test_file))
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("file_path", result)
        self.assertIn("vulnerabilities", result)
        self.assertIn("security_score", result)
        self.assertGreater(len(result["vulnerabilities"]), 0)


if __name__ == "__main__":
    unittest.main()
