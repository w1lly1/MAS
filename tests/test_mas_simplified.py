"""
MAS测试套件 - 精简版
基于实际代码接口的务实测试
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from tests.base import MASTestCase


class TestMASAgentInitialization(MASTestCase):
    """测试MAS Agent初始化"""
    
    def setUp(self):
        super().setUp()
        # Mock数据库和配置
        self.mock_db_service = patch('infrastructure.database.service.DatabaseService')
        self.mock_config = patch('infrastructure.config.settings.HUGGINGFACE_CONFIG', {
            'models': {
                'code_quality': {'name': 'microsoft/codebert-base'},
                'security': {'name': 'microsoft/codebert-base'},
                'performance': {'name': 'microsoft/codebert-base'},
                'user_communication': {'name': 'microsoft/codebert-base'}
            },
            'cache_dir': str(self.temp_dir)
        })
        
        self.mock_db_service.start()
        self.mock_config.start()
    
    def tearDown(self):
        super().tearDown()
        self.mock_db_service.stop()
        self.mock_config.stop()
    
    def test_code_quality_agent_basic_init(self):
        """测试代码质量Agent基本初始化"""
        try:
            from core.agents.ai_driven_code_quality_agent import AIDrivenCodeQualityAgent
            
            agent = AIDrivenCodeQualityAgent()
            
            # 验证基本属性存在
            self.assertIsNotNone(agent.agent_id)
            self.assertIsNotNone(agent.name)
            self.assertIsNotNone(agent.status)
            self.assertIsNotNone(agent.db_service)
            
            # 验证基本方法存在（从BaseAgent继承）
            self.assertTrue(hasattr(agent, 'start'))
            self.assertTrue(hasattr(agent, 'stop'))
            self.assertTrue(hasattr(agent, 'execute_task'))
            
        except Exception as e:
            self.fail(f"代码质量Agent初始化失败: {e}")
    
    def test_security_agent_basic_init(self):
        """测试安全分析Agent基本初始化"""
        try:
            from core.agents.ai_driven_security_agent import AIDrivenSecurityAgent
            
            agent = AIDrivenSecurityAgent()
            
            # 验证基本属性
            self.assertIsNotNone(agent.agent_id)
            self.assertIsNotNone(agent.name)
            
        except Exception as e:
            self.fail(f"安全分析Agent初始化失败: {e}")
    
    def test_performance_agent_basic_init(self):
        """测试性能分析Agent基本初始化"""
        try:
            from core.agents.ai_driven_performance_agent import AIDrivenPerformanceAgent
            
            agent = AIDrivenPerformanceAgent()
            
            # 验证基本属性
            self.assertIsNotNone(agent.agent_id)
            self.assertIsNotNone(agent.name)
            
        except Exception as e:
            self.fail(f"性能分析Agent初始化失败: {e}")
    
    def test_user_communication_agent_basic_init(self):
        """测试用户沟通Agent基本初始化"""
        try:
            from core.agents.ai_driven_user_communication_agent import AIDrivenUserCommunicationAgent
            
            agent = AIDrivenUserCommunicationAgent()
            
            # 验证基本属性
            self.assertIsNotNone(agent.agent_id)
            self.assertIsNotNone(agent.name)
            
        except Exception as e:
            self.fail(f"用户沟通Agent初始化失败: {e}")
    
    def test_summary_agent_basic_init(self):
        """测试汇总Agent基本初始化"""
        try:
            from core.agents.analysis_result_summary_agent import SummaryAgent
            
            agent = SummaryAgent()
            
            # 验证基本属性
            self.assertIsNotNone(agent.agent_id)
            self.assertIsNotNone(agent.name)
            
        except Exception as e:
            self.fail(f"汇总Agent初始化失败: {e}")


class TestMASAgentManager(MASTestCase):
    """测试MAS Agent管理器"""
    
    def test_agent_manager_init(self):
        """测试Agent管理器初始化"""
        try:
            from core.agents.agent_manager import AgentManager
            
            manager = AgentManager()
            
            # 验证基本属性
            self.assertTrue(hasattr(manager, 'agents'))
            
        except Exception as e:
            self.fail(f"Agent管理器初始化失败: {e}")


class TestMASIntegrationModule(MASTestCase):
    """测试MAS集成模块"""
    
    def test_agent_integration_import(self):
        """测试Agent集成模块导入"""
        try:
            from core.agents_integration import AgentIntegration
            
            integration = AgentIntegration()
            self.assertIsNotNone(integration)
            
        except Exception as e:
            self.fail(f"Agent集成模块导入失败: {e}")


class TestMASAsyncOperations(MASTestCase):
    """测试MAS异步操作"""
    
    def setUp(self):
        super().setUp()
        # Mock所有外部依赖
        self.mock_patches = []
        
        # Mock数据库
        db_patch = patch('infrastructure.database.service.DatabaseService')
        self.mock_patches.append(db_patch)
        db_patch.start()
        
        # Mock配置
        config_patch = patch('infrastructure.config.settings.HUGGINGFACE_CONFIG', {
            'models': {'code_quality': {'name': 'test-model'}},
            'cache_dir': str(self.temp_dir)
        })
        self.mock_patches.append(config_patch)
        config_patch.start()
    
    def tearDown(self):
        super().tearDown()
        for patch in self.mock_patches:
            patch.stop()
    
    def test_agent_async_start_stop(self):
        """测试Agent异步启动停止"""
        try:
            from core.agents.ai_driven_code_quality_agent import AIDrivenCodeQualityAgent
            
            agent = AIDrivenCodeQualityAgent()
            
            # 测试异步方法（如果有的话）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # 测试启动
                if hasattr(agent, 'start') and asyncio.iscoroutinefunction(agent.start):
                    loop.run_until_complete(agent.start())
                    
                # 测试停止
                if hasattr(agent, 'stop') and asyncio.iscoroutinefunction(agent.stop):
                    loop.run_until_complete(agent.stop())
                    
                self.assertTrue(True)  # 如果到这里没有异常，测试通过
                
            finally:
                loop.close()
                
        except Exception as e:
            self.fail(f"Agent异步操作测试失败: {e}")


class TestMASConfiguration(MASTestCase):
    """测试MAS配置系统"""
    
    def test_settings_import(self):
        """测试设置模块导入"""
        try:
            from infrastructure.config import settings
            
            # 验证配置存在
            self.assertTrue(hasattr(settings, 'HUGGINGFACE_CONFIG'))
            
        except Exception as e:
            self.fail(f"设置模块导入失败: {e}")
    
    def test_prompts_import(self):
        """测试提示词模块导入"""
        try:
            from infrastructure.config import prompts
            
            # 验证提示词功能存在
            self.assertTrue(hasattr(prompts, 'get_prompt'))
            
        except Exception as e:
            self.fail(f"提示词模块导入失败: {e}")
    
    def test_ai_agent_config_import(self):
        """测试AI Agent配置导入"""
        try:
            from core.ai_agent_config import AIAgentConfig
            
            config = AIAgentConfig()
            self.assertIsNotNone(config)
            
        except Exception as e:
            self.fail(f"AI Agent配置导入失败: {e}")


class TestMASDatabase(MASTestCase):
    """测试MAS数据库系统"""
    
    def test_database_models_import(self):
        """测试数据库模型导入"""
        try:
            from infrastructure.database import models
            
            # 验证基本模型存在
            self.assertTrue(hasattr(models, 'Base'))
            
        except Exception as e:
            self.fail(f"数据库模型导入失败: {e}")
    
    def test_database_service_import(self):
        """测试数据库服务导入"""
        try:
            from infrastructure.database.service import DatabaseService
            
            # 只测试导入，不创建实例避免数据库连接
            self.assertTrue(hasattr(DatabaseService, '__init__'))
            self.assertTrue(hasattr(DatabaseService, 'get_session'))
            
        except Exception as e:
            self.fail(f"数据库服务导入失败: {e}")


if __name__ == "__main__":
    unittest.main()
