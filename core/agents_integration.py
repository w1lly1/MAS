import asyncio
import logging
import uuid
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from .agents import (
        AgentManager, 
        AIDrivenUserCommunicationAgent, 
        SummaryAgent
    )
    # 导入AI驱动智能体
    from .agents.ai_driven_code_quality_agent import AIDrivenCodeQualityAgent
    from .agents.ai_driven_security_agent import AIDrivenSecurityAgent
    from .agents.ai_driven_performance_agent import AIDrivenPerformanceAgent
    from .agents.static_scan_agent import StaticCodeScanAgent
    from .agents.base_agent import Message
    from .ai_agent_config import get_ai_agent_config, AgentMode
except ImportError as e:
    logger.error(f"❌ 导入智能体类失败: {e}")
    raise

class AgentIntegration:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.agent_manager = AgentManager.get_instance()
            self.agents = {}
            self._system_ready = False
            self.ai_config = get_ai_agent_config()
            self.__class__._initialized = True
        
    async def initialize_system(self):
        """初始化智能体系统"""
        if self._system_ready:
            return
            
        try:
            # 只使用AI驱动智能体
            agent_strategy = self.ai_config.get_agent_selection_strategy()
            mode = self.ai_config.get_agent_mode()
            
            print(f"🤖 初始化AI驱动智能体系统 - 模式: {mode.value}")
            
            # 定义AI驱动智能体类
            ai_agent_classes = {
                # 核心智能体（必需）
                'ai_user_comm': AIDrivenUserCommunicationAgent,
                'summary': SummaryAgent,
                # AI驱动分析智能体
                'static_scan': StaticCodeScanAgent,
                'ai_code_quality': AIDrivenCodeQualityAgent,
                'ai_security': AIDrivenSecurityAgent,
                'ai_performance': AIDrivenPerformanceAgent,
            }
            
            # 创建AI驱动智能体
            agents_to_create = {
                'ai_user_comm': AIDrivenUserCommunicationAgent,
                'summary': SummaryAgent
            }
            
            # 添加AI分析智能体
            for analysis_type, agent_name in agent_strategy.items():
                if agent_name in ai_agent_classes:
                    agents_to_create[agent_name] = ai_agent_classes[agent_name]
                else:
                    print(f"⚠️ 智能体 {agent_name} 未找到，跳过")
            
            # 创建智能体实例
            for name, agent_class in agents_to_create.items():
                try:
                    print(f"🔧 创建AI智能体: {name}")
                    self.agents[name] = agent_class()
                except Exception as e:
                    logger.error(f"❌ 创建智能体 {name} 失败: {e}")
                    print(f"⚠️ 跳过智能体 {name}: {e}")
                    continue
            
            # 注册到管理器
            for agent in self.agents.values():
                self.agent_manager.register_agent(agent)
                
            # 启动所有智能体
            await self.agent_manager.start_all_agents()
            
            # 初始化AI用户交流功能
            if 'ai_user_comm' in self.agents:
                try:
                    print("🧠 初始化AI用户交流功能...")
                    await self.agents['ai_user_comm'].initialize_ai_communication()
                except Exception as e:
                    logger.warning(f"⚠️ AI用户交流初始化失败: {e}")
            
            self._system_ready = True
            
            print(f"✅ 智能体系统初始化完成 - 共创建 {len(self.agents)} 个智能体")
            print(f"🎯 活跃智能体: {list(self.agents.keys())}")
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            await self._cleanup_on_error()
            raise
            
    async def _cleanup_on_error(self):
        """错误时清理资源"""
        try:
            if hasattr(self, 'agent_manager'):
                await self.agent_manager.stop_all_agents()
            self.agents.clear()
            self._system_ready = False
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")
        
    async def process_message_from_cli(self, message: str, target_dir: Optional[str] = None) -> str:
        """处理来自命令行的消息"""
        if not self._system_ready:
            return "❌ 智能体系统未就绪，请稍后重试"
            
        if 'ai_user_comm' not in self.agents:
            return "❌ AI用户沟通智能体不可用"
            
        try:
            # 构造消息内容
            content = {
                "message": message,
                "session_id": "cli_session",
                "target_directory": target_dir,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # 发送给用户沟通智能体处理
            await self.agents['ai_user_comm'].handle_message(Message(
                id=str(uuid.uuid4()),  # 生成唯一ID
                sender="cli_interface",
                receiver="ai_user_comm_agent",
                content=content,
                timestamp=asyncio.get_event_loop().time(),
                message_type="user_input"
            ))
            
            return "✅ 消息已处理"
                
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            return f"❌ 处理消息失败: {str(e)}"
        
    async def get_agent_status(self) -> Dict[str, Any]:
        """获取所有智能体状态"""
        status = {
            "system_ready": self._system_ready,
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            try:
                # 假设智能体有状态检查方法
                agent_status = getattr(agent, 'get_status', lambda: {"status": "unknown"})()
                status["agents"][name] = agent_status
            except Exception as e:
                status["agents"][name] = {"status": "error", "error": str(e)}
                
        return status
        
    async def shutdown_system(self):
        """关闭系统"""
        if not self._system_ready:
            logger.info("系统未启动，无需关闭")
            return
            
        try:
            logger.info("正在关闭智能体系统...")
            await self.agent_manager.stop_all_agents()
            self.agents.clear()
            self._system_ready = False
            logger.info("✅ 智能体系统已关闭")
        except Exception as e:
            logger.error(f"关闭系统时出错: {e}")
            raise

    async def switch_agent_mode(self, mode: AgentMode):
        """切换智能体运行模式（当前只支持AI驱动模式）"""
        if mode != AgentMode.AI_DRIVEN:
            print("⚠️ 当前系统只支持AI驱动模式")
            return
            
        if self._system_ready:
            # 需要重新初始化系统
            await self.shutdown_system()
        
        # 更新配置
        self.ai_config.set_agent_mode(mode)
        
        # 重新初始化
        await self.initialize_system()
        print(f"✅ 智能体系统已重新初始化")
    
    def get_active_agents(self) -> Dict[str, str]:
        """获取当前活跃的智能体列表"""
        return {name: agent.__class__.__name__ for name, agent in self.agents.items()}
    
    def get_ai_config_status(self) -> Dict[str, Any]:
        """获取AI配置状态"""
        return self.ai_config.get_config_summary()
    
    async def test_ai_agents(self) -> Dict[str, Any]:
        """测试AI智能体的可用性"""
        test_results = {}
        
        ai_agents = {name: agent for name, agent in self.agents.items() if name.startswith('ai_')}
        
        for name, agent in ai_agents.items():
            try:
                # 尝试初始化AI模型
                if hasattr(agent, '_initialize_models'):
                    await agent._initialize_models()
                    test_results[name] = {"status": "available", "ai_ready": True}
                else:
                    test_results[name] = {"status": "available", "ai_ready": "unknown"}
            except Exception as e:
                test_results[name] = {"status": "error", "error": str(e), "ai_ready": False}
        
        return test_results

def get_agent_integration_system() -> AgentIntegration:
    """获取智能体集成系统实例（单例）"""
    return AgentIntegration()