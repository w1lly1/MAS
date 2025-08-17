import asyncio
import logging
from typing import Dict, Any, Optional
from .agents import (
    AgentManager, 
    UserCommunicationAgent, 
    StaticCodeScanAgent,
    CodeQualityAgent,
    SecurityAgent,
    PerformanceAgent,
    SummaryAgent
)

logger = logging.getLogger(__name__)

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
            self.__class__._initialized = True
        
    async def initialize_system(self):
        """初始化智能体系统"""
        if self._system_ready:
            logger.info("系统已经初始化，跳过重复初始化")
            return
            
        try:
            # 创建并注册所有智能体
            agent_classes = {
                'user_comm': UserCommunicationAgent,
                'static_scan': StaticCodeScanAgent,
                'code_quality': CodeQualityAgent,
                'security': SecurityAgent,
                'performance': PerformanceAgent,
                'summary': SummaryAgent
            }
            
            for name, agent_class in agent_classes.items():
                try:
                    self.agents[name] = agent_class()
                    logger.info(f"✅ 创建智能体: {name}")
                except Exception as e:
                    logger.error(f"❌ 创建智能体 {name} 失败: {e}")
                    raise
            
            # 注册到管理器
            for agent in self.agents.values():
                self.agent_manager.register_agent(agent)
                
            # 启动所有智能体
            await self.agent_manager.start_all_agents()
            self._system_ready = True
            logger.info("✅ 所有HuggingFace模型智能体已启动")
            
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
            return "❌ AgentIntegration system not ready. Please initialize first."
            
        if 'user_comm' not in self.agents:
            return "❌ User communication agent not available"
            
        try:
            # 构造消息内容
            content = {
                "message": message,
                "session_id": "cli_session",
                "target_directory": target_dir,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # 发送给用户沟通智能体并等待响应
            response = await self.agents['user_comm'].receive_message({
                "sender": "cli_interface",
                "receiver": "user_comm_agent", 
                "content": content,
                "message_type": "user_input"
            })
            
            if response:
                return f"✅ 多AI模型分析系统已处理消息: {message}\n响应: {response}"
            else:
                return f"✅ 多AI模型分析系统已处理消息: {message}"
                
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

def get_agent_integration_system() -> AgentIntegration:
    """获取智能体集成系统实例（单例）"""
    return AgentIntegration()