import asyncio
import logging
from typing import Dict, Optional
from .base_agent import BaseAgent, Message

# 设置日志
logger = logging.getLogger(__name__)

class AgentManager:
    _instance = None
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.running_tasks = {}
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def register_agent(self, agent: BaseAgent):
        """注册智能体 - 静默注册,不输出到控制台"""
        self.agents[agent.agent_id] = agent
        logger.debug(f"注册智能体: {agent.agent_id} ({agent.name})")
        
    def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            print(f"❌ 注销智能体: {agent_id}")
            
    async def route_message(self, message: Message):
        """路由消息到目标智能体"""
        print(f"🔄 路由消息: {message.sender} → {message.receiver} (类型: {message.message_type})")
        if message.receiver in self.agents:
            await self.agents[message.receiver].receive_message(message)
            print(f"✅ 消息已投递给: {message.receiver}")
        else:
            print(f"❌ Warning: Target agent {message.receiver} not found")
            print(f"📋 可用智能体: {list(self.agents.keys())}")
            
    def list_agents(self):
        """列出所有注册的智能体"""
        print("📋 已注册的智能体:")
        for agent_id, agent in self.agents.items():
            print(f"   • {agent_id}: {agent.name}")
        return list(self.agents.keys())
            
    async def start_all_agents(self):
        """启动所有智能体"""
        tasks = []
        for agent in self.agents.values():
            await agent.start()
            tasks.append(asyncio.create_task(agent.process_messages()))
        return tasks
        
    async def stop_all_agents(self):
        """停止所有智能体"""
        for agent in self.agents.values():
            await agent.stop()
            
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """获取指定智能体"""
        return self.agents.get(agent_id)