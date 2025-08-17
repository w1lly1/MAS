import asyncio
from typing import Dict, Optional
from .base_agent import BaseAgent, Message

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
        """注册智能体"""
        self.agents[agent.agent_id] = agent
        
    def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
    async def route_message(self, message: Message):
        """路由消息到目标智能体"""
        if message.receiver in self.agents:
            await self.agents[message.receiver].receive_message(message)
        else:
            print(f"Warning: Target agent {message.receiver} not found")
            
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