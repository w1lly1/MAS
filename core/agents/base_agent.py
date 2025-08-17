import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class Message:
    id: str
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: float
    message_type: str = "default"

class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.status = AgentStatus.IDLE
        self.message_queue = asyncio.Queue()
        self.is_running = False
        
    async def start(self):
        """启动智能体"""
        self.is_running = True
        
    async def stop(self):
        """停止智能体"""
        self.is_running = False
        
    async def send_message(self, receiver: str, content: Dict[str, Any], message_type: str = "default"):
        """发送消息给其他智能体"""
        message = Message(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            timestamp=asyncio.get_event_loop().time(),
            message_type=message_type
        )
        # 通过智能体管理器转发消息
        from .agent_manager import AgentManager
        await AgentManager.get_instance().route_message(message)
        
    async def receive_message(self, message: Message):
        """接收消息"""
        await self.message_queue.put(message)
        
    async def process_messages(self):
        """处理消息队列"""
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self.handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing message in {self.name}: {e}")
                
    @abstractmethod
    async def handle_message(self, message: Message):
        """处理具体消息的抽象方法"""
        pass
        
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务的抽象方法"""
        self.status = AgentStatus.PROCESSING
        try:
            result = await self._execute_task_impl(task_data)
            self.status = AgentStatus.COMPLETED
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            raise
            
    @abstractmethod
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """子类需要实现的具体任务执行逻辑"""
        pass