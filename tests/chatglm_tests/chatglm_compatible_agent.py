"""
ChatGLM兼容的用户沟通代理 - 修复版本
基于兼容性测试结果，使用直接模型调用避免pipeline问题
"""

import os
import re
import json
import logging
import datetime
import asyncio
from typing import Dict, Any, Optional, List, Tuple

# 导入基础组件
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from core.agents.base_agent import BaseAgent, Message
except ImportError:
    # 如果导入失败，创建简单的基类
    class BaseAgent:
        def __init__(self, agent_id, name):
            self.agent_id = agent_id
            self.name = name
    
    class Message:
        def __init__(self, message_type, content):
            self.message_type = message_type
            self.content = content

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class ChatGLMCompatibleAgent(BaseAgent):
    """ChatGLM兼容的用户沟通代理"""
    
    def __init__(self):
        super().__init__("chatglm_comm_agent", "ChatGLM兼容用户沟通代理")
        
        # ChatGLM组件
        self.model = None
        self.tokenizer = None
        
        # 会话管理
        self.session_memory = {}
        
        # 模型配置
        self.model_name = "THUDM/chatglm2-6b"
        self.ai_enabled = False
        
        print(f"🤖 初始化ChatGLM兼容代理: {self.model_name}")
    
    async def initialize_chatglm(self):
        """初始化ChatGLM模型 - 使用兼容方法"""
        try:
            print("🔧 开始初始化ChatGLM模型（兼容模式）...")
            
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            model_name = self.model_name
            print(f"📦 加载模型: {model_name}")
            
            # 加载tokenizer - 避免pipeline相关问题
            print("🔧 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            print("✅ Tokenizer加载成功")
            
            # 加载模型 - 使用AutoModel而不是AutoModelForCausalLM
            print("🔧 加载模型...")
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto" if self._has_gpu() else None
            )
            print("✅ 模型加载成功")
            
            # 测试基本功能
            print("🧪 测试基本对话功能...")
            test_response = await self._test_basic_chat()
            
            if test_response:
                print(f"✅ 基本测试成功: {test_response[:50]}...")
                self.ai_enabled = True
                print("🎉 ChatGLM初始化完成")
                return True
            else:
                print("❌ 基本测试失败")
                return False
                
        except Exception as e:
            print(f"❌ ChatGLM初始化失败: {e}")
            import traceback
            print(f"🐛 详细错误: {traceback.format_exc()}")
            self.ai_enabled = False
            return False
    
    async def _test_basic_chat(self) -> Optional[str]:
        """测试基本对话功能"""
        try:
            test_input = "你好"
            print(f"🧪 测试输入: {test_input}")
            
            # 使用ChatGLM的原生chat方法
            response, _ = self.model.chat(
                self.tokenizer, 
                test_input, 
                history=[],
                max_length=2048,
                temperature=0.8
            )
            
            print(f"🎉 测试回应: {response}")
            return response
            
        except Exception as e:
            print(f"❌ 基本测试失败: {e}")
            return None
    
    def _has_gpu(self) -> bool:
        """检测GPU可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    async def generate_response(self, user_input: str, session_id: str = "default") -> Optional[str]:
        """生成AI回应"""
        if not self.ai_enabled or not self.model or not self.tokenizer:
            print("❌ 模型未就绪")
            return None
        
        try:
            print(f"🧠 处理用户输入: {user_input}")
            
            # 获取会话历史
            history = self._get_session_history(session_id)
            
            # 使用ChatGLM原生chat方法
            response, new_history = self.model.chat(
                self.tokenizer,
                user_input,
                history=history,
                max_length=2048,
                temperature=0.8
            )
            
            # 更新会话历史
            self._update_session_history(session_id, new_history)
            
            print(f"✅ AI回应生成: {response}")
            return response
            
        except Exception as e:
            print(f"❌ 回应生成失败: {e}")
            import traceback
            print(f"🐛 详细错误: {traceback.format_exc()}")
            return None
    
    def _get_session_history(self, session_id: str) -> List:
        """获取会话历史"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {
                "history": [],
                "last_active": datetime.datetime.now().isoformat()
            }
        
        return self.session_memory[session_id]["history"]
    
    def _update_session_history(self, session_id: str, new_history: List):
        """更新会话历史"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {
                "history": [],
                "last_active": datetime.datetime.now().isoformat()
            }
        
        # 限制历史长度
        if len(new_history) > 10:  # 保持最近5轮对话
            new_history = new_history[-10:]
        
        self.session_memory[session_id]["history"] = new_history
        self.session_memory[session_id]["last_active"] = datetime.datetime.now().isoformat()
    
    async def handle_user_input(self, user_input: str, session_id: str = "default") -> str:
        """处理用户输入的主接口"""
        try:
            # 生成AI回应
            response = await self.generate_response(user_input, session_id)
            
            if response:
                return response
            else:
                return "抱歉，我暂时无法生成回应。请稍后再试。"
                
        except Exception as e:
            print(f"❌ 处理用户输入失败: {e}")
            return "系统错误，请重试。"

# 测试函数
async def test_chatglm_compatible_agent():
    """测试ChatGLM兼容代理"""
    print("🧪 测试ChatGLM兼容代理...")
    
    agent = ChatGLMCompatibleAgent()
    
    # 初始化
    if await agent.initialize_chatglm():
        print("✅ 代理初始化成功")
        
        # 测试对话
        test_cases = [
            "你好",
            "你是谁？",
            "请介绍代码分析的重要性",
            "How are you?"
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n🧪 测试 {i}: {test_input}")
            response = await agent.handle_user_input(test_input)
            print(f"✅ 回应: {response}")
        
        return agent
    else:
        print("❌ 代理初始化失败")
        return None

if __name__ == "__main__":
    # 运行测试
    result = asyncio.run(test_chatglm_compatible_agent())
    
    if result:
        print("🎉 ChatGLM兼容代理测试成功！")
    else:
        print("❌ ChatGLM兼容代理测试失败！")
