import os
import re
import json
import datetime
import torch
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from transformers import AutoModelForCausalLM
from .base_agent import BaseAgent, Message
from infrastructure.config.ai_agents import get_ai_agent_config
from utils import log, LogLevel

class AIDrivenDatabaseManageAgent(BaseAgent):
    """AI驱动数据库管理智能体 - 负责解析用户需求并转换为数据库操作
    
    核心功能:
    1. 用户需求解析 - 使用AI模型理解用户意图
    2. 数据库操作转换 - 将自然语言需求转换为具体数据库操作
    3. 知识检索 - 为后续分析模型提供数据库知识查询
    4. 与用户沟通代理模型保持一致
    """
    
    def __init__(self):
        super().__init__("db_manage_agent", "AI Database Management Agent")
        
        # AI模型组件 - 与用户沟通代理保持一致
        self.conversation_model = None
        self.tokenizer = None
        self.used_device = "gpu"
        self.used_device_map = None
        
        # 从统一配置获取
        self.agent_config = get_ai_agent_config().get_user_communication_agent_config()
        
        # 模型配置
        self.model_name = self.agent_config.get("model_name", "Qwen/Qwen1.5-7B-Chat")
        
        # 硬件要求：从配置读取
        self.max_memory_mb = self.agent_config.get("max_memory_mb", 14336)
        
        # AI模型状态
        self.ai_enabled = False
        
        # 数据库操作记录
        self.database_operations = {}
        
        # 会话管理
        self.session_memory = {}
    
    async def initialize_data_manage(self, agent_integration=None):
        """初始化AI模型和代理集成"""
        try:
            self.agent_integration = agent_integration
            await self._initialize_ai_models()
            return True
        except Exception as e:
            log("db_manage_agent", LogLevel.ERROR, f"AI数据库管理代理初始化错误: {e}")
            return False
    
    async def _initialize_ai_models(self):
        """初始化Qwen1.5-7B模型 - 与用户沟通代理保持一致"""
        try:
            from transformers import pipeline, AutoTokenizer

            log("db_manage_agent", LogLevel.INFO, "🔧 开始初始化数据库管理模型...")
            log("db_manage_agent", LogLevel.INFO, f"📦 正在加载模型: {self.model_name}")

            cache_dir = get_ai_agent_config().get_model_cache_dir()
            # 确保缓存目录是绝对路径
            if not os.path.isabs(cache_dir):
                cache_dir = os.path.abspath(cache_dir)
            log("db_manage_agent", LogLevel.INFO, f"💾 缓存目录: {cache_dir}")

            # 确保缓存目录存在
            os.makedirs(cache_dir, exist_ok=True)

            local_files_only = False
            # 检查模型文件是否已存在
            model_path = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
            # 检查快照目录是否存在且不为空
            snapshots_path = os.path.join(model_path, "snapshots")
            model_files_exist = (
                os.path.exists(model_path) and 
                os.path.exists(snapshots_path) and 
                os.listdir(snapshots_path)
            )

            if model_files_exist:
                local_files_only = True
                log("db_manage_agent", LogLevel.INFO, "🔍 检测到本地缓存模型文件，将使用本地文件加载")
            else:
                log("db_manage_agent", LogLevel.INFO, "🌐 未检测到本地缓存模型，将从网络下载")

            # 初始化tokenizer
            log("db_manage_agent", LogLevel.INFO, "🔧 使用Qwen配置加载tokenizer...")
            if local_files_only and model_files_exist:
                # 使用本地路径加载tokenizer，避免网络请求
                snapshot_dirs = os.listdir(snapshots_path)
                if snapshot_dirs:
                    model_local_path = os.path.join(snapshots_path, snapshot_dirs[0])
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_local_path,
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                else:
                    raise Exception("未找到有效的模型快照目录")
            else:
                # 在线模式或本地文件不完整时使用模型名称
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    local_files_only=local_files_only
                )
            log("db_manage_agent", LogLevel.INFO, "✅ Tokenizer加载成功")

            # 配置tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                log("db_manage_agent", LogLevel.INFO, "✅ Tokenizer配置成功")

            # 设置padding_side
            self.tokenizer.padding_side = "left"
            log("db_manage_agent", LogLevel.INFO, "🔧 已设置padding_side")

            # 初始化对话生成pipeline
            log("db_manage_agent", LogLevel.INFO, f"💻 使用设备: {self.used_device}")

            log("db_manage_agent", LogLevel.INFO, "🔧 加载模型...")
            if local_files_only and model_files_exist:
                snapshot_dirs = os.listdir(snapshots_path)
                if snapshot_dirs:
                    model_local_path = os.path.join(snapshots_path, snapshot_dirs[0])
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_local_path,
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        device_map="auto" if self.used_device == "gpu" else None,
                        torch_dtype=torch.float16 if self.used_device == "gpu" else torch.float32,
                        local_files_only=True
                    )
                else:
                    raise Exception("未找到有效的模型快照目录")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    device_map="auto" if self.used_device == "gpu" else None,
                    torch_dtype=torch.float16 if self.used_device == "gpu" else torch.float32,
                    local_files_only=local_files_only
                )

            log("db_manage_agent", LogLevel.INFO, "🔥 预热数据库管理模型...")
            test_prompt = "你好"
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            self.ai_enabled = True
            log("db_manage_agent", LogLevel.INFO, "🎉 AI数据库管理模型初始化完成")

        except ImportError:
            error_msg = "transformers库未安装,AI功能无法使用"
            log("db_manage_agent", LogLevel.ERROR, f"❌ {error_msg}")
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"AI模型初始化失败: {e}"
            log("db_manage_agent", LogLevel.ERROR, f"❌ {error_msg}")
            raise Exception(error_msg)

    async def handle_message(self, message: Message):
        """处理消息 - 实现BaseAgent的抽象方法"""
        try:
            if message.message_type == "user_requirement":
                await self._process_user_requirement(message.content)
            elif message.message_type == "knowledge_request":
                await self._process_knowledge_request(message.content)
            else:
                log("db_manage_agent", LogLevel.ERROR, f"❌ 系统错误: 收到未知消息类型: {message.message_type}")
        except Exception as e:
            log("db_manage_agent", LogLevel.ERROR, f"❌ 系统错误: 消息处理异常 ({str(e)})")
            raise

    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体任务 - 实现BaseAgent的抽象方法"""
        # 暂时返回空结果
        return {"status": "success", "message": "Task executed successfully"}

    # === 对外接口方法 ===

    async def user_requirement_interpret(self, user_requirement: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        用户需求解析接口 - 接收用户沟通模型的输入
        
        参数:
            user_requirement: JSON格式的用户需求
            session_id: 对话会话ID
            
        返回:
            解析后的数据库操作规划
        """
        log("db_manage_agent", LogLevel.INFO, f"📝 开始解析用户需求，会话ID: {session_id}")
        
        # 暂时返回默认值
        return {
            "status": "success",
            "session_id": session_id,
            "interpreted_operations": [],
            "message": "用户需求解析功能待实现"
        }

    async def get_knowledge_from_database(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        知识检索接口 - 提供给后续待开发的二次分析模型
        
        参数:
            scan_results: JSON格式的代码扫描结果
            
        返回:
            从数据库检索到的相关知识
        """
        log("db_manage_agent", LogLevel.INFO, "🔍 开始检索数据库知识")
        
        # 暂时返回默认值
        return {
            "status": "success",
            "knowledge_data": {},
            "message": "知识检索功能待实现"
        }

    # === 内部处理方法 ===

    async def _process_user_requirement(self, content: Dict[str, Any]):
        """处理用户需求消息"""
        user_requirement = content.get("requirement", {})
        session_id = content.get("session_id", "default")
        
        # 调用用户需求解析接口
        result = await self.user_requirement_interpret(user_requirement, session_id)
        
        # 记录操作结果
        if session_id not in self.database_operations:
            self.database_operations[session_id] = []
        
        self.database_operations[session_id].append({
            "timestamp": self._get_current_time(),
            "operation": "user_requirement_interpret",
            "result": result
        })

    async def _process_knowledge_request(self, content: Dict[str, Any]):
        """处理知识请求消息"""
        scan_results = content.get("scan_results", {})
        
        # 调用知识检索接口
        result = await self.get_knowledge_from_database(scan_results)
        
        # 记录操作结果
        if "knowledge_requests" not in self.database_operations:
            self.database_operations["knowledge_requests"] = []
        
        self.database_operations["knowledge_requests"].append({
            "timestamp": self._get_current_time(),
            "operation": "get_knowledge_from_database",
            "result": result
        })

    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        return datetime.datetime.now().isoformat()