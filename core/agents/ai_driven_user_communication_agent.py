"""
AI驱动的用户沟通代理 - 完全基于AI模型驱动
使用真实的AI模型进行自然语言理解和对话生成
"""
import os
import re
import json
import datetime
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from .base_agent import BaseAgent, Message
from infrastructure.config.ai_agents import get_ai_agent_config
from utils import log, LogLevel

# 导入报告管理器
try:
    from infrastructure.reports import report_manager
except ImportError:
    report_manager = None

try:
    from infrastructure.config.prompts import get_prompt
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
        from infrastructure.config.prompts import get_prompt
    except ImportError:
        # 最后的降级方案,定义一个简单的get_prompt函数和系统提示词
        def get_prompt(task_type, model_name=None, **kwargs):
            if task_type == "conversation":
                user_message = kwargs.get("user_message", "")
                return f"用户: {user_message}\nAI助手:"
            return "AI助手:"

class AIDrivenUserCommunicationAgent(BaseAgent):
    """AI驱动用户沟通智能体 - 完全基于真实AI模型
    
    核心功能:
    1. 真正的AI对话模型 - 使用transformers模型进行自然语言理解
    2. AI驱动的意图识别 - 通过prompt engineering实现智能理解
    3. 上下文感知对话 - 维护会话状态和记忆
    4. 智能任务分派 - AI决策何时启动代码分析
    """
    
    def __init__(self):
        super().__init__("user_comm_agent", "AI User Communication Agent")
        
        # AI模型组件
        self.conversation_model = None
        self.tokenizer = None
        self.used_device = "gpu"
        self.used_device_map = None  # 添加设备映射参数
        
        # 会话管理
        self.session_memory = {}
        self.agent_integration = None
        # 数据库相关：可注入真实 DB Agent；默认使用内部 Mock
        self.db_agent = None
        
        # 从统一配置获取
        self.agent_config = get_ai_agent_config().get_user_communication_agent_config()
        
        # 模型配置 - 仅使用验证通过的模型
        self.model_name = self.agent_config.get("model_name", "Qwen/Qwen1.5-7B-Chat")  # 兼容transformers 4.56.0
        
        # 硬件要求：从配置读取
        self.max_memory_mb = self.agent_config.get("max_memory_mb", 14336)
        
        # 数据库配置
        self._mock_db = True
        self._mock_requirement_id = 1000
        # 调试开关：允许跳过后续代码分析步骤，便于快速验证路由逻辑
        self.mock_code_analysis = os.getenv("MAS_MOCK_CODE_ANALYSIS", "0") == "1"
        self.db_intent_keywords = [
            "记录",
            "保存",
            "写入",
            "知识库",
            "数据库",
            "语义检索",
            "历史",
            "存档",
            "同步",
        ]
        
        # AI模型状态
        self.ai_enabled = False
        
        # 分析结果存储
        self.analysis_results = {}
    
    def set_model(self, model_name: str):
        """动态设置AI模型"""
        if model_name != "Qwen/Qwen1.5-7B-Chat":
            log("user_comm_agent", LogLevel.WARNING, f"⚠️ 仅支持 Qwen/Qwen1.5-7B-Chat 模型")
            return
        
        self.model_name = model_name
        log("user_comm_agent", LogLevel.INFO, f"🔄 已切换到模型: {model_name}")
        
        # 如果AI已经初始化，需要重新初始化
        if self.ai_enabled:
            log("user_comm_agent", LogLevel.INFO, "♻️ 检测到模型已初始化，将重新加载...")
            self.ai_enabled = False
            self.conversation_model = None
            self.tokenizer = None
    
    def get_supported_models(self) -> list:
        """获取支持的模型列表"""
        return ["Qwen/Qwen1.5-7B-Chat"]
    
    async def initialize(self, agent_integration=None):
        """初始化AI模型和代理集成"""
        self.agent_integration = agent_integration
        await self._initialize_ai_models()
        log("user_comm_agent", LogLevel.INFO, "✅ AI用户沟通代理初始化完成")
    
    async def initialize_ai_communication(self):
        """初始化AI用户交流能力 - 向后兼容方法"""
        try:
            log("user_comm_agent", LogLevel.INFO, "初始化智能对话AI...")
            await self._initialize_ai_models()
            log("user_comm_agent", LogLevel.INFO, "智能对话AI初始化成功")
            return True
        except Exception as e:
            log("user_comm_agent", LogLevel.ERROR, f"AI交流初始化错误: {e}")
            return False

    async def _initialize_ai_models(self):
        """初始化Qwen1.5-7B模型"""
        try:
            from transformers import pipeline, AutoTokenizer

            log("user_comm_agent", LogLevel.INFO, "🔧 开始初始化AI对话模型...")
            log("user_comm_agent", LogLevel.INFO, f"📦 正在加载模型: {self.model_name}")

            cache_dir = get_ai_agent_config().get_model_cache_dir()
            # 确保缓存目录是绝对路径
            if not os.path.isabs(cache_dir):
                cache_dir = os.path.abspath(cache_dir)
            log("user_comm_agent", LogLevel.INFO, f"💾 缓存目录: {cache_dir}")

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
                log("user_comm_agent", LogLevel.INFO, "🔍 检测到本地缓存模型文件，将使用本地文件加载")
            else:
                log("user_comm_agent", LogLevel.INFO, "🌐 未检测到本地缓存模型，将从网络下载")

            # 初始化tokenizer
            log("user_comm_agent", LogLevel.INFO, "🔧 使用Qwen配置加载tokenizer...")
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
            log("user_comm_agent", LogLevel.INFO, "✅ Tokenizer加载成功")

            # 配置tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                log("user_comm_agent", LogLevel.INFO, "✅ Tokenizer配置成功")

            # 设置padding_side
            self.tokenizer.padding_side = "left"
            log("user_comm_agent", LogLevel.INFO, "🔧 已设置padding_side")

            # 初始化对话生成pipeline
            log("user_comm_agent", LogLevel.INFO, f"💻 使用设备: {self.used_device}")

            log("user_comm_agent", LogLevel.INFO, "🔧 正在创建对话生成pipeline...")
            # 使用更明确的方式指定模型路径以避免网络请求
            if local_files_only and model_files_exist:
                # 直接使用本地模型路径而不是模型标识符
                snapshot_dirs = os.listdir(snapshots_path)
                if snapshot_dirs:
                    model_local_path = os.path.join(snapshots_path, snapshot_dirs[0])
                    self.conversation_model = pipeline(
                        "text-generation",
                        model=model_local_path,  # 使用本地路径而非模型名
                        tokenizer=self.tokenizer,
                        device=self.used_device,
                        trust_remote_code=True,
                        model_kwargs={
                            "cache_dir": cache_dir,
                            "local_files_only": True,
                            "device_map": "auto" if self.used_device_map == "gpu" else None,
                        }
                    )
                else:
                    raise Exception("未找到有效的模型快照目录")
            else:
                # 在线模式或本地文件不完整时使用模型名称
                self.conversation_model = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=self.used_device,
                    trust_remote_code=True,
                    model_kwargs={
                        "cache_dir": cache_dir,
                        "local_files_only": local_files_only,
                        "device_map": "auto" if self.used_device_map == "gpu" else None,
                    }
                )
            log("user_comm_agent", LogLevel.INFO, "✅ Pipeline创建成功")

            # 预热模型
            log("user_comm_agent", LogLevel.INFO, "🔥 预热AI模型...")
            test_result = self.conversation_model("你好", max_new_tokens=10, do_sample=False)
            if test_result and len(test_result) > 0:
                log("user_comm_agent", LogLevel.INFO, "✅ 模型预热成功")

            self.ai_enabled = True
            log("user_comm_agent", LogLevel.INFO, "🎉 AI对话模型初始化完成")

        except ImportError:
            error_msg = "transformers库未安装,AI功能无法使用"
            log("user_comm_agent", LogLevel.ERROR, f"❌ {error_msg}")
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"AI模型初始化失败: {e}"
            log("user_comm_agent", LogLevel.ERROR, f"❌ {error_msg}")
            raise Exception(error_msg)

    async def handle_message(self, message: Message):
        """处理用户输入消息"""
        try:
            if message.message_type == "user_input":
                await self._process_user_input(message.content)
            elif message.message_type == "system_feedback":
                await self._process_system_feedback(message.content)
            elif message.message_type == "analysis_result":
                await self._process_analysis_result(message.content)
            else:
                log("user_comm_agent", LogLevel.ERROR, f"❌ 系统错误: 收到未知消息类型: {message.message_type}")
        except Exception as e:
            log("user_comm_agent", LogLevel.ERROR, f"❌ 系统错误: 消息处理异常 ({str(e)})")
            raise

    async def _process_user_input(self, content: Dict[str, Any]):
        """处理用户输入 - AI驱动的对话引擎"""
        user_message = content.get("message", "")
        session_id = content.get("session_id", "default")
        target_directory = content.get("target_directory")
        
        log("user_comm_agent", LogLevel.INFO, "📦 处理用户输入...")
        
        # 使用AI驱动的对话处理
        if self.ai_enabled and self.conversation_model:
            try:
                log("user_comm_agent", LogLevel.INFO, "🚀 开始AI对话处理...")
                response, actions = await self.process_ai_conversation(
                    user_message, session_id, target_directory
                )
                
                if response:
                    log("user_comm_agent", LogLevel.INFO, f"✅ AI回应生成成功: {len(response)} 字符")
                    
                    await self._execute_ai_actions(actions, session_id)
                    return
                
            except Exception as e:
                log("user_comm_agent", LogLevel.ERROR, f"❌ AI处理异常: {str(e)}")
                return
        
        # AI模型未启用
        log("user_comm_agent", LogLevel.ERROR, "❌ 系统错误: AI模型未启用或初始化失败")

    async def process_ai_conversation(self, user_message: str, session_id: str, target_directory: str = None):
        """AI驱动的对话处理"""
        try:
            log("user_comm_agent", LogLevel.INFO, "🚀 开始AI对话处理流程...")
            
            # 1. 更新会话上下文
            self._update_session_context(user_message, session_id, target_directory)
            # 2. 准备AI对话上下文
            conversation_history = self._format_conversation_history(session_id)
            
            # 3. 根据模型能力构建 system / user 内容或退回旧 prompt
            use_qwen_chat = (
                self.tokenizer is not None
                and hasattr(self.tokenizer, "apply_chat_template")
                and isinstance(self.model_name, str)
                and self.model_name.startswith("Qwen/")
            )

            if use_qwen_chat:
                # Qwen chat 路径：使用系统级提示词 + 用户内容
                system_prompt = get_prompt(
                        task_type="conversation",
                        model_name=self.model_name,
                        user_message=user_message,
                        conversation_history=conversation_history,
                    )
                user_content = f"用户说: {user_message}\n会话历史: {conversation_history}"
                raw_ai_response = await self._generate_ai_response(
                    system_prompt=system_prompt,
                    user_content=user_content,
                )
            else:
                # 兼容旧模型：继续通过 get_prompt 构造单一字符串 prompt
                try:
                    ai_prompt = get_prompt(
                        task_type="conversation",
                        model_name=self.model_name,
                        user_message=user_message,
                        conversation_history=conversation_history,
                    )
                except (ValueError, KeyError) as e:
                    log("user_comm_agent", LogLevel.WARNING, f"获取Prompt失败,使用简化格式: {e}")
                    ai_prompt = f"用户: {user_message}\n助手:"

                raw_ai_response = await self._generate_ai_response(prompt=ai_prompt)
            
            if not raw_ai_response:
                log("user_comm_agent", LogLevel.ERROR, "❌ AI回应生成失败")
                raise Exception("AI回应生成失败")

            # 打印原始AI响应用于调试
            log("user_comm_agent", LogLevel.INFO, f" Raw AI Response: {raw_ai_response}")

            # 5. 从回应中解析任务规划 JSON（如存在）
            ai_response, task_plan = self._parse_task_plan_from_response(raw_ai_response)
            log("user_comm_agent", LogLevel.INFO, f"✅ AI回应生成成功: {len(ai_response)} 字符 (原始长度: {len(raw_ai_response)})")

            if not task_plan:
                log("user_comm_agent", LogLevel.ERROR, "⚠️ AI回复解析失败")
            # else:
            #     # 打印解析后的任务计划用于调试
            #     log("user_comm_agent", LogLevel.INFO, f" Parsed Task Plan: {task_plan}")

            # 6. 更新会话记忆（仅记录对用户可见的回答部分）
            self._update_session_memory_simple(session_id, ai_response, user_message)
            
            code_tasks = task_plan.get("code_analysis_tasks", []) if task_plan else []
            db_tasks = task_plan.get("db_tasks", []) if task_plan else []
            explanation = task_plan.get("explanation", "") if task_plan else ""

            # 7. 决策后续动作：若没有结构化任务，则回退到简单意图检测
            if code_tasks:
                next_action = "start_analysis"
            elif db_tasks:
                next_action = "handle_db_tasks"
            else:
                next_action = "continue_conversation"

            actions: Dict[str, Any] = {
                "intent": "conversation",
                "next_action": next_action,
                "extracted_info": {
                    "code_analysis_tasks": code_tasks,
                    "db_tasks": db_tasks,
                    "explanation": explanation,
                },
                "code_analysis_tasks": code_tasks,
                "db_tasks": db_tasks,
                "confidence": 1.0,
            }
            
            return ai_response, actions
            
        except Exception as e:
            log("user_comm_agent", LogLevel.ERROR, f"❌ AI对话处理失败: {e}")
            raise
    
    # === 会话管理方法 ===
    
    def _update_session_context(self, message: str, session_id: str, target_directory: str = None):
        """更新会话上下文"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {
                "messages": [],
                "collected_info": {},
                "last_active": self._get_current_time()
            }
        
        session = self.session_memory[session_id]
        session["messages"].append({
            "content": message,
            "timestamp": self._get_current_time(),
            "type": "user"
        })
        session["last_active"] = self._get_current_time()
        
        if target_directory:
            session["target_directory"] = target_directory
    
    def _format_conversation_history(self, session_id: str) -> str:
        """格式化对话历史"""
        session = self.session_memory.get(session_id, {})
        messages = session.get("messages", [])
        
        if not messages:
            return "首次对话"
        
        # 获取最近的3-5条消息
        recent_messages = messages[-5:]
        formatted = []
        
        for msg in recent_messages:
            role = "用户" if msg.get("type") == "user" else "AI"
            content = msg.get("content", "")[:100]  # 限制长度
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def _update_session_memory_simple(self, session_id: str, ai_response: str, user_message: str):
        """简化的会话记忆更新"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {
                "messages": [],
                "last_active": self._get_current_time()
            }
        
        session = self.session_memory[session_id]
        
        # 添加AI回应到对话历史
        session["messages"].append({
            "content": ai_response,
            "timestamp": self._get_current_time(),
            "type": "ai"
        })
        
        session["last_active"] = self._get_current_time()
        
        # 保持对话历史在合理范围内
        if len(session["messages"]) > 20:
            session["messages"] = session["messages"][-15:]

    def _parse_task_plan_from_response(self, ai_response: str) -> Tuple[str, Dict[str, Any]]:
        """
        从 AI 回应中直接解析 code_analysis_tasks 和 db_tasks 关键字的 JSON 任务规划。

        返回:
            user_visible_response: 给用户展示的文本
            task_plan: 解析出的 dict，解析失败时为空 dict
        """
        # 尝试在整个响应中查找 JSON 对象
        import re
        import json

        # 查找可能包含任务规划的 JSON 对象
        json_pattern = r'\{[^{}]*?(?:"[^{}]*?"[^{}]*?)*\}'
        matches = re.findall(json_pattern, ai_response, re.DOTALL)

        best_match = None
        best_plan = {}

        # 遍历所有匹配的 JSON 对象，找到包含 code_analysis_tasks 或 db_tasks 的那个
        for match in matches:
            try:
                plan = json.loads(match)
                if isinstance(plan, dict) and ("code_analysis_tasks" in plan or "db_tasks" in plan):
                    # 找到了包含任务规划的 JSON 对象
                    best_match = match
                    best_plan = plan
                    break
            except json.JSONDecodeError as e:
                # 添加调试日志，显示解析失败的JSON内容
                # log("user_comm_agent", LogLevel.DEBUG, f"JSON解析失败: {e}, 内容: {match[:100]}...")
                continue

        if not best_match:
            # 未找到结构化任务规划，直接返回原始文本
            # log("user_comm_agent", LogLevel.WARNING, f"⚠️ 未解析出结构化任务规划JSON, 响应内容: {ai_response[:200]}...")
            log("user_comm_agent", LogLevel.WARNING, "⚠️ 未解析出结构化任务规划JSON")
            return ai_response.strip(), {}

        explanation = best_plan.get("explanation", "")
        if explanation and explanation.strip():
            user_visible = explanation.strip()
        else:
            user_visible = ai_response.replace(best_match, "").strip()

        # 规范化字段，保证下游总是拿到列表
        code_tasks = best_plan.get("code_analysis_tasks") or []
        db_tasks = best_plan.get("db_tasks") or []
        if not isinstance(code_tasks, list):
            code_tasks = [code_tasks]
        if not isinstance(db_tasks, list):
            db_tasks = [db_tasks]
        best_plan["code_analysis_tasks"] = code_tasks
        best_plan["db_tasks"] = db_tasks
        best_plan["explanation"] = explanation

        if not user_visible:
            user_visible = ai_response.strip()

        return user_visible, best_plan

    async def _execute_ai_actions(self, actions: Dict[str, Any], session_id: str):
        """执行AI建议的操作"""
        next_action = actions.get("next_action")
        code_tasks = actions.get("code_analysis_tasks") or []
        db_tasks = actions.get("db_tasks") or []
        # log("user_comm_agent", LogLevel.INFO,
        #     f"执行AI动作 next_action={next_action} code_tasks={len(code_tasks)} db_tasks={len(db_tasks)} mock_code_analysis={self.mock_code_analysis} session_id={session_id}")

        log("user_comm_agent", LogLevel.INFO,  f"📦 next_action={next_action}")

        if next_action == "start_analysis":
            extracted_info = actions.get("extracted_info", {})
            if self.mock_code_analysis:
                log("user_comm_agent", LogLevel.INFO, "🧪 [MockAnalysis] 拦截代码分析任务，以下信息仅日志展示：")
                try:
                    pretty = json.dumps(code_tasks, ensure_ascii=False, indent=2)
                except TypeError:
                    pretty = str(code_tasks)
                log("user_comm_agent", LogLevel.INFO, pretty if pretty else "(无 code_tasks)")
            else:
                await self._start_code_analysis(extracted_info, session_id)
        elif next_action == "handle_db_tasks":
            await self._dispatch_db_tasks(db_tasks, session_id)
            pass
        elif next_action == "continue_conversation":
            # 继续信息收集 - 提供明确的用户指导
            task_plan = actions.get("extracted_info", {})
            explanation = task_plan.get("explanation", "")
            if explanation:
                log("user_comm_agent", LogLevel.INFO, f"⚠️ {explanation}")
            else:
                log("user_comm_agent", LogLevel.INFO, "⚠️ 为了更好地帮助您，我需要更多信息。请提供更多关于您想要执行的任务的详细信息。")
        else:
            # 继续对话
            pass
    
    async def _dispatch_db_tasks(self, db_tasks: List[Dict[str, Any]], session_id: str):
        """
        将解析出的 db_tasks 分发给 DB Agent。
        
        阶段一/二：如果未注入真实 DB Agent，则使用 Mock 实现，仅打印调试信息方便验证路由是否正确。
        阶段三以后：若注入了真实 DB Agent，则调用其 handle_tasks 接口。
        """
        if not db_tasks:
            return
        
        try:
            if self.db_agent and hasattr(self.db_agent, "handle_tasks"):
                handler = getattr(self.db_agent, "handle_tasks")
                result = handler(db_tasks=db_tasks, session_id=session_id)
                if asyncio.iscoroutine(result):
                    await result
            elif self._mock_db:
                # 简单 Mock：打印到控制台，方便在手工对话中观察
                log("user_comm_agent", LogLevel.INFO, "📚 [MockDBAgent] 收到数据库任务:")
                try:
                    pretty = json.dumps(db_tasks, ensure_ascii=False, indent=2)
                except TypeError:
                    pretty = str(db_tasks)
                log("user_comm_agent", LogLevel.INFO, pretty)
        except Exception as e:
            log("user_comm_agent", LogLevel.ERROR, f"❌ 处理数据库任务失败: {e}")
            log("user_comm_agent", LogLevel.INFO, f"⚠️ 数据库相关操作暂时不可用: {e}")
    
    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        return datetime.datetime.now().isoformat()
    
    # === AI核心方法 ===
    
    async def _generate_ai_response(
        self,
        prompt: str = None,
        system_prompt: Optional[str] = None,
        user_content: Optional[str] = None,
    ) -> str:
        """使用Qwen1.5-7B模型生成回应(改进: 使用chat模板和会话结构)
        
        参数:
            prompt: 兼容旧pipeline的单字符串prompt
            system_prompt: chat模型使用的系统角色与协议说明
            user_content: chat模型使用的用户输入与会话内容
        """
        try:
            if not self.ai_enabled or not self.conversation_model:
                raise Exception("AI模型未初始化")
            
            # 如果支持chat模板并且是Qwen模型,优先走chat形式
            if (
                self.tokenizer
                and hasattr(self.tokenizer, "apply_chat_template")
                and isinstance(self.model_name, str)
                and self.model_name.startswith("Qwen/")
                and system_prompt
                and user_content
            ):
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if self.used_device == "gpu":
                    input_ids = input_ids.to("cuda")
                outputs = self.conversation_model.model.generate(
                    input_ids,
                    max_new_tokens=300,
                    temperature=0.85,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated_text = self.tokenizer.decode(
                    outputs[0][input_ids.shape[1] :], skip_special_tokens=True
                )
                ai_response = generated_text.strip()
                # 简单去除重复开场
                repeats = [
                    "我是MAS代码分析助手",
                    "我可以帮您",
                    "您好！我是MAS代码分析助手",
                ]
                for r in repeats:
                    if ai_response.startswith(r):
                        ai_response = ai_response[len(r) :].lstrip(": ：,，")
                if len(ai_response) < 5:
                    # 退回旧pipeline方式
                    fallback_prompt = (
                        prompt
                        if prompt
                        else f"{system_prompt}\n\n{user_content}".strip()
                    )
                    result = self.conversation_model(
                        fallback_prompt,
                        max_new_tokens=200,
                        temperature=0.9,
                        do_sample=True,
                    )
                    ai_response = self._clean_ai_response(
                        result[0]["generated_text"], fallback_prompt
                    )
                return ai_response
            
            # 回退: 使用原pipeline
            if not prompt:
                # 如果未提供prompt,但有system+user,则拼接成文本prompt
                if system_prompt or user_content:
                    prompt_parts = []
                    if system_prompt:
                        prompt_parts.append(system_prompt)
                    if user_content:
                        prompt_parts.append(user_content)
                    prompt = "\n\n".join(prompt_parts)
                else:
                    raise Exception("缺少有效的生成prompt参数")

            result = self.conversation_model(
                prompt,
                max_new_tokens=150,
                temperature=0.85,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            if result and len(result) > 0:
                raw_text = result[0]["generated_text"]
                ai_response = self._clean_ai_response(raw_text, prompt)
                if not ai_response or len(ai_response.strip()) < 5:
                    ai_response = raw_text[-300:].strip()
                return ai_response
            raise Exception("模型返回空结果")
        except Exception as e:
            log("user_comm_agent", LogLevel.ERROR, f"❌ AI生成失败: {e}")
            raise
    
    def _clean_ai_response(self, raw_text: str, prompt: str) -> str:
        """清理AI生成的回应"""
        # 移除prompt部分，只保留新生成的内容
        if prompt in raw_text:
            ai_response = raw_text.replace(prompt, "").strip()
        else:
            ai_response = raw_text.strip()
        
        # 只清理明显的前缀，保留实际内容
        cleanup_patterns = [
            r'^助手:\s*',
            r'^AI助手:\s*',
            r'^回答:\s*',
        ]
        
        for pattern in cleanup_patterns:
            ai_response = re.sub(pattern, '', ai_response, flags=re.IGNORECASE)
        
        ai_response = ai_response.strip()
        
        # 如果结果为空或太短，返回原始文本（去掉prompt）
        if len(ai_response) < 5:
            # 尝试从原始文本中提取有用内容
            lines = raw_text.strip().split('\n')
            for line in lines:
                if line.strip() and not line.strip().startswith('用户:') and len(line.strip()) > 5:
                    return line.strip()
            # 如果找不到合适的内容，返回一个默认回应
            return "我明白了，有什么可以帮助您的吗？"
        
        return ai_response
    
    # === 其他必要方法的简化实现 ===
    
    async def _process_system_feedback(self, content: Dict[str, Any]):
        """处理系统反馈"""
        feedback_type = content.get("type", "unknown")
        feedback_message = content.get("message", "")
        log("user_comm_agent", LogLevel.INFO, f"📊 系统反馈: {feedback_message}")
    
    async def _process_analysis_result(self, content: Dict[str, Any]):
        """处理分析结果"""
        agent_type = content.get("agent_type")
        requirement_id = content.get("requirement_id")
        log("user_comm_agent", LogLevel.INFO, f"📊 收到 {agent_type} 分析结果 (任务ID: {requirement_id})")
    
    async def _start_code_analysis(self, extracted_info: Dict[str, Any], session_id: str):
        """启动代码分析"""
        # 从会话中获取目录路径
        session = self.session_memory.get(session_id, {})
        target_directory = session.get("target_directory")
        
        # 尝试从用户消息中提取路径
        if not target_directory:
            messages = session.get("messages", [])
            for msg in reversed(messages):
                if msg.get("type") == "user":
                    content = msg.get("content", "")
                    # 查找路径模式
                    import re
                    path_patterns = [
                        r'/[a-zA-Z0-9/_.-]+',  # Unix路径
                        r'[A-Z]:\\[a-zA-Z0-9\\._-]+',  # Windows路径
                    ]
                    for pattern in path_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            target_directory = matches[0]
                            break
                    if target_directory:
                        break
        
        if target_directory:
            log("user_comm_agent", LogLevel.INFO, f"🚀 启动代码分析，目标目录: {target_directory}")
            
            # 检查目录是否存在
            import os
            if os.path.exists(target_directory):
                try:
                    # 启动MAS分析流程
                    if self.agent_integration:
                        log("user_comm_agent", LogLevel.INFO, "📊 调用多智能体分析系统...")
                        await self._trigger_mas_analysis(target_directory, session_id)
                    else:
                        log("user_comm_agent", LogLevel.INFO, "📊 开始分析代码目录结构...")
                        await self._analyze_directory_structure(target_directory, session_id)
                except Exception as e:
                    log("user_comm_agent", LogLevel.ERROR, f"❌ 代码分析启动失败: {e}")
            else:
                log("user_comm_agent", LogLevel.ERROR, f"❌ 目录不存在: {target_directory}")
        else:
            log("user_comm_agent", LogLevel.ERROR, "❌ 无法找到有效的代码目录路径")
    
    async def _trigger_mas_analysis(self, target_directory: str, session_id: str):
        """
        触发MAS多智能体分析(增强: 调用集成器analyze_directory 并等待结果生成)
        增加等待超时机制: 默认1小时, 每10秒刷新一次进度, 直到生成 run_summary 或超时。
        """
        try:
            if self.agent_integration and hasattr(self.agent_integration, 'analyze_directory'):
                result = await self.agent_integration.analyze_directory(target_directory)
                status = result.get('status')
                if status == 'dispatched':
                    path = result.get('report_path')
                    run_id = result.get('run_id')
                    total_files = result.get('total_files')
                    log("user_comm_agent", LogLevel.INFO, f"✅ 分析任务已派发，共 {total_files} 个文件，dispatch报告: {path}")
                    # 启动等待流程
                    await self._wait_for_run_completion(run_id, total_files)
                elif status == 'empty':
                    log("user_comm_agent", LogLevel.WARNING, "⚠️ 目录中未找到可分析的Python文件，分析未执行")
                else:
                    log("user_comm_agent", LogLevel.ERROR, f"❌ 分析失败: {result.get('message','未知错误')}")
            else:
                log("user_comm_agent", LogLevel.ERROR, "❌ 集成器不可用，无法执行多智能体分析")
        except Exception as e:
            log("user_comm_agent", LogLevel.ERROR, f"❌ MAS分析启动异常: {e}")

    async def _wait_for_run_completion(self, run_id: str, total_files: int, timeout: int = None, poll_interval: int = None):
        """等待MAS运行完成，避免频繁打印进度。"""
        if timeout is None:
            timeout = self.agent_config.get("analysis_wait_timeout", 1200)
        if poll_interval is None:
            poll_interval = self.agent_config.get("analysis_poll_interval", 60)
        
        analysis_dir = Path(__file__).parent.parent.parent / 'reports' / 'analysis'
        run_dir = analysis_dir / run_id
        consolidated_dir = run_dir / 'consolidated'
        start_time = asyncio.get_event_loop().time()
        
        log("user_comm_agent", LogLevel.INFO,
            f"WaitLoop start run_id={run_id} timeout={timeout}s poll_interval={poll_interval}s total_files={total_files}"
        )
        
        while True:
            elapsed = int(asyncio.get_event_loop().time() - start_time)
            if elapsed >= timeout:
                log("user_comm_agent", LogLevel.WARNING, f"⏱️ 分析仍在运行，可稍后使用 'mas results {run_id}' 查看结果。")
                return
            
            if not run_dir.exists():
                await asyncio.sleep(poll_interval)
                continue
            
            consolidated_files = []
            if consolidated_dir.exists():
                consolidated_files = list(consolidated_dir.glob("consolidated_*.json"))
            
            severity_agg = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
            total_issues = 0
            for report_path in consolidated_files:
                try:
                    data = json.loads(report_path.read_text(encoding='utf-8'))
                except Exception as exc:
                    log("user_comm_agent", LogLevel.WARNING, f"读取报告失败 {report_path.name}: {exc}")
                    continue
                
                for level, count in data.get('severity_stats', {}).items():
                    if level in severity_agg:
                        severity_agg[level] += count
                total_issues += data.get('issue_count', 0)
            
            summary_candidates = list(run_dir.glob("run_summary*.json"))
            summary_file = summary_candidates[0] if summary_candidates else None
            
            if summary_file:
                try:
                    summary_data = json.loads(summary_file.read_text(encoding='utf-8'))
                except Exception:
                    summary_data = {}
                
                log("user_comm_agent", LogLevel.INFO, "✅ MAS 分析完成。")
                log("user_comm_agent", LogLevel.INFO, f"运行级汇总报告: {summary_file}")
                stats = summary_data.get('severity_stats') or severity_agg
                if stats:
                    log("user_comm_agent", LogLevel.INFO, f"总体问题统计: {stats}")
                return
            
            if total_files and len(consolidated_files) >= total_files:
                log("user_comm_agent", LogLevel.INFO, "✅ MAS 分析完成。")
                return
            
            await asyncio.sleep(poll_interval)
    
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行用户沟通任务"""
        return {"status": "user_communication_ready", "timestamp": self._get_current_time()}
    
    def generate_conversation_report(self, session_data: Dict[str, Any]) -> Optional[str]:
        """生成对话会话报告"""
        if not report_manager:
            return None
        
        try:
            report_data = {
                "session_id": session_data.get("session_id", "unknown"),
                "start_time": session_data.get("start_time"),
                "end_time": datetime.datetime.now().isoformat(),
                "total_messages": len(session_data.get("messages", [])),
                "user_requests": session_data.get("user_requests", []),
                "ai_responses": session_data.get("ai_responses", []),
                "analysis_triggered": session_data.get("analysis_triggered", False),
                "code_paths_analyzed": session_data.get("code_paths", [])
            }
            
            report_path = report_manager.generate_analysis_report(
                report_data, 
                f"conversation_session_{session_data.get('session_id', 'unknown')}.json"
            )
            return str(report_path)
            
        except Exception as e:
            log("user_comm_agent", LogLevel.ERROR, f"生成对话报告时出现错误: {e}")
            return None