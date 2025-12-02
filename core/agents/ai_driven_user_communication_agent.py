"""
AI驱动的用户沟通代理 - 完全基于AI模型驱动
使用真实的AI模型进行自然语言理解和对话生成
"""
import os
import re
import json
import logging
import datetime
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from .base_agent import BaseAgent, Message

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
        # 最后的降级方案,定义一个简单的get_prompt函数
        def get_prompt(task_type, model_name=None, **kwargs):
            if task_type == "conversation":
                user_message = kwargs.get("user_message", "")
                return f"用户: {user_message}\nAI助手:"
            return "AI助手:"

# 设置用户沟通智能体的日志为警告级别,减少非必要输出
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class AIDrivenUserCommunicationAgent(BaseAgent):
    """AI驱动用户沟通智能体 - 完全基于真实AI模型
    
    核心功能:
    1. 真正的AI对话模型 - 使用transformers模型进行自然语言理解
    2. AI驱动的意图识别 - 通过prompt engineering实现智能理解
    3. 上下文感知对话 - 维护会话状态和记忆
    4. 智能任务分派 - AI决策何时启动代码分析
    """
    
    def __init__(self):
        super().__init__("user_comm_agent", "AI驱动用户沟通智能体")
        
        # AI模型组件
        self.conversation_model = None
        self.tokenizer = None
        
        # 会话管理
        self.session_memory = {}
        self.agent_integration = None
        # 数据库相关：可注入真实 DB Agent；默认使用内部 Mock
        self.db_agent = None
        
        # 从统一配置获取
        from infrastructure.config.ai_agents import get_ai_agent_config
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
            "tap",
            "自动补全",
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
            print(f"⚠️ 仅支持 Qwen/Qwen1.5-7B-Chat 模型")
            return
        
        self.model_name = model_name
        print(f"🔄 已切换到模型: {model_name}")
        
        # 如果AI已经初始化，需要重新初始化
        if self.ai_enabled:
            print("♻️ 检测到模型已初始化，将重新加载...")
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
        logger.info("✅ AI用户沟通代理初始化完成")
    
    async def initialize_ai_communication(self):
        """初始化AI用户交流能力 - 向后兼容方法"""
        try:
            logger.info("初始化智能对话AI...")
            await self._initialize_ai_models()
            logger.info("智能对话AI初始化成功")
            return True
        except Exception as e:
            logger.error(f"AI交流初始化错误: {e}")
            return False

    async def _initialize_ai_models(self):
        """初始化Qwen1.5-7B模型"""
        try:
            from transformers import pipeline, AutoTokenizer
            
            print("🔧 开始初始化AI对话模型...")
            print(f"📦 正在加载模型: {self.model_name}")
            
            # 初始化tokenizer
            print("🔧 使用Qwen配置加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            print("✅ Tokenizer加载成功")
            
            # 配置tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("🔧 已设置pad_token")
            
            # 设置padding_side
            self.tokenizer.padding_side = "left"
            print("🔧 已设置padding_side")
            
            # 初始化对话生成pipeline
            device = "cuda" if self._has_gpu() else "cpu"
            print(f"�️ 使用设备: {device}")
            
            print("� 正在创建对话生成pipeline...")
            self.conversation_model = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device_map="auto" if self._has_gpu() else None,
                trust_remote_code=True
            )
            print("✅ Pipeline创建成功")
            
            # 预热模型
            print("🔥 预热AI模型...")
            test_result = self.conversation_model("你好", max_new_tokens=10, do_sample=False)
            if test_result and len(test_result) > 0:
                print("✅ 模型预热成功")
            
            self.ai_enabled = True
            print("🎉 AI对话模型初始化完成")
            
        except ImportError:
            error_msg = "transformers库未安装,AI功能无法使用"
            print(f"❌ {error_msg}")
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"AI模型初始化失败: {e}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
    
    def _has_gpu(self) -> bool:
        """检测是否有GPU可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

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
                logger.warning(f"未知消息类型: {message.message_type}")
                print(f"❌ 系统错误: 收到未知消息类型: {message.message_type}")
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            print(f"❌ 系统错误: 消息处理异常 ({str(e)})")
            raise

    async def _process_user_input(self, content: Dict[str, Any]):
        """处理用户输入 - AI驱动的对话引擎"""
        user_message = content.get("message", "")
        session_id = content.get("session_id", "default")
        target_directory = content.get("target_directory")
        
        logger.info(f"处理用户输入: {user_message[:50]}...")
        
        # 使用AI驱动的对话处理
        if self.ai_enabled and self.conversation_model:
            try:
                logger.info("开始AI对话处理...")
                response, actions = await self.process_ai_conversation(
                    user_message, session_id, target_directory
                )
                
                if response:
                    logger.info(f"AI回应生成成功: {len(response)} 字符")
                    print(response)
                    
                    await self._execute_ai_actions(actions, session_id)
                    return
                
            except Exception as e:
                logger.error(f"AI对话处理失败: {e}")
                print(f"❌ AI处理异常: {str(e)}")
                return
        
        # AI模型未启用
        logger.error("AI模型未启用,无法处理用户输入")
        print("❌ 系统错误: AI模型未启用或初始化失败")

    async def process_ai_conversation(self, user_message: str, session_id: str, target_directory: str = None):
        """AI驱动的对话处理"""
        try:
            logger.info("开始AI对话处理流程...")
            
            # 1. 更新会话上下文
            self._update_session_context(user_message, session_id, target_directory)
            
            # 2. 准备AI对话上下文
            conversation_history = self._format_conversation_history(session_id)
            
            # 3. 构建AI prompt
            try:
                ai_prompt = get_prompt(
                    task_type="conversation",
                    model_name=self.model_name,
                    user_message=user_message,
                    conversation_history=conversation_history
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"获取Prompt失败,使用简化格式: {e}")
                ai_prompt = f"用户: {user_message}\n助手:"
            
            # 4. 使用AI模型生成回应
            raw_ai_response = await self._generate_ai_response(ai_prompt)
            
            if not raw_ai_response:
                logger.error("AI回应生成失败")
                raise Exception("AI回应生成失败")
            
            # 5. 从回应中解析任务规划 JSON（如存在）
            ai_response, task_plan = self._parse_task_plan_from_response(raw_ai_response)
            logger.info(f"AI回应生成成功: {len(ai_response)} 字符（原始长度: {len(raw_ai_response)}）")
            if not task_plan:
                logger.warning(
                    "未解析出 TASK_PLAN_JSON；后续动作将退回启发式。session_id=%s message=%s",
                    session_id,
                    user_message[:80],
                )
            
            # 6. 更新会话记忆（仅记录对用户可见的回答部分）
            self._update_session_memory_simple(session_id, ai_response, user_message)
            
            code_tasks = task_plan.get("code_analysis_tasks", []) if task_plan else []
            db_tasks = task_plan.get("db_tasks", []) if task_plan else []
            
            if not db_tasks:
                inferred_db_tasks = self._infer_db_tasks_from_message(user_message)
                if inferred_db_tasks:
                    logger.info("基于关键词推断出 %d 个 db_tasks (fallback)", len(inferred_db_tasks))
                    db_tasks = inferred_db_tasks
            
            # 7. 决策后续动作：若没有结构化任务，则回退到简单意图检测
            if task_plan:
                next_action = self._decide_next_action_from_plan(user_message, code_tasks, db_tasks)
            else:
                next_action = self._detect_simple_intent(user_message, ai_response)
            
            actions: Dict[str, Any] = {
                "intent": "conversation",
                "next_action": next_action,
                "extracted_info": {
                    "code_analysis_tasks": code_tasks,
                    "db_tasks": db_tasks,
                },
                "code_analysis_tasks": code_tasks,
                "db_tasks": db_tasks,
                "confidence": 1.0,
            }
            
            return ai_response, actions
            
        except Exception as e:
            logger.error(f"AI对话处理失败: {e}")
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
    
    def _detect_simple_intent(self, user_message: str, ai_response: str) -> str:
        """基于关键词的简单意图检测"""
        user_lower = user_message.lower()
        has_db_intent = any(keyword in user_message for keyword in getattr(self, "db_intent_keywords", []))
        
        # 检测代码分析相关关键词
        analysis_keywords = ["分析", "检查", "审查", "扫描", "analysis", "scan", "check", "review"]
        path_keywords = ["路径", "目录", "文件夹", "代码", "项目", "path", "directory", "folder", "code", "/home/", "C:\\"]
        
        # 检查是否包含路径模式
        import re
        has_path = bool(re.search(r'/[a-zA-Z0-9/_.-]+|[A-Z]:\\[a-zA-Z0-9\\._-]+', user_message))
        has_analysis_intent = any(keyword in user_lower for keyword in analysis_keywords)

        if has_db_intent and not has_analysis_intent:
            return "handle_db_tasks"
        
        # 如果包含路径，直接启动分析
        if has_path and any(keyword in user_lower for keyword in analysis_keywords + ["帮我", "help", "请"]) and not has_db_intent:
            return "start_analysis"
        
        # 如果同时包含分析关键词和路径关键词
        if has_analysis_intent:
            if any(keyword in user_lower for keyword in path_keywords):
                return "start_analysis"
            else:
                return "collect_info"
        
        return "continue_conversation"
    
    def _parse_task_plan_from_response(self, ai_response: str) -> Tuple[str, Dict[str, Any]]:
        """
        从 AI 回应中解析 TASK_PLAN_JSON_START/TASK_PLAN_JSON_END 之间的 JSON 任务规划。
        
        返回:
            user_visible_response: 去掉 JSON 片段后给用户展示的文本
            task_plan: 解析出的 dict，解析失败时为空 dict
        """
        start_tag = "TASK_PLAN_JSON_START"
        end_tag = "TASK_PLAN_JSON_END"
        start_idx = ai_response.find(start_tag)
        end_idx = ai_response.find(end_tag)
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            # 未找到结构化任务规划，直接返回原始文本
            return ai_response.strip(), {}
        
        # 用户可见部分：起始标签之前的内容
        user_visible = ai_response[:start_idx].strip()
        
        # JSON 部分：起始标签之后到结束标签之前
        json_str = ai_response[start_idx + len(start_tag):end_idx].strip()
        
        # 去掉可能的代码块标记等噪声
        # 支持 ```json ... ``` 或 ``` 包裹的情况
        if json_str.startswith("```"):
            # 去掉首尾 ```* 包裹
            lines = [line for line in json_str.splitlines() if not line.strip().startswith("```")]
            json_str = "\n".join(lines).strip()
        
        try:
            plan = json.loads(json_str) if json_str else {}
            if not isinstance(plan, dict):
                raise ValueError("parsed JSON is not an object")
        except Exception as e:
            logger.warning(f"解析任务规划 JSON 失败: {e}")
            return ai_response.strip(), {}
        
        # 规范化字段，保证下游总是拿到列表
        code_tasks = plan.get("code_analysis_tasks") or []
        db_tasks = plan.get("db_tasks") or []
        if not isinstance(code_tasks, list):
            code_tasks = [code_tasks]
        if not isinstance(db_tasks, list):
            db_tasks = [db_tasks]
        plan["code_analysis_tasks"] = code_tasks
        plan["db_tasks"] = db_tasks
        
        return user_visible or ai_response.strip(), plan
    
    def _decide_next_action_from_plan(
        self,
        user_message: str,
        code_tasks: List[Dict[str, Any]],
        db_tasks: List[Dict[str, Any]],
    ) -> str:
        """
        基于结构化任务规划粗略决定下一步动作。
        
        当前实现保持保守策略：
        - 如存在明显的代码分析任务，则尝试触发代码分析；
        - 否则若仅有 DB 任务，则继续对话并交给 DB Agent 处理；
        - 否则回退到简单对话。
        """
        if code_tasks:
            return "start_analysis"
        if db_tasks:
            return "handle_db_tasks"
        return self._detect_simple_intent(user_message, user_message)
    
    def _infer_db_tasks_from_message(self, message: str) -> List[Dict[str, Any]]:
        """当模型未输出 TASK_PLAN 时，基于关键词构造最小 db_tasks。"""
        keywords = getattr(self, "db_intent_keywords", [])
        if not any(keyword in message for keyword in keywords):
            return []
        
        task = {
            "operation_type": "record_issue_feedback",
            "entity_type": "KnowledgeBase",
            "payload": {
                "summary": message[:200],
                "semantic_query_text": message[:500],
                "requires_vector_index": True,
            }
        }
        return [task]
    
    async def _execute_ai_actions(self, actions: Dict[str, Any], session_id: str):
        """执行AI建议的操作"""
        next_action = actions.get("next_action")
        code_tasks = actions.get("code_analysis_tasks") or []
        db_tasks = actions.get("db_tasks") or []
        logger.info(
            "执行AI动作 next_action=%s code_tasks=%d db_tasks=%d mock_code_analysis=%s session_id=%s",
            next_action,
            len(code_tasks),
            len(db_tasks),
            self.mock_code_analysis,
            session_id,
        )
        
        # 先尝试处理数据库相关任务（若有）
        if db_tasks:
            await self._dispatch_db_tasks(db_tasks, session_id)
        
        if next_action == "start_analysis":
            extracted_info = actions.get("extracted_info", {})
            if self.mock_code_analysis:
                print("🧪 [MockAnalysis] 拦截代码分析任务，以下信息仅日志展示：")
                try:
                    pretty = json.dumps(code_tasks, ensure_ascii=False, indent=2)
                except TypeError:
                    pretty = str(code_tasks)
                print(pretty if pretty else "(无 code_tasks)")
            else:
                await self._start_code_analysis(extracted_info, session_id)
        elif next_action == "collect_info":
            # 继续信息收集
            pass
        elif next_action == "handle_db_tasks":
            # 已在上方统一处理 DB 任务，这里无需额外动作
            pass
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
                print("📚 [MockDBAgent] 收到数据库任务:")
                try:
                    pretty = json.dumps(db_tasks, ensure_ascii=False, indent=2)
                except TypeError:
                    pretty = str(db_tasks)
                print(pretty)
        except Exception as e:
            logger.error(f"处理数据库任务失败: {e}")
            print(f"⚠️ 数据库相关操作暂时不可用: {e}")
    
    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        return datetime.datetime.now().isoformat()
    
    # === AI核心方法 ===
    
    async def _generate_ai_response(self, prompt: str) -> str:
        """使用Qwen1.5-7B模型生成回应(改进: 使用chat模板和会话结构)"""
        try:
            if not self.ai_enabled or not self.conversation_model:
                raise Exception("AI模型未初始化")
            
            # 如果支持chat模板并且是Qwen模型,构造messages
            if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template') and self.model_name.startswith("Qwen/"):
                # 尝试从prompt中分离用户最新消息(简化处理)
                user_msg = prompt.split('用户:')[-1].split('\n')[0].strip() if '用户:' in prompt else prompt[-80:]
                messages = [
                    {"role": "system", "content": "你是MAS多智能体系统的专业AI代码分析助手,回答要简洁自然。"},
                    {"role": "user", "content": user_msg}
                ]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                import torch
                if self._has_gpu():
                    input_ids = input_ids.to('cuda')
                outputs = self.conversation_model.model.generate(
                    input_ids,
                    max_new_tokens=120,
                    temperature=0.85,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                ai_response = generated_text.strip()
                # 简单去除重复开场
                repeats = ["我是MAS代码分析助手", "我可以帮您", "您好！我是MAS代码分析助手"]
                for r in repeats:
                    if ai_response.startswith(r):
                        ai_response = ai_response[len(r):].lstrip(': ：,，')
                if len(ai_response) < 5:
                    # 退回旧pipeline方式
                    result = self.conversation_model(prompt, max_new_tokens=80, temperature=0.9, do_sample=True)
                    ai_response = self._clean_ai_response(result[0]["generated_text"], prompt)
                return ai_response
            
            # 回退: 使用原pipeline
            result = self.conversation_model(
                prompt,
                max_new_tokens=60,
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
                    ai_response = raw_text[-120:].strip()
                return ai_response
            raise Exception("模型返回空结果")
        except Exception as e:
            print(f"❌ AI生成失败: {e}")
            logger.error(f"AI模型生成失败: {e}")
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
        print(f"📊 系统反馈: {feedback_message}")
    
    async def _process_analysis_result(self, content: Dict[str, Any]):
        """处理分析结果"""
        agent_type = content.get("agent_type")
        requirement_id = content.get("requirement_id")
        print(f"📊 收到 {agent_type} 分析结果 (任务ID: {requirement_id})")
    
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
            print(f"🚀 启动代码分析，目标目录: {target_directory}")
            
            # 检查目录是否存在
            import os
            if os.path.exists(target_directory):
                try:
                    # 启动MAS分析流程
                    if self.agent_integration:
                        print("📊 调用多智能体分析系统...")
                        await self._trigger_mas_analysis(target_directory, session_id)
                    else:
                        print("📊 开始分析代码目录结构...")
                        await self._analyze_directory_structure(target_directory, session_id)
                except Exception as e:
                    print(f"❌ 代码分析启动失败: {e}")
            else:
                print(f"❌ 目录不存在: {target_directory}")
        else:
            print("❌ 无法找到有效的代码目录路径")
    
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
                    print(f"✅ 分析任务已派发，共 {total_files} 个文件，dispatch报告: {path}")
                    # 启动等待流程
                    await self._wait_for_run_completion(run_id, total_files)
                elif status == 'empty':
                    print("⚠️ 目录中未找到可分析的Python文件，分析未执行")
                else:
                    print(f"❌ 分析失败: {result.get('message','未知错误')}")
            else:
                print("❌ 集成器不可用，无法执行多智能体分析")
        except Exception as e:
            print(f"❌ MAS分析启动异常: {e}")

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
        
        logger.info(
            "WaitLoop start run_id=%s timeout=%ss poll_interval=%ss total_files=%s",
            run_id,
            timeout,
            poll_interval,
            total_files,
        )
        
        while True:
            elapsed = int(asyncio.get_event_loop().time() - start_time)
            if elapsed >= timeout:
                print(f"⏱️ 分析仍在运行，可稍后使用 'mas results {run_id}' 查看结果。")
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
                    logger.warning("读取报告失败 %s: %s", report_path.name, exc)
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
                
                print("✅ MAS 分析完成。")
                print(f"运行级汇总报告: {summary_file}")
                stats = summary_data.get('severity_stats') or severity_agg
                if stats:
                    print(f"总体问题统计: {stats}")
                print(f"使用命令: mas results {run_id} 查看详情")
                return
            
            if total_files and len(consolidated_files) >= total_files:
                print("✅ MAS 分析完成。")
                print(f"提示: 汇总报告尚未生成，可稍后运行 'mas results {run_id}' 查看。")
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
            logging.error(f"生成对话报告时出现错误: {e}")
            return None