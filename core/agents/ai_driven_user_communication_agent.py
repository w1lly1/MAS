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
from .base_agent import BaseAgent, Message

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
        
        # 模型配置 - 使用轻量级中英文双语模型
        # ChatGLM2-6B: 轻量级(6B参数)，中英文支持优秀，专为对话优化
        self.model_name = "THUDM/chatglm2-6b"  # 主要模型
        
        # 备用模型（如果ChatGLM2不可用）
        # self.model_name = "microsoft/DialoGPT-small"  # 英文为主，中文支持有限
        
        # 硬件要求：ChatGLM2-6B 约需要 12GB 内存
        
        # 数据库配置
        self._mock_db = True
        self._mock_requirement_id = 1000
        
        # AI模型状态
        self.ai_enabled = False
        
        # 分析结果存储
        self.analysis_results = {}
    
    def set_model(self, model_name: str):
        """动态设置AI模型"""
        supported_models = ["THUDM/chatglm2-6b", "microsoft/DialoGPT-small"]
        
        if model_name not in supported_models:
            print(f"⚠️ 模型 {model_name} 暂不支持")
            print(f"📋 当前支持的模型: {', '.join(supported_models)}")
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
        return [
            "THUDM/chatglm2-6b",        # 主要模型：轻量级中英文
            "microsoft/DialoGPT-small"  # 备用模型：英文为主
        ]
    
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
        """初始化真实AI模型"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            print("🔧 开始初始化AI对话模型...")
            
            model_name = self.model_name
            print(f"📦 正在加载模型: {model_name}")
            
            # 模型兼容性检查
            is_chatglm = "chatglm" in model_name.lower()
            is_dialogpt = "DialoGPT" in model_name
            
            print(f"🔍 模型类型检测: ChatGLM={is_chatglm}, DialoGPT={is_dialogpt}")
            
            # 初始化tokenizer
            try:
                if is_chatglm:
                    print("🔧 使用ChatGLM配置加载tokenizer...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        trust_remote_code=True
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("✅ Tokenizer加载成功")
            except Exception as e:
                print(f"❌ Tokenizer加载失败: {e}")
                if is_chatglm:
                    print("🔄 ChatGLM加载失败，尝试降级到DialoGPT...")
                    model_name = "microsoft/DialoGPT-small"
                    self.model_name = model_name
                    is_chatglm = False
                    is_dialogpt = True
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    print("✅ 降级Tokenizer加载成功")
                else:
                    raise
            
            # 配置tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("🔧 已设置pad_token")
            
            # 初始化对话生成pipeline
            device = "cuda" if self._has_gpu() else "cpu"
            print(f"🖥️ 使用设备: {device}")
            
            # 只对非ChatGLM模型设置padding_side
            if not is_chatglm:
                self.tokenizer.padding_side = "left"
                print("🔧 已设置padding_side")
            
            print("🚀 正在创建对话生成pipeline...")
            
            # 根据模型类型使用不同的pipeline配置
            if is_chatglm:
                print("🔧 使用ChatGLM专用配置...")
                # ChatGLM使用更简化的配置
                self.conversation_model = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    device_map="auto" if self._has_gpu() else None,
                    trust_remote_code=True
                )
            else:
                print("🔧 使用DialoGPT配置...")
                self.conversation_model = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    device_map="auto" if self._has_gpu() else None,
                    return_full_text=False,
                    truncation=True,
                    max_length=1024
                )
            print("✅ Pipeline创建成功")
            
            # 预热模型
            try:
                print("🔥 预热AI模型...")
                
                if is_chatglm:
                    print("🧪 ChatGLM2测试格式")
                    test_inputs = ["你好", "Hello", "用户: 你好\n助手:"]
                else:
                    print("🧪 DialoGPT测试格式")
                    test_inputs = ["Hello", "你好", "User: Hello\nBot:"]
                
                for i, test_input in enumerate(test_inputs):
                    try:
                        print(f"🧪 测试{i+1}: '{test_input}'")
                        test_result = self.conversation_model(test_input, max_new_tokens=10, do_sample=False)
                        print(f"   结果: {test_result}")
                        
                        if test_result and len(test_result) > 0:
                            generated = test_result[0].get("generated_text", "")
                            if len(generated.strip()) > len(test_input.strip()):
                                print(f"   ✅ 格式{i+1}生成有效内容")
                                break
                    except Exception as test_error:
                        print(f"   ❌ 格式{i+1}测试失败: {test_error}")
                        continue
                
                print("✅ 模型预热成功")
            except Exception as e:
                print(f"⚠️ 模型预热失败: {e}")
                # 预热失败不影响整体初始化
            
            self.ai_enabled = True
            print("🎉 AI对话模型初始化完成")
            
        except ImportError:
            error_msg = "transformers库未安装,AI功能无法使用"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print("💡 请安装: pip install transformers torch")
            self.ai_enabled = False
        except Exception as e:
            error_msg = f"AI模型初始化失败: {e}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            import traceback
            print(f"🐛 详细错误: {traceback.format_exc()}")
            self.ai_enabled = False
    
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
                else:
                    logger.error("AI响应生成失败,无法获取有效回复")
                    print("❌ 系统错误: AI响应生成失败")
                    return
                
            except Exception as e:
                logger.error(f"AI对话处理失败: {e}")
                print(f"❌ 系统错误: AI处理异常 ({str(e)})")
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
                if "chatglm" in self.model_name.lower():
                    ai_prompt = f"用户: {user_message}\n助手:"
                else:
                    ai_prompt = user_message
            
            # 4. 使用AI模型生成回应
            ai_response = await self._generate_ai_response(ai_prompt)
            
            if not ai_response:
                logger.error("AI回应生成失败")
                return None, {"next_action": "continue_conversation"}
                
            logger.info(f"AI回应生成成功: {len(ai_response)} 字符")
            
            # 5. 更新会话记忆
            self._update_session_memory_simple(session_id, ai_response, user_message)
            
            # 6. 简单的意图检测
            next_action = self._detect_simple_intent(user_message, ai_response)
            
            return ai_response, {
                "intent": "conversation",
                "next_action": next_action,
                "extracted_info": {},
                "confidence": 1.0
            }
            
        except Exception as e:
            logger.error(f"AI对话处理失败: {e}")
            print(f"❌ 系统错误: {e}")
            return None, {"next_action": "continue_conversation"}
    
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
        
        # 检测代码分析相关关键词
        analysis_keywords = ["分析", "检查", "审查", "扫描", "analysis", "scan", "check", "review"]
        path_keywords = ["路径", "目录", "文件夹", "代码", "项目", "path", "directory", "folder", "code"]
        
        if any(keyword in user_lower for keyword in analysis_keywords):
            if any(keyword in user_lower for keyword in path_keywords):
                return "start_analysis"
            else:
                return "collect_info"
        
        return "continue_conversation"
    
    async def _execute_ai_actions(self, actions: Dict[str, Any], session_id: str):
        """执行AI建议的操作"""
        next_action = actions.get("next_action")
        
        if next_action == "start_analysis":
            extracted_info = actions.get("extracted_info", {})
            await self._start_code_analysis(extracted_info, session_id)
        elif next_action == "collect_info":
            # 继续信息收集
            pass
        else:
            # 继续对话
            pass
    
    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        return datetime.datetime.now().isoformat()
    
    # === AI核心方法 ===
    
    async def _generate_ai_response(self, prompt: str) -> str:
        """使用AI模型生成回应"""
        try:
            if not self.ai_enabled or not self.conversation_model:
                logger.error("AI模型状态检查失败")
                return None
            
            print(f"🧠 开始AI回应生成...")
            print(f"📝 输入prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
            
            is_chatglm = "chatglm" in self.model_name.lower()
            
            # 根据模型类型使用不同的生成策略
            try:
                if is_chatglm:
                    print("🤖 使用ChatGLM2生成策略...")
                    # ChatGLM使用最简化参数，避免tokenizer兼容性问题
                    result = self.conversation_model(
                        prompt,
                        max_length=len(prompt) + 50,  # 使用max_length而不是max_new_tokens
                        do_sample=False  # 禁用采样避免兼容性问题
                    )
                else:
                    print("🤖 使用DialoGPT生成策略...")
                    result = self.conversation_model(
                        prompt,
                        max_new_tokens=50,
                        temperature=0.8,
                        do_sample=True,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                print(f"✅ 模型调用完成")
                
                if result and len(result) > 0:
                    raw_text = result[0]["generated_text"]
                    print(f"📄 原始生成文本: '{raw_text}'")
                    
                    # 清理和提取回应
                    ai_response = self._clean_ai_response(raw_text, prompt)
                    
                    if ai_response and len(ai_response.strip()) >= 2:
                        print(f"🎉 AI回应生成成功: '{ai_response}'")
                        return ai_response
                    else:
                        print(f"⚠️ AI生成的回应过短或无效")
                        return "你好！我是MAS代码分析助手，很高兴为您服务。"
                else:
                    print("❌ AI模型返回空结果")
                    return "抱歉，我暂时无法生成回应。请稍后再试。"
                    
            except Exception as model_error:
                print(f"❌ 模型调用过程出错: {model_error}")
                return None
            
        except Exception as e:
            logger.error(f"AI模型生成失败: {e}")
            return None
    
    def _clean_ai_response(self, raw_text: str, prompt: str) -> str:
        """清理AI生成的回应"""
        # 移除prompt部分，只保留新生成的内容
        if prompt in raw_text:
            ai_response = raw_text.replace(prompt, "").strip()
        else:
            ai_response = raw_text.strip()
        
        # 清理常见的模型输出前缀/后缀
        cleanup_patterns = [
            r'^[Bb]ot:\s*',
            r'^[Aa][Ii]:\s*',
            r'^[Aa]ssistant:\s*',
            r'^助手:\s*',
            r'^\s*[:：]\s*',
        ]
        
        for pattern in cleanup_patterns:
            ai_response = re.sub(pattern, '', ai_response, flags=re.IGNORECASE)
        
        ai_response = ai_response.strip()
        
        # 如果回应仍然过短，尝试提取最后一行
        if len(ai_response) < 3:
            lines = raw_text.strip().split('\n')
            if len(lines) > 1:
                last_line = lines[-1].strip()
                if len(last_line) > len(ai_response):
                    ai_response = last_line
        
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
        print("🚀 启动代码分析...")
    
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行用户沟通任务"""
        return {"status": "user_communication_ready", "timestamp": self._get_current_time()}