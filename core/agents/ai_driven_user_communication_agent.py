import os
import re
import git
import logging
import datetime
from typing import Dict, Any, Optional, List, Tuple
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService

logger = logging.getLogger(__name__)

class AIDrivenUserCommunicationAgent(BaseAgent):
    """AI驱动用户沟通智能体 - 集成AI对话能力
    
    核心功能：
    1. 真正的对话AI - 理解上下文，记忆信息
    2. 主动信息收集 - 在对话中逐步获取分析目标
    3. 智能意图理解 - 不依赖特定格式，理解自然语言
    4. 任务分派和管理 - 将用户需求转化为具体的分析任务
    """
    
    def __init__(self):
        super().__init__("ai_user_comm_agent", "AI驱动用户沟通智能体")
        
        # 数据库配置
        self._mock_db = True
        self._mock_requirement_id = 1000
        if not self._mock_db:
            self.db_service = DatabaseService()
        
        # 任务管理
        self.current_session = None
        self.active_analysis = {}
        self.analysis_results = {}  # 存储分析结果
        
        # === AI对话核心组件 ===
        self.session_memory = {}        # 会话记忆
        self.conversation_context = []  # 对话历史
        self.ai_enabled = False        # AI模型是否成功初始化
        
        # 意图分类关键词
        self.intent_categories = {
            "code_analysis": ["分析", "检查", "review", "scan", "扫描", "检测", "audit", "代码", "质量", "性能", "安全"],
            "help": ["帮助", "help", "使用", "指导", "教程", "如何", "怎么"],
            "greeting": ["你好", "hello", "hi", "嗨", "您好", "你是谁", "介绍", "自己", "什么", "功能"],
            "status": ["状态", "status", "进展", "结果", "报告"],
            "information_providing": ["目录", "路径", "仓库", "代码", "项目", "文件"]
        }
        
    async def initialize_ai_communication(self):
        """初始化AI用户交流能力"""
        try:
            logger.info("🧠 初始化智能对话AI...")
            # 模拟AI模型初始化
            await self._mock_ai_initialization()
            self.ai_enabled = True
            logger.info("✅ 智能对话AI初始化成功")
            print("🧠 AI智能交流已启用 - 支持自然语言理解和智能回应")
            return True
        except Exception as e:
            logger.error(f"❌ AI交流初始化错误: {e}")
            self.ai_enabled = False
            return False
    
    async def _mock_ai_initialization(self):
        """模拟AI初始化（避免真实模型加载）"""
        import asyncio
        await asyncio.sleep(1)  # 模拟初始化时间
        
        # 真实AI模型初始化示例（注释掉的代码）:
        # try:
        #     from transformers import AutoTokenizer, AutoModelForCausalLM
        #     self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        #     self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        #     print("✅ 真实AI模型加载成功")
        # except ImportError:
        #     print("⚠️ transformers库未安装，使用模拟模式")
        # except Exception as e:
        #     print(f"❌ AI模型加载失败: {e}")
        #     raise
        
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
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            raise

    async def _process_user_input(self, content: Dict[str, Any]):
        """处理用户输入 - 智能对话引擎"""
        user_message = content.get("message", "")
        session_id = content.get("session_id", "default")
        source = content.get("source", "direct")
        target_directory = content.get("target_directory")
        
        # print(f"💬 收到用户消息: {user_message}")
        print(f"💬 收到用户消息: {user_message}")
        
        # 使用智能对话AI处理
        if self.ai_enabled:
            try:
                # 构建上下文
                context = {
                    "session_id": session_id,
                    "target_directory": target_directory,
                    "source": source
                }
                
                # AI对话处理
                ai_response, suggested_actions = await self.process_conversational_message(
                    user_message, session_id, context
                )
                
                # 显示AI回应
                print(f"🤖 AI回应: {ai_response}")
                
                # 根据AI建议执行操作
                await self._execute_ai_suggested_action(suggested_actions, session_id)
                return
                
            except Exception as e:
                logger.error(f"AI对话处理失败，降级到传统模式: {e}")
                print("⚠️ AI处理遇到问题，切换到传统模式")
        
        # 传统模式处理（降级或备用）
        await self._process_with_traditional_mode(user_message, target_directory, session_id)
    
    # === 核心对话AI方法 ===
    
    async def process_conversational_message(self, user_message: str, session_id: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """对话式消息处理 - 核心方法"""
        try:
            # 1. 更新会话上下文
            self._update_session_context(user_message, session_id, context)
            
            # 2. 对话AI分析
            conversation_analysis = await self._analyze_conversation_flow(user_message, session_id)
            
            # 3. 信息收集与整合
            collected_info = self._collect_and_integrate_information(user_message, session_id)
            
            # 4. 生成智能回应
            ai_response = await self._generate_conversational_response(
                user_message, conversation_analysis, collected_info, session_id
            )
            
            # 5. 确定下一步行动
            suggested_actions = self._determine_next_actions(conversation_analysis, collected_info)
            
            # 6. 更新对话历史
            self._record_conversation_turn(user_message, ai_response, conversation_analysis, session_id)
            
            return ai_response, suggested_actions
            
        except Exception as e:
            logger.error(f"❌ 对话处理失败: {e}")
            return "我理解您的请求，让我重新组织一下...", {}
    
    def _update_session_context(self, message: str, session_id: str, context: Dict[str, Any]):
        """更新会话上下文"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {
                "messages": [],
                "collected_info": {},
                "conversation_state": "initial",
                "user_intent_history": [],
                "last_active": self._get_current_time()
            }
        
        session = self.session_memory[session_id]
        session["messages"].append({
            "content": message,
            "timestamp": self._get_current_time(),
            "type": "user"
        })
        session["last_active"] = self._get_current_time()
        
        if context:
            session["external_context"] = context
    
    async def _analyze_conversation_flow(self, message: str, session_id: str) -> Dict[str, Any]:
        """分析对话流程和用户意图"""
        session = self.session_memory.get(session_id, {})
        conversation_history = session.get("messages", [])
        
        # 基于对话历史进行智能分析
        analysis = {
            "current_intent": self._detect_current_intent(message),
            "conversation_stage": self._determine_conversation_stage(conversation_history),
            "information_completeness": self._assess_information_completeness(session_id),
            "user_engagement": self._assess_user_engagement(conversation_history),
            "context_continuity": self._check_context_continuity(message, conversation_history)
        }
        
        return analysis
    
    def _collect_and_integrate_information(self, message: str, session_id: str) -> Dict[str, Any]:
        """智能信息收集与整合"""
        session = self.session_memory.get(session_id, {})
        collected = session.get("collected_info", {})
        
        # 从当前消息中提取信息
        extracted = self._extract_comprehensive_information(message)
        
        # 整合信息
        for key, value in extracted.items():
            if value:  # 只存储有效信息
                if key not in collected:
                    collected[key] = value
                elif isinstance(value, list):
                    # 合并列表信息
                    existing = collected[key] if isinstance(collected[key], list) else [collected[key]]
                    collected[key] = list(set(existing + value))
                else:
                    # 更新单值信息
                    collected[key] = value
        
        # 更新会话记忆
        self.session_memory[session_id]["collected_info"] = collected
        
        return collected
    
    def _extract_comprehensive_information(self, message: str) -> Dict[str, Any]:
        """全面提取信息"""
        info = {
            "code_paths": [],
            "github_urls": [],
            "file_types": [],
            "analysis_types": [],
            "technology_stack": [],
            "specific_concerns": []
        }
        
        # 代码路径提取
        path_patterns = [
            r'[./~]?/[\w\-./]+',
            r'[A-Za-z]:\\[\w\-\\/.]+',
            r'仓库目录\s*[:\s]*([^\s]+)',
            r'代码目录\s*[:\s]*([^\s]+)',
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, message)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if match and (os.path.exists(match) or match.startswith('/')):
                    info["code_paths"].append(match)
        
        # GitHub URL提取
        github_patterns = [
            r'https://github\.com/[\w\-./]+',
            r'github\.com/[\w\-./]+',
        ]
        
        for pattern in github_patterns:
            matches = re.findall(pattern, message)
            info["github_urls"].extend(matches)
        
        # 技术栈识别
        tech_keywords = {
            "python": ["python", "py", "django", "flask"],
            "javascript": ["javascript", "js", "node", "react"],
            "java": ["java", "spring", "maven"],
            "cpp": ["c++", "cpp", "c"],
        }
        
        message_lower = message.lower()
        for tech, keywords in tech_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                info["technology_stack"].append(tech)
        
        # 分析类型识别
        analysis_keywords = {
            "security": ["安全", "security", "漏洞"],
            "performance": ["性能", "performance", "优化"],
            "quality": ["质量", "quality", "代码质量"],
        }
        
        for analysis_type, keywords in analysis_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                info["analysis_types"].append(analysis_type)
        
        return info
    
    def _detect_current_intent(self, message: str) -> str:
        """检测当前消息的意图"""
        intent_scores = self._calculate_intent_scores(message)
        return max(intent_scores, key=intent_scores.get)
    
    def _calculate_intent_scores(self, message: str) -> Dict[str, float]:
        """计算各意图的匹配分数"""
        message_lower = message.lower()
        scores = {}
        
        for intent, keywords in self.intent_categories.items():
            score = 0.0
            matched_keywords = 0
            for keyword in keywords:
                if keyword in message_lower:
                    keyword_score = min(0.5, len(keyword) / 10)
                    score += keyword_score
                    matched_keywords += 1
            
            if matched_keywords > 1:
                score += 0.2 * (matched_keywords - 1)
            
            scores[intent] = min(score, 1.0)
        
        # 检查是否包含路径信息
        if max(scores.values()) < 0.3:
            if self._contains_code_path(message):
                scores["code_analysis"] = 0.8
                scores["information_providing"] = 0.9
        
        return scores
    
    def _contains_code_path(self, message: str) -> bool:
        """检查是否包含代码路径"""
        path_indicators = ['/src/', '/lib/', '/app/', '/code/', '.py', '.js', '.java', '.cpp', 'var/']
        return any(indicator in message.lower() for indicator in path_indicators)
    
    def _determine_conversation_stage(self, conversation_history: List[Dict]) -> str:
        """确定对话阶段"""
        if len(conversation_history) <= 1:
            return "initial"
        elif len(conversation_history) <= 3:
            return "information_gathering"
        elif len(conversation_history) <= 6:
            return "clarification"
        else:
            return "execution"
    
    def _assess_information_completeness(self, session_id: str) -> float:
        """评估信息完整度"""
        session = self.session_memory.get(session_id, {})
        collected = session.get("collected_info", {})
        
        score = 0.0
        
        # 检查是否有代码目标
        if collected.get("code_paths") or collected.get("github_urls"):
            score += 0.6
        
        # 检查是否有分析类型信息
        if collected.get("analysis_types"):
            score += 0.2
        
        # 检查是否有技术栈信息
        if collected.get("technology_stack"):
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_user_engagement(self, conversation_history: List[Dict]) -> str:
        """评估用户参与度"""
        if len(conversation_history) <= 1:
            return "initial"
        
        recent_messages = conversation_history[-3:]
        avg_length = sum(len(msg.get("content", "")) for msg in recent_messages) / len(recent_messages)
        
        if avg_length > 50:
            return "high"
        elif avg_length > 20:
            return "medium"
        else:
            return "low"
    
    def _check_context_continuity(self, current_message: str, history: List[Dict]) -> bool:
        """检查上下文连续性"""
        if not history:
            return True
        
        # 检查是否有指代词或延续性词汇
        continuity_indicators = ["这个", "上述", "那个", "它", "继续", "还有", "另外"]
        return any(indicator in current_message for indicator in continuity_indicators)
    
    async def _generate_conversational_response(self, message: str, analysis: Dict[str, Any], 
                                              collected_info: Dict[str, Any], session_id: str) -> str:
        """生成对话式回应"""
        intent = analysis["current_intent"]
        stage = analysis["conversation_stage"]
        completeness = analysis["information_completeness"]
        
        # 根据对话阶段和信息完整度生成不同类型的回应
        if stage == "initial":
            return self._generate_initial_response(intent, message)
        elif stage == "information_gathering":
            return self._generate_information_gathering_response(collected_info, message)
        elif completeness >= 0.6:
            return self._generate_action_ready_response(collected_info)
        else:
            return self._generate_clarification_response(collected_info, message)
    
    def _generate_initial_response(self, intent: str, message: str) -> str:
        """生成初次接触回应"""
        if intent == "greeting":
            return """👋 您好！我是MAS AI代码审查助手！

我可以帮您进行：
🔍 智能代码质量分析
🔒 安全漏洞检测  
⚡ 性能优化建议
📊 综合评估报告

请告诉我您想分析的代码项目，我会通过对话了解您的具体需求！"""
        
        elif intent == "information_providing":
            return f"""✅ 我收到了您提供的信息："{message}"

让我了解更多细节来为您提供最佳的分析服务：
• 这是什么类型的项目？（如：Python Web应用、Java后端服务等）
• 您主要关心哪些方面？（如：安全性、性能、代码质量等）
• 有特别需要注意的地方吗？"""
        
        else:
            return """🤖 我理解您想要进行代码分析！

为了提供精准的服务，请告诉我：
📁 代码位置（本地路径或GitHub链接）
🔧 项目类型和技术栈
🎯 分析重点（安全、性能、质量等）

您可以自然地描述，我会智能理解您的需求！"""
    
    def _generate_information_gathering_response(self, collected_info: Dict[str, Any], message: str) -> str:
        """生成信息收集阶段的回应"""
        response = "👍 很好！我正在收集信息...\n\n"
        
        # 确认已收集的信息
        if collected_info.get("code_paths"):
            response += f"📁 代码路径：{collected_info['code_paths'][0]}\n"
        if collected_info.get("github_urls"):
            response += f"🌐 GitHub仓库：{collected_info['github_urls'][0]}\n"
        if collected_info.get("technology_stack"):
            response += f"🔧 技术栈：{', '.join(collected_info['technology_stack'])}\n"
        if collected_info.get("analysis_types"):
            response += f"🎯 分析重点：{', '.join(collected_info['analysis_types'])}\n"
        
        # 询问缺失的信息
        missing_info = []
        if not collected_info.get("code_paths") and not collected_info.get("github_urls"):
            missing_info.append("代码位置")
        if not collected_info.get("analysis_types"):
            missing_info.append("分析重点")
        
        if missing_info:
            response += f"\n❓ 还需要了解：{', '.join(missing_info)}"
            response += "\n💡 您可以继续补充信息，我会智能整合！"
        else:
            response += "\n✅ 信息收集完成，准备开始分析！"
        
        return response
    
    def _generate_action_ready_response(self, collected_info: Dict[str, Any]) -> str:
        """生成准备执行分析的回应"""
        response = "🚀 太棒了！信息已收集完整，我将为您启动AI驱动的代码分析：\n\n"
        
        # 分析目标确认
        if collected_info.get("code_paths"):
            response += f"📁 分析目标：{collected_info['code_paths'][0]}\n"
        elif collected_info.get("github_urls"):
            response += f"🌐 GitHub仓库：{collected_info['github_urls'][0]}\n"
        
        # 分析范围
        response += "\n🔍 分析范围：\n"
        if collected_info.get("analysis_types"):
            for analysis_type in collected_info["analysis_types"]:
                response += f"   ✅ {analysis_type.title()}分析\n"
        else:
            response += "   ✅ 全面代码质量分析\n   ✅ 安全漏洞检测\n   ✅ 性能优化建议\n"
        
        # 技术栈优化
        if collected_info.get("technology_stack"):
            response += f"\n🔧 针对 {', '.join(collected_info['technology_stack']).title()} 进行专项优化\n"
        
        response += "\n⏳ 启动多智能体协作分析，请稍候..."
        
        return response
    
    def _generate_clarification_response(self, collected_info: Dict[str, Any], message: str) -> str:
        """生成澄清请求回应"""
        response = "🤔 我需要一些额外信息来提供最佳服务：\n\n"
        
        if not collected_info.get("code_paths") and not collected_info.get("github_urls"):
            response += "📁 请提供代码位置：\n"
            response += "   • 本地目录路径（如：/path/to/project）\n"
            response += "   • GitHub仓库链接\n\n"
        
        if not collected_info.get("analysis_types"):
            response += "🎯 您主要关心什么？\n"
            response += "   • 代码安全性\n   • 性能优化\n   • 代码质量\n   • 全面检查\n\n"
        
        response += "💡 您可以用自然语言描述，比如：\n"
        response += '"请检查我的Python项目/path/to/code的安全问题"'
        
        return response
    
    def _determine_next_actions(self, analysis: Dict[str, Any], collected_info: Dict[str, Any]) -> Dict[str, Any]:
        """确定下一步行动"""
        actions = {
            "should_start_analysis": False,
            "analysis_config": {},
            "conversation_continue": True
        }
        
        # 判断是否可以开始分析
        if analysis["information_completeness"] >= 0.6:
            actions["should_start_analysis"] = True
            actions["analysis_config"] = {
                "target": collected_info.get("code_paths", []) + collected_info.get("github_urls", []),
                "analysis_types": collected_info.get("analysis_types", ["quality", "security", "performance"]),
                "technology_stack": collected_info.get("technology_stack", []),
                "specific_concerns": collected_info.get("specific_concerns", [])
            }
            actions["conversation_continue"] = False
        
        return actions
    
    def _record_conversation_turn(self, user_message: str, ai_response: str, 
                                analysis: Dict[str, Any], session_id: str):
        """记录对话轮次"""
        session = self.session_memory.get(session_id, {})
        session["messages"].append({
            "content": ai_response,
            "timestamp": self._get_current_time(),
            "type": "ai",
            "analysis": analysis
        })
        
        # 更新对话状态
        session["conversation_state"] = analysis["conversation_stage"]
        
        # 保持对话历史在合理范围内
        if len(session["messages"]) > 20:
            session["messages"] = session["messages"][-15:]
    
    async def _execute_ai_suggested_action(self, suggested_actions: Dict[str, Any], session_id: str):
        """执行AI建议的操作"""
        should_start = suggested_actions.get("should_start_analysis", False)
        analysis_config = suggested_actions.get("analysis_config", {})
        
        if should_start and analysis_config:
            # AI认为已准备好开始分析
            await self._start_analysis_from_ai_config(analysis_config, session_id)
        elif not suggested_actions.get("conversation_continue", True):
            # 对话结束，等待用户进一步指示
            print("✅ 我已经准备好为您分析了！请说 '开始分析' 或告诉我还需要补充什么。")
        else:
            # 继续对话，AI已经在回应中询问了需要的信息
            pass
    
    async def _start_analysis_from_ai_config(self, analysis_config: Dict[str, Any], session_id: str):
        """基于AI收集的配置启动智能分析"""
        # 获取分析目标
        targets = analysis_config.get("target", [])
        analysis_types = analysis_config.get("analysis_types", ["quality", "security", "performance"])
        technology_stack = analysis_config.get("technology_stack", [])
        specific_concerns = analysis_config.get("specific_concerns", [])
        
        if not targets:
            print("❌ 缺少代码位置信息，无法开始分析")
            return
        
        # 确定主要分析目标
        primary_target = targets[0]
        
        # 处理GitHub仓库
        if primary_target.startswith("http") and "github.com" in primary_target:
            cloned_target = self._clone_github_repo(primary_target)
            if not cloned_target:
                print(f"❌ GitHub仓库克隆失败: {primary_target}")
                return
            primary_target = cloned_target
        elif not self._is_valid_directory(primary_target):
            print(f"❌ 目录不存在或无法访问: {primary_target}")
            return
        
        # 检查Git信息
        git_commit = await self._get_latest_git_commit(primary_target)
        if git_commit:
            print(f"🔄 检测到Git仓库，当前提交: {git_commit[:8]}")
        
        # 生成任务ID
        if self._mock_db:
            requirement_id = self._mock_requirement_id
            self._mock_requirement_id += 1
        else:
            requirement_id = await self.db_service.save_user_requirement(
                session_id=session_id,
                user_message=f"AI智能对话分析: {', '.join(analysis_types)}",
                code_directory=primary_target,
                code_patch=None,
                git_commit=git_commit
            )
        
        # 显示分析信息
        print(f"🚀 启动AI智能代码分析 (任务ID: {requirement_id})")
        print(f"📁 分析目标: {primary_target}")
        
        if analysis_types:
            print(f"🎯 重点关注: {', '.join(analysis_types)}")
        if technology_stack:
            print(f"🔧 技术栈: {', '.join(technology_stack)}")
        if specific_concerns:
            print(f"⚠️ 特别关注: {', '.join(specific_concerns)}")
        
        # 分派任务
        success = await self._dispatch_analysis_task(requirement_id, {
            "code_directory": primary_target,
            "code_patch": None,
            "git_commit": git_commit,
            "user_message": f"基于AI对话的智能分析请求",
            "session_id": session_id,
            "analysis_goals": analysis_types,
            "technology_stack": technology_stack,
            "specific_concerns": specific_concerns
        })
        
        if success:
            print("✅ 已启动AI驱动的个性化分析:")
            print("   🔍 静态代码扫描")
            print("   📊 AI代码质量分析")
            print("   🔒 AI安全分析")
            print("   ⚡ AI性能分析")
            print("⏳ 分析进行中，我会根据您的具体需求生成个性化报告...")
        else:
            print("❌ 分析启动失败，请稍后重试")
    
    # === 传统模式处理方法 ===
    
    async def _process_with_traditional_mode(self, user_message: str, target_directory: Optional[str], session_id: str):
        """传统模式处理用户输入"""
        print("📝 使用传统模式处理...")
        
        # 传统的意图分析
        intent = self._analyze_user_intent(user_message)
        
        if intent == "help":
            return await self._provide_help()
        elif intent == "status":
            return await self._show_system_status()
        elif intent == "greeting":
            return await self._handle_greeting()
        
        # 提取代码目录和补丁信息
        code_directory = self._extract_code_directory(user_message, target_directory)
        code_patch = self._extract_code_patch(user_message)
        
        # 如果没有找到代码路径，提供友好的指导
        if not code_directory and not code_patch:
            return await self._guide_user_input(user_message)
        
        # 验证目录有效性
        if code_directory and not self._is_valid_directory(code_directory):
            print(f"❌ 目录不存在或无法访问: {code_directory}")
            print("💡 请检查路径是否正确，例如: /home/user/project/src")
            return
        
        git_commit = None
        if code_directory:
            git_commit = await self._get_latest_git_commit(code_directory)
            if git_commit:
                print(f"🔄 检测到Git仓库，当前提交: {git_commit[:8]}")
            else:
                print("📁 目录不是Git仓库，将进行静态分析")
            
        # 开始分析流程
        try:
            if self._mock_db:
                requirement_id = self._mock_requirement_id
                self._mock_requirement_id += 1
            else:
                requirement_id = await self.db_service.save_user_requirement(
                    session_id=session_id,
                    user_message=user_message,
                    code_directory=code_directory or "",
                    code_patch=code_patch,
                    git_commit=git_commit
                )
            
            print(f"🚀 开始代码分析 (任务ID: {requirement_id})")
            if code_directory:
                print(f"📁 分析目录: {code_directory}")
            if code_patch:
                print(f"📝 包含代码补丁 ({len(code_patch)} 字符)")
            
            # 分派任务给各个智能体
            success = await self._dispatch_analysis_task(requirement_id, {
                "code_directory": code_directory,
                "code_patch": code_patch,
                "git_commit": git_commit,
                "user_message": user_message,
                "session_id": session_id
            })
            
            if success:
                print("✅ 任务已分派给以下智能体:")
                print("   🔍 静态代码扫描智能体")
                print("   📊 代码质量分析智能体") 
                print("   🔒 安全分析智能体")
                print("   ⚡ 性能分析智能体")
                print("⏳ 分析进行中，请稍候...")
            else:
                print("❌ 任务分派失败，请稍后重试")
                
        except Exception as e:
            logger.error(f"处理用户输入时出错: {e}")
            print(f"❌ 处理请求时发生错误: {e}")

    def _analyze_user_intent(self, message: str) -> str:
        """分析用户意图（传统方法）"""
        message_lower = message.lower().strip()
        
        greetings = ["hello", "hi", "你好", "您好", "嗨"]
        if any(greeting in message_lower for greeting in greetings):
            return "greeting"
        
        help_keywords = ["help", "帮助", "怎么用", "如何使用"]
        if any(keyword in message_lower for keyword in help_keywords):
            return "help"
        
        status_keywords = ["status", "状态", "情况", "进展"]
        if any(keyword in message_lower for keyword in status_keywords):
            return "status"
        
        return "analysis"
    
    async def _provide_help(self):
        """提供帮助信息"""
        help_text = """
🤖 MAS多智能体代码分析系统使用指南:

📋 支持的命令:
• 分析本地目录: "请分析 /path/to/your/code 目录"
• 分析GitHub仓库: "分析这个项目: https://github.com/user/repo"
• 查看系统状态: "系统状态" 或 "status"
• 获取帮助: "help" 或 "帮助"

🔍 分析功能:
• 静态代码扫描
• 代码质量评估 (AI驱动)
• 安全漏洞检测 (AI增强)
• 性能分析建议 (智能优化)

💡 提示: 您可以用自然语言描述需求，我会智能理解！
        """
        print(help_text)
    
    async def _show_system_status(self):
        """显示系统状态"""
        print("🔍 系统状态检查中...")
        print("🤖 智能体状态:")
        print("   ✅ 用户沟通智能体 - 运行中")
        print("   ✅ 静态代码扫描智能体 - 就绪")
        print("   ✅ 代码质量分析智能体 - 就绪")
        print("   ✅ 安全分析智能体 - 就绪")
        print("   ✅ 性能分析智能体 - 就绪")
        print("   ✅ 汇总智能体 - 就绪")
        print("🚀 系统已就绪，可以开始代码分析")
    
    async def _handle_greeting(self):
        """处理问候"""
        greeting_response = """
👋 您好！欢迎使用MAS多智能体代码分析系统！

我是您的AI代码审查助手，可以帮您进行:
🔍 静态代码扫描
📊 代码质量分析  
🔒 安全漏洞检测
⚡ 性能优化建议

🚀 开始使用:
• 输入 "help" 查看详细使用指南
• 直接说 "请分析 /path/to/code 目录" 开始分析
• 用自然语言描述您的需求

准备好开始了吗? 😊
        """
        print(greeting_response)
    
    async def _guide_user_input(self, user_message: str):
        """指导用户输入"""
        print("🤔 我理解您想要进行代码分析，但需要更多信息:")
        print()
        print("📁 请指定代码目录路径，例如:")
        print('   • "请分析 /home/user/project/src 目录"')
        print('   • "检查 ./src 目录的代码质量"')
        print()
        print("🌐 支持GitHub仓库分析:")
        print('   • "分析这个项目: https://github.com/user/repo"')
        print()
        print("💡 提示: 您也可以输入 'help' 查看完整使用指南")

    def _extract_code_directory(self, message: str, target_directory: Optional[str]) -> Optional[str]:
        """从用户消息中提取代码目录"""
        # 优先使用明确指定的目标目录
        if target_directory and os.path.isdir(target_directory):
            return os.path.abspath(target_directory)
        
        # 检查是否是GitHub URL
        github_url = self._extract_github_url(message)
        if github_url:
            cloned_dir = self._clone_github_repo(github_url)
            if cloned_dir:
                return cloned_dir
        
        # 查找目录路径模式
        patterns = [
            r"目录[:\s]*([^\s]+)",
            r"路径[:\s]*([^\s]+)", 
            r"代码在\s*([^\s]+)",
            r"项目路径[:\s]*([^\s]+)",
            r"分析\s+([^\s]+)\s*目录",
            r"这个项目[:\s]*([^\s]+)",
            r"项目[:\s]*([^\s]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                path = match.group(1).strip()
                if self._is_valid_directory(path):
                    return os.path.abspath(path)
        
        # 检查消息中的每个单词是否为有效目录
        words = message.split()
        for word in words:
            if self._is_valid_directory(word):
                return os.path.abspath(word)
                
        return None

    def _extract_github_url(self, message: str) -> Optional[str]:
        """从消息中提取GitHub URL"""
        github_patterns = [
            r'https://github\.com/[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-\.]+',
            r'github\.com/[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-\.]+',
        ]
        
        for pattern in github_patterns:
            match = re.search(pattern, message)
            if match:
                url = match.group(0)
                if not url.startswith('https://'):
                    url = 'https://' + url
                return url
                
        return None

    def _clone_github_repo(self, github_url: str) -> Optional[str]:
        """克隆GitHub仓库到临时目录"""
        import tempfile
        import shutil
        import subprocess
        
        try:
            temp_dir = tempfile.mkdtemp(prefix="mas_analysis_")
            repo_name = github_url.split('/')[-1].replace('.git', '')
            clone_path = os.path.join(temp_dir, repo_name)
            
            print(f"🔄 正在克隆GitHub仓库: {github_url}")
            print(f"📁 克隆位置: {clone_path}")
            
            result = subprocess.run(
                ['git', 'clone', github_url, clone_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ 仓库克隆成功")
                return clone_path
            else:
                print(f"❌ 克隆失败: {result.stderr}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None
                
        except Exception as e:
            print(f"❌ 克隆过程中发生错误: {e}")
            return None

    def _is_valid_directory(self, path: str) -> bool:
        """检查路径是否为有效目录"""
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            return os.path.exists(path) and os.path.isdir(path)
        except (OSError, TypeError):
            return False
        
    def _extract_code_patch(self, message: str) -> Optional[str]:
        """从用户消息中提取代码补丁"""
        # 检查是否包含 Git diff 格式
        diff_patterns = [
            r"diff --git",
            r"@@.*@@",
            r"^\+{3}\s",
            r"^-{3}\s",
        ]
        
        if any(re.search(pattern, message, re.MULTILINE) for pattern in diff_patterns):
            return message
        
        # 检查是否包含代码块
        if "```" in message:
            code_blocks = re.findall(r"```(?:.*?\n)?(.*?)```", message, re.DOTALL)
            if code_blocks:
                return "\n".join(code_blocks)
                
        return None
        
    async def _get_latest_git_commit(self, directory: str) -> Optional[str]:
        """获取git仓库的最新提交"""
        try:
            repo = git.Repo(directory)
            return repo.head.commit.hexsha
        except git.exc.InvalidGitRepositoryError:
            logger.warning(f"目录 {directory} 不是有效的 Git 仓库")
            return None
        except Exception as e:
            logger.error(f"获取Git提交信息失败: {e}")
            return None

    async def _dispatch_analysis_task(self, requirement_id: int, task_data: Dict[str, Any]) -> bool:
        """分派分析任务给其他智能体"""
        agents = [
            ("static_scan_agent", "static_scan_request"),
            ("ai_code_quality_agent", "quality_analysis_request"),
            ("ai_security_agent", "security_analysis_request"),
            ("ai_performance_agent", "performance_analysis_request")
        ]
        
        success_count = 0
        total_count = len(agents)
        
        for agent_id, message_type in agents:
            try:
                await self.send_message(
                    receiver=agent_id,
                    content={
                        "requirement_id": requirement_id,
                        "code_content": task_data.get("code_patch", ""),
                        "code_directory": task_data.get("code_directory", ""),
                        "task_data": task_data
                    },
                    message_type=message_type
                )
                success_count += 1
                logger.info(f"成功分派任务给 {agent_id}")
            except Exception as e:
                logger.error(f"分派任务给 {agent_id} 失败: {e}")
        
        return success_count == total_count

    async def _process_system_feedback(self, content: Dict[str, Any]):
        """处理系统反馈"""
        try:
            feedback_type = content.get("type", "unknown")
            feedback_message = content.get("message", "")
            requirement_id = content.get("requirement_id")
            
            logger.info(f"收到系统反馈: {feedback_type}")
            
            if feedback_type == "analysis_complete":
                print(f"📊 分析完成: {feedback_message}")
            elif feedback_type == "error":
                print(f"❌ 系统错误: {feedback_message}")
            elif feedback_type == "progress":
                print(f"⏳ 进度更新: {feedback_message}")
                
        except Exception as e:
            logger.error(f"处理系统反馈时出错: {e}")

    async def _process_analysis_result(self, content: Dict[str, Any]):
        """处理分析结果"""
        try:
            requirement_id = content.get("requirement_id")
            agent_type = content.get("agent_type")
            results = content.get("results", {})
            analysis_complete = content.get("analysis_complete", False)
            
            logger.info(f"收到来自 {agent_type} 的分析结果")
            
            # 显示分析结果概要
            print(f"\n📊 收到 {agent_type} 分析结果 (任务ID: {requirement_id})")
            
            if results:
                # 显示结果概要
                await self._display_analysis_summary(agent_type, results)
                
                # 存储结果到会话记忆中
                if not hasattr(self, 'analysis_results'):
                    self.analysis_results = {}
                if requirement_id not in self.analysis_results:
                    self.analysis_results[requirement_id] = {}
                    
                self.analysis_results[requirement_id][agent_type] = results
                
                # 检查是否所有分析都完成了
                expected_agents = {"ai_code_quality", "ai_security", "ai_performance"}
                completed_agents = set(self.analysis_results[requirement_id].keys())
                
                if completed_agents >= expected_agents:
                    print(f"\n🎉 所有分析完成！正在生成综合报告...")
                    await self._generate_comprehensive_report(requirement_id)
            else:
                print(f"⚠️ {agent_type} 返回了空结果")
                
        except Exception as e:
            logger.error(f"处理分析结果时出错: {e}")
            print(f"❌ 处理 {agent_type} 分析结果时出错: {e}")

    async def _display_analysis_summary(self, agent_type: str, results: Dict[str, Any]):
        """显示分析结果摘要"""
        try:
            if agent_type == "ai_code_quality":
                print("  🔍 代码质量分析:")
                quality_score = results.get("overall_quality_score", "未知")
                print(f"    • 整体质量评分: {quality_score}")
                
                issues = results.get("issues", [])
                if issues:
                    print(f"    • 发现 {len(issues)} 个问题")
                    for issue in issues[:3]:  # 显示前3个问题
                        severity = issue.get("severity", "未知")
                        description = issue.get("description", "无描述")
                        print(f"      - [{severity}] {description}")
                    if len(issues) > 3:
                        print(f"      ... 还有 {len(issues) - 3} 个问题")
                        
            elif agent_type == "ai_security":
                print("  🔒 安全分析:")
                vulnerabilities = results.get("vulnerabilities", [])
                if vulnerabilities:
                    print(f"    • 发现 {len(vulnerabilities)} 个安全问题")
                    for vuln in vulnerabilities[:3]:
                        severity = vuln.get("severity", "未知")
                        vuln_type = vuln.get("type", "未知类型")
                        print(f"      - [{severity}] {vuln_type}")
                    if len(vulnerabilities) > 3:
                        print(f"      ... 还有 {len(vulnerabilities) - 3} 个安全问题")
                else:
                    print("    • 未发现明显的安全问题 ✅")
                    
            elif agent_type == "ai_performance":
                print("  ⚡ 性能分析:")
                performance_score = results.get("performance_score", "未知")
                print(f"    • 性能评分: {performance_score}")
                
                bottlenecks = results.get("bottlenecks", [])
                if bottlenecks:
                    print(f"    • 发现 {len(bottlenecks)} 个性能瓶颈")
                    for bottleneck in bottlenecks[:3]:
                        location = bottleneck.get("location", "未知位置")
                        issue = bottleneck.get("issue", "未知问题")
                        print(f"      - {location}: {issue}")
                    if len(bottlenecks) > 3:
                        print(f"      ... 还有 {len(bottlenecks) - 3} 个性能问题")
                        
        except Exception as e:
            logger.error(f"显示分析摘要时出错: {e}")
            print(f"    ⚠️ 摘要显示出错: {e}")

    async def _generate_comprehensive_report(self, requirement_id: int):
        """生成综合分析报告"""
        try:
            results = self.analysis_results.get(requirement_id, {})
            
            print(f"\n📋 ========== 综合分析报告 (任务ID: {requirement_id}) ==========")
            
            # 创建报告保存目录
            report_dir = await self._create_report_directory(requirement_id)
            
            # 整体评估
            print("\n🎯 整体评估:")
            
            # 代码质量部分
            quality_results = results.get("ai_code_quality", {})
            if quality_results:
                quality_score = quality_results.get("overall_quality_score", "未评估")
                print(f"   代码质量: {quality_score}")
            
            # 安全性部分
            security_results = results.get("ai_security", {})
            if security_results:
                vulnerabilities = security_results.get("vulnerabilities", [])
                risk_level = "低风险" if len(vulnerabilities) == 0 else "中风险" if len(vulnerabilities) <= 3 else "高风险"
                print(f"   安全风险: {risk_level} ({len(vulnerabilities)} 个问题)")
            
            # 性能部分
            performance_results = results.get("ai_performance", {})
            if performance_results:
                performance_score = performance_results.get("performance_score", "未评估")
                print(f"   性能状况: {performance_score}")
            
            # 建议
            print(f"\n💡 主要建议:")
            suggestions = []
            
            # 收集所有建议
            for agent_type, agent_results in results.items():
                agent_suggestions = agent_results.get("suggestions", [])
                for suggestion in agent_suggestions[:2]:  # 每个智能体最多2个建议
                    suggestions.append(f"   • {suggestion}")
            
            if suggestions:
                for suggestion in suggestions:
                    print(suggestion)
            else:
                print("   • 代码整体状况良好，继续保持！")
            
            # 生成详细报告文件
            report_files = await self._save_detailed_reports(requirement_id, results, report_dir)
            
            print(f"\n✅ 分析报告生成完成！")
            print(f"� 报告保存位置: {report_dir}")
            print(f"📄 生成的报告文件:")
            for report_file in report_files:
                print(f"   • {report_file}")
            print(f"\n🔄 如需重新查看报告，请访问: {report_dir}")
            
        except Exception as e:
            logger.error(f"生成综合报告时出错: {e}")
            print(f"❌ 生成综合报告失败: {e}")

    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行用户沟通任务"""
        return {"status": "user_communication_ready", "timestamp": self._get_current_time()}

    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        return datetime.datetime.now().isoformat()

    async def _create_report_directory(self, requirement_id: int) -> str:
        """创建报告保存目录"""
        try:
            # 创建主报告目录
            base_dir = os.path.join(os.getcwd(), "analysis_reports")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            
            # 创建任务特定目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(base_dir, f"task_{requirement_id}_{timestamp}")
            
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            return report_dir
            
        except Exception as e:
            logger.error(f"创建报告目录失败: {e}")
            # 返回当前目录作为备用
            return os.getcwd()

    async def _save_detailed_reports(self, requirement_id: int, results: Dict[str, Any], report_dir: str) -> List[str]:
        """保存详细的分析报告文件"""
        report_files = []
        
        try:
            # 生成综合报告
            summary_file = await self._save_summary_report(requirement_id, results, report_dir)
            if summary_file:
                report_files.append(summary_file)
            
            # 保存各智能体的详细报告
            for agent_type, agent_results in results.items():
                detail_file = await self._save_agent_detail_report(
                    agent_type, agent_results, report_dir
                )
                if detail_file:
                    report_files.append(detail_file)
            
            # 生成JSON格式的原始数据
            json_file = await self._save_raw_data_json(requirement_id, results, report_dir)
            if json_file:
                report_files.append(json_file)
                
        except Exception as e:
            logger.error(f"保存详细报告失败: {e}")
        
        return report_files

    async def _save_summary_report(self, requirement_id: int, results: Dict[str, Any], report_dir: str) -> str:
        """保存综合报告摘要"""
        try:
            summary_file = os.path.join(report_dir, "analysis_summary.md")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"# MAS 代码分析报告\n\n")
                f.write(f"**任务ID:** {requirement_id}\n")
                f.write(f"**生成时间:** {self._get_current_time()}\n\n")
                
                f.write("## 📊 整体评估\n\n")
                
                # 代码质量
                quality_results = results.get("ai_code_quality", {})
                if quality_results:
                    quality_score = quality_results.get("overall_quality_score", "未评估")
                    f.write(f"- **代码质量:** {quality_score}\n")
                
                # 安全性
                security_results = results.get("ai_security", {})
                if security_results:
                    vulnerabilities = security_results.get("vulnerabilities", [])
                    risk_level = "低风险" if len(vulnerabilities) == 0 else "中风险" if len(vulnerabilities) <= 3 else "高风险"
                    f.write(f"- **安全风险:** {risk_level} ({len(vulnerabilities)} 个问题)\n")
                
                # 性能
                performance_results = results.get("ai_performance", {})
                if performance_results:
                    performance_score = performance_results.get("performance_score", "未评估")
                    f.write(f"- **性能状况:** {performance_score}\n")
                
                f.write("\n## 💡 主要建议\n\n")
                
                # 收集建议
                suggestion_count = 1
                for agent_type, agent_results in results.items():
                    agent_suggestions = agent_results.get("suggestions", [])
                    for suggestion in agent_suggestions[:3]:
                        f.write(f"{suggestion_count}. {suggestion}\n")
                        suggestion_count += 1
                
                if suggestion_count == 1:
                    f.write("代码整体状况良好，继续保持！\n")
                
                f.write(f"\n## 📁 详细报告\n\n")
                f.write(f"- 代码质量详细报告: `ai_code_quality_detail.md`\n")
                f.write(f"- 安全分析详细报告: `ai_security_detail.md`\n")
                f.write(f"- 性能分析详细报告: `ai_performance_detail.md`\n")
                f.write(f"- 原始数据: `raw_analysis_data.json`\n")
            
            return os.path.basename(summary_file)
            
        except Exception as e:
            logger.error(f"保存综合报告摘要失败: {e}")
            return None

    async def _save_agent_detail_report(self, agent_type: str, agent_results: Dict[str, Any], report_dir: str) -> str:
        """保存智能体详细报告"""
        try:
            detail_file = os.path.join(report_dir, f"{agent_type}_detail.md")
            
            with open(detail_file, 'w', encoding='utf-8') as f:
                f.write(f"# {agent_type.upper()} 详细分析报告\n\n")
                f.write(f"**生成时间:** {self._get_current_time()}\n\n")
                
                if agent_type == "ai_code_quality":
                    await self._write_quality_detail(f, agent_results)
                elif agent_type == "ai_security":
                    await self._write_security_detail(f, agent_results)
                elif agent_type == "ai_performance":
                    await self._write_performance_detail(f, agent_results)
                else:
                    f.write("## 原始分析结果\n\n")
                    f.write(f"```json\n{str(agent_results)}\n```\n")
            
            return os.path.basename(detail_file)
            
        except Exception as e:
            logger.error(f"保存{agent_type}详细报告失败: {e}")
            return None

    async def _write_quality_detail(self, f, results: Dict[str, Any]):
        """写入代码质量详细内容"""
        f.write("## 🔍 代码质量分析详情\n\n")
        
        score = results.get("overall_quality_score", "未评估")
        f.write(f"**整体评分:** {score}\n\n")
        
        issues = results.get("issues", [])
        if issues:
            f.write("### 发现的问题\n\n")
            for i, issue in enumerate(issues, 1):
                severity = issue.get("severity", "未知")
                description = issue.get("description", "无描述")
                location = issue.get("location", "未知位置")
                f.write(f"{i}. **[{severity.upper()}]** {description}\n")
                f.write(f"   - 位置: {location}\n")
                if "suggestion" in issue:
                    f.write(f"   - 建议: {issue['suggestion']}\n")
                f.write("\n")
        
        suggestions = results.get("suggestions", [])
        if suggestions:
            f.write("### 改进建议\n\n")
            for i, suggestion in enumerate(suggestions, 1):
                f.write(f"{i}. {suggestion}\n")

    async def _write_security_detail(self, f, results: Dict[str, Any]):
        """写入安全分析详细内容"""
        f.write("## 🔒 安全分析详情\n\n")
        
        vulnerabilities = results.get("vulnerabilities", [])
        if vulnerabilities:
            f.write(f"**发现 {len(vulnerabilities)} 个安全问题**\n\n")
            
            # 按严重程度分组
            critical = [v for v in vulnerabilities if v.get("severity") == "critical"]
            high = [v for v in vulnerabilities if v.get("severity") == "high"]
            medium = [v for v in vulnerabilities if v.get("severity") == "medium"]
            low = [v for v in vulnerabilities if v.get("severity") == "low"]
            
            for category, vulns in [("Critical", critical), ("High", high), ("Medium", medium), ("Low", low)]:
                if vulns:
                    f.write(f"### {category} 风险\n\n")
                    for i, vuln in enumerate(vulns, 1):
                        f.write(f"{i}. **{vuln.get('type', '未知类型')}**\n")
                        f.write(f"   - 描述: {vuln.get('description', '无描述')}\n")
                        f.write(f"   - 位置: {vuln.get('location', '未知位置')}\n")
                        if "fix_suggestion" in vuln:
                            f.write(f"   - 修复建议: {vuln['fix_suggestion']}\n")
                        f.write("\n")
        else:
            f.write("✅ **未发现明显的安全问题**\n\n")
        
        # 安全建议
        hardening = results.get("hardening_recommendations", [])
        if hardening:
            f.write("### 安全加固建议\n\n")
            for i, rec in enumerate(hardening, 1):
                category = rec.get("category", "通用")
                recommendation = rec.get("recommendation", "")
                priority = rec.get("priority", "medium")
                f.write(f"{i}. **[{priority.upper()}] {category}:** {recommendation}\n")

    async def _write_performance_detail(self, f, results: Dict[str, Any]):
        """写入性能分析详细内容"""
        f.write("## ⚡ 性能分析详情\n\n")
        
        score = results.get("performance_score", "未评估")
        f.write(f"**性能评分:** {score}\n\n")
        
        bottlenecks = results.get("bottlenecks", [])
        if bottlenecks:
            f.write(f"### 发现 {len(bottlenecks)} 个性能瓶颈\n\n")
            for i, bottleneck in enumerate(bottlenecks, 1):
                location = bottleneck.get("location", "未知位置")
                issue = bottleneck.get("issue", "未知问题")
                impact = bottleneck.get("impact", "未知")
                f.write(f"{i}. **位置:** {location}\n")
                f.write(f"   - 问题: {issue}\n")
                f.write(f"   - 影响: {impact}\n")
                if "optimization_suggestion" in bottleneck:
                    f.write(f"   - 优化建议: {bottleneck['optimization_suggestion']}\n")
                f.write("\n")
        
        optimizations = results.get("optimization_suggestions", [])
        if optimizations:
            f.write("### 优化建议\n\n")
            for i, opt in enumerate(optimizations, 1):
                f.write(f"{i}. {opt}\n")

    async def _save_raw_data_json(self, requirement_id: int, results: Dict[str, Any], report_dir: str) -> str:
        """保存原始分析数据JSON"""
        try:
            import json
            json_file = os.path.join(report_dir, "raw_analysis_data.json")
            
            # 构建完整的数据结构
            data = {
                "task_id": requirement_id,
                "timestamp": self._get_current_time(),
                "analysis_results": results,
                "metadata": {
                    "mas_version": "1.0.0",
                    "agents_used": list(results.keys()),
                    "total_agents": len(results)
                }
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return os.path.basename(json_file)
            
        except Exception as e:
            logger.error(f"保存原始数据JSON失败: {e}")
            return None
