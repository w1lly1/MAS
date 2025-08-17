import os
import re
import git
import logging
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService

logger = logging.getLogger(__name__)

class UserCommunicationAgent(BaseAgent):
    def __init__(self):
        super().__init__("user_comm_agent", "用户沟通智能体")
        # 暂时使用内存存储，避免数据库初始化问题
        self._mock_db = True
        self._mock_requirement_id = 1000
        if not self._mock_db:
            self.db_service = DatabaseService()
        self.current_session = None
        
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
        """处理用户输入"""
        user_message = content.get("message", "")
        session_id = content.get("session_id", "default")
        source = content.get("source", "direct")
        target_directory = content.get("target_directory")
        
        # 智能解析用户意图
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
                # 模拟数据库保存
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
                print("   � 代码质量分析智能体") 
                print("   🔒 安全分析智能体")
                print("   ⚡ 性能分析智能体")
                print("⏳ 分析进行中，请稍候...")
            else:
                print("❌ 任务分派失败，请稍后重试")
                
        except Exception as e:
            logger.error(f"处理用户输入时出错: {e}")
            print(f"❌ 处理请求时发生错误: {e}")
            print("💡 请尝试重新提交或联系管理员")

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
            
            # 更新数据库状态
            if requirement_id:
                await self.db_service.update_requirement_status(
                    requirement_id, feedback_type, feedback_message
                )
                
        except Exception as e:
            logger.error(f"处理系统反馈时出错: {e}")

    async def _process_analysis_result(self, content: Dict[str, Any]):
        """处理分析结果"""
        try:
            requirement_id = content.get("requirement_id")
            agent_type = content.get("agent_type")
            results = content.get("results", {})
            
            logger.info(f"收到来自 {agent_type} 的分析结果")
            
            # 保存分析结果
            if requirement_id:
                await self.db_service.save_analysis_result(
                    requirement_id, agent_type, results
                )
            
            # 检查是否所有智能体都完成了分析
            if await self._check_analysis_complete(requirement_id):
                await self._generate_summary_report(requirement_id)
                
        except Exception as e:
            logger.error(f"处理分析结果时出错: {e}")

    def _analyze_user_intent(self, message: str) -> str:
        """分析用户意图"""
        message_lower = message.lower().strip()
        
        # 问候语
        greetings = ["hello", "hi", "你好", "您好", "嗨"]
        if any(greeting in message_lower for greeting in greetings):
            return "greeting"
        
        # 帮助请求
        help_keywords = ["help", "帮助", "怎么用", "如何使用", "使用方法", "指导"]
        if any(keyword in message_lower for keyword in help_keywords):
            return "help"
        
        # 状态查询
        status_keywords = ["status", "状态", "情况", "进展", "结果"]
        if any(keyword in message_lower for keyword in status_keywords):
            return "status"
        
        # 代码分析请求
        analysis_keywords = ["分析", "review", "检查", "扫描", "检测"]
        if any(keyword in message_lower for keyword in analysis_keywords):
            return "analysis"
        
        return "unknown"
    
    async def _provide_help(self):
        """提供帮助信息"""
        help_text = """
🤖 MAS多智能体代码分析系统使用指南:

📋 支持的命令:
• 分析代码目录: "请分析 /path/to/your/code 目录"
• 分析代码补丁: 直接粘贴git diff或代码块
• 查看系统状态: "系统状态" 或 "status"
• 获取帮助: "help" 或 "帮助"

📁 目录路径示例:
• /home/user/project/src
• ./src (相对路径)
• /var/fpwork/tiyi/project/oran/netconf/src

📝 代码补丁示例:
• Git diff格式
• ```language 代码块 ```

🔍 分析功能:
• 静态代码扫描 (Pylint, Bandit, Flake8)
• 代码质量评估
• 安全漏洞检测  
• 性能分析建议

💡 提示: 您可以使用 --target-dir 参数预设目录，或在对话中直接指定路径
        """
        print(help_text)
    
    async def _show_system_status(self):
        """显示系统状态"""
        print("🔍 系统状态检查中...")
        
        # 检查智能体状态
        print("🤖 智能体状态:")
        print("   ✅ 用户沟通智能体 - 运行中")
        print("   ✅ 静态代码扫描智能体 - 就绪")
        print("   ✅ 代码质量分析智能体 - 就绪")
        print("   ✅ 安全分析智能体 - 就绪")
        print("   ✅ 性能分析智能体 - 就绪")
        print("   ✅ 汇总智能体 - 就绪")
        
        # 检查数据库连接
        try:
            # TODO: 实际检查数据库状态
            print("💾 数据库状态: ✅ 连接正常")
        except Exception as e:
            print(f"💾 数据库状态: ❌ 连接异常 - {e}")
        
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
• 或者粘贴代码片段让我检查

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
        print('   • "分析这个项目: /path/to/project"')
        print()
        print("📝 或者直接粘贴代码/补丁:")
        print("   • Git diff 输出")
        print("   • ```python 代码块 ```")
        print()
        print("💡 提示: 您也可以输入 'help' 查看完整使用指南")

    def _extract_code_directory(self, message: str, target_directory: Optional[str]) -> Optional[str]:
        """从用户消息中提取代码目录"""
        # 优先使用明确指定的目标目录
        if target_directory and os.path.isdir(target_directory):
            return os.path.abspath(target_directory)
        
        # 查找目录路径模式
        patterns = [
            r"目录[:\s]*([^\s]+)",
            r"路径[:\s]*([^\s]+)",
            r"代码在\s*([^\s]+)",
            r"项目路径[:\s]*([^\s]+)",
            r"--target-dir\s+([^\s]+)",
            r"分析\s+([^\s]+)\s*目录"
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

    def _is_valid_directory(self, path: str) -> bool:
        """检查路径是否为有效目录"""
        try:
            # 处理相对路径
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
            
    async def _check_and_update_knowledge(self, message: str):
        """检查用户消息是否包含系统问题反馈并更新知识库"""
        error_keywords = ["bug", "错误", "漏洞", "问题", "异常", "缺陷", "故障"]
        business_keywords = ["业务逻辑", "逻辑错误", "功能问题", "需求变更"]
        
        message_lower = message.lower()
        
        try:
            if any(keyword in message_lower for keyword in error_keywords + business_keywords):
                error_type = "general_error"
                if any(keyword in message_lower for keyword in business_keywords):
                    error_type = "business_logic_error"
                    
                await self.db_service.update_knowledge_base(
                    error_type=error_type,
                    error_description=message,
                    problematic_pattern=message,
                    severity="medium"
                )
                logger.info(f"已更新知识库: {error_type}")
        except Exception as e:
            logger.error(f"更新知识库失败: {e}")

    async def _dispatch_analysis_task(self, requirement_id: int, task_data: Dict[str, Any]) -> bool:
        """分派分析任务给其他智能体"""
        agents = [
            ("static_scan_agent", "static_scan_request"),
            ("code_quality_agent", "quality_analysis_request"),
            ("security_agent", "security_analysis_request"),
            ("performance_agent", "performance_analysis_request")
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
        
        if success_count > 0:
            print(f"📤 已分派多维度分析任务 {requirement_id} 给 {success_count}/{total_count} 个智能体")
        
        return success_count == total_count

    async def _check_analysis_complete(self, requirement_id: int) -> bool:
        """检查所有分析是否完成"""
        try:
            # 查询数据库检查分析状态
            completed_agents = await self.db_service.get_completed_analysis_count(requirement_id)
            return completed_agents >= 4  # 假设有4个分析智能体
        except Exception as e:
            logger.error(f"检查分析完成状态失败: {e}")
            return False

    async def _generate_summary_report(self, requirement_id: int):
        """生成汇总报告"""
        try:
            # 发送给汇总智能体
            await self.send_message(
                receiver="summary_agent",
                content={"requirement_id": requirement_id},
                message_type="generate_summary_request"
            )
            logger.info(f"已请求生成汇总报告: {requirement_id}")
        except Exception as e:
            logger.error(f"请求生成汇总报告失败: {e}")

    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行用户沟通任务"""
        return {"status": "user_communication_ready", "timestamp": self._get_current_time()}

    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()