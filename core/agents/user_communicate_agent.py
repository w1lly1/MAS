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
        
        logger.info(f"处理来自 {source} 的用户输入: {user_message[:100]}...")
        print(f"🤖 处理来自 {source} 的用户输入: {user_message[:100]}...")
        
        # 提取代码目录和补丁信息
        code_directory = self._extract_code_directory(user_message, target_directory)
        code_patch = self._extract_code_patch(user_message)
        git_commit = None
        
        # 如果没有提供补丁，尝试获取git最新提交
        if not code_patch and code_directory:
            git_commit = await self._get_latest_git_commit(code_directory)
            
        # 保存用户需求到数据库
        try:
            requirement_id = await self.db_service.save_user_requirement(
                session_id=session_id,
                user_message=user_message,
                code_directory=code_directory or "",
                code_patch=code_patch,
                git_commit=git_commit
            )
            
            # 检查是否包含系统问题反馈
            await self._check_and_update_knowledge(user_message)
            
            # 转发任务给其他智能体
            if code_directory or code_patch:
                success = await self._dispatch_analysis_task(requirement_id, {
                    "code_directory": code_directory,
                    "code_patch": code_patch,
                    "git_commit": git_commit,
                    "user_message": user_message,
                    "session_id": session_id
                })
                
                if success:
                    print(f"✅ 已接收分析请求 #{requirement_id}")
                    if code_directory:
                        print(f"📁 代码目录: {code_directory}")
                    if git_commit:
                        print(f"🔄 Git提交: {git_commit[:8]}...")
                    print("🤖 智能体正在处理中，请稍候...")
                else:
                    print(f"❌ 分派任务失败，请稍后重试")
                
            else:
                print("❌ 请提供代码目录路径或代码补丁以进行分析")
                print("💡 示例: '请分析 /path/to/code 目录'")
                
        except Exception as e:
            logger.error(f"处理用户输入时出错: {e}")
            print(f"❌ 处理用户输入时出错: {e}")

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