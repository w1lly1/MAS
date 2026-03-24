import asyncio
import logging
import uuid
import os
import re
import torch
import tempfile
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path
from infrastructure.reports import report_manager
from utils import log, LogLevel

# 设置环境变量来控制第三方库日志输出
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

try:
    from .agents import (
        AgentManager, 
        AIDrivenUserCommunicationAgent as UserCommunicationAgent,  # 使用正确的导入名称
        SummaryAgent
    )
    # 导入AI驱动智能体
    from .agents.ai_driven_code_quality_agent import AIDrivenCodeQualityAgent
    from .agents.ai_driven_security_agent import AIDrivenSecurityAgent
    from .agents.ai_driven_performance_agent import AIDrivenPerformanceAgent
    from .agents.static_scan_agent import StaticCodeScanAgent
    from .agents.ai_driven_readability_enhancement_agent import AIDrivenReadabilityEnhancementAgent
    from .agents.ai_driven_database_manage_agent import AIDrivenDatabaseManageAgent
    from .agents.base_agent import Message
    from infrastructure.config.ai_agents import get_ai_agent_config, AgentMode
except ImportError as e:
    log("MAS", LogLevel.ERROR, f"❌ 导入智能体类失败: {e}")
    raise

class AgentIntegration:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.agent_manager = AgentManager.get_instance()
            self.agents = {}
            self._system_ready = False
            self.ai_config = get_ai_agent_config()
            if not hasattr(self, 'requirement_counter'):
                self.requirement_counter = 1000
            self._session_id = str(uuid.uuid4())
            self.__class__._initialized = True

    def _has_gpu(self) -> bool:
        """检测是否有GPU可用"""
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            log("MAS", LogLevel.INFO, f"🔧 GPU 可用: {has_gpu}")
            return has_gpu
        except:
            log("MAS", LogLevel.WARNING, "⚠️ GPU 不可用 (torch 导入失败)")
            return False

    async def initialize_system(self, use_cpu_mode: bool = False):
        """初始化智能体系统"""
        if self._system_ready:
            return

        self.use_cpu_mode = use_cpu_mode
        try:
            # 只使用AI驱动智能体
            agent_strategy = self.ai_config.get_agent_selection_strategy()
            mode = self.ai_config.get_agent_mode()
            
            log("MAS", LogLevel.INFO, f"🤖 初始化AI驱动智能体系统 - 模式: {mode.value}")
            
            # 定义系统能力
            ai_agent_classes = {
                # 核心智能体(必需)
                'user_comm': UserCommunicationAgent,
                'data_manage': AIDrivenDatabaseManageAgent,
                'summary': SummaryAgent,
                # AI驱动分析智能体
                'static_scan': StaticCodeScanAgent,
                'ai_code_quality': AIDrivenCodeQualityAgent,
                'ai_security': AIDrivenSecurityAgent,
                'ai_performance': AIDrivenPerformanceAgent,
                'ai_readability_enhancement': AIDrivenReadabilityEnhancementAgent,
            }
            
            # 创建需要的智能体
            agents_to_create = {
                'user_comm': UserCommunicationAgent,
                'data_manage': AIDrivenDatabaseManageAgent,
                'summary': SummaryAgent,
                'ai_readability_enhancement': AIDrivenReadabilityEnhancementAgent
            }
            
            # 添加AI分析智能体
            for analysis_type, agent_name in agent_strategy.items():
                if agent_name in ai_agent_classes:
                    agents_to_create[agent_name] = ai_agent_classes[agent_name]
                else:
                    log("MAS", LogLevel.WARNING, f"智能体 {agent_name} 未找到，跳过")
            
            # 静默初始化 - 一次性显示初始化开始
            log("MAS", LogLevel.INFO, "🚀 正在初始化智能体系统...")
            
            # default GPU in use
            used_device = "gpu"
            if self.use_cpu_mode:
                used_device = "cpu"
            elif not self._has_gpu():
                used_device = "cpu"
            
            # 创建智能体实例 - 静默创建，减少输出
            for name, agent_class in agents_to_create.items():
                try:
                    # 简化日志输出，只输出到日志文件不打印到控制台
                    log("MAS", LogLevel.DEBUG, f"创建AI智能体: {name}")
                    agent_instance = agent_class()
                    agent_instance.used_device = used_device
                    self.agents[name] = agent_instance
                except Exception as e:
                    log("MAS", LogLevel.ERROR, f"创建智能体 {name} 失败: {e}")
                    continue
            
            # 注册到管理器 - 静默注册
            for agent in self.agents.values():
                self.agent_manager.register_agent(agent)
            # 关联 user_comm 与集成器以便回调
            if 'user_comm' in self.agents:
                try:
                    setattr(self.agents['user_comm'], 'agent_integration', self)
                except Exception:
                    pass
            
            # 启动所有智能体
            await self.agent_manager.start_all_agents()
            
            # 初始化AI用户交流功能 - 静默初始化
            if 'user_comm' in self.agents:
                try:
                    ai_comm_init_success = await self.agents['user_comm'].initialize_ai_communication()
                    # 共享模型到数据库管理代理，避免重复加载
                    if ai_comm_init_success and 'data_manage' in self.agents:
                        try:
                            user_comm_agent = self.agents['user_comm']
                            data_manage_agent = self.agents['data_manage']
                            shared_model = getattr(user_comm_agent, "conversation_model", None)
                            shared_tokenizer = getattr(user_comm_agent, "tokenizer", None)
                            if shared_model and hasattr(shared_model, "model"):
                                data_manage_agent.set_shared_model(
                                    model=shared_model.model,
                                    tokenizer=shared_tokenizer,
                                    model_name=getattr(user_comm_agent, "model_name", None),
                                )
                        except Exception as e:
                            log("MAS", LogLevel.WARNING, f"⚠️ 共享模型注入失败: {e}")
                    data_manage_init_success = await self.agents['data_manage'].initialize_data_manage()
                    if not ai_comm_init_success:
                        log("MAS", LogLevel.ERROR, "⚠️ AI交互模块初始化失败，系统可能无法正常处理自然语言")
                    if not data_manage_init_success:
                        log("MAS", LogLevel.ERROR, "⚠️ 数据库管理模块初始化失败，系统可能无法正常处理数据库操作")
                except Exception as e:
                    log("MAS", LogLevel.WARNING, f"⚠️ AI交互或数据库管理模块初始化异常: {e}")
            
            self._system_ready = True
            
            # 简化输出 - 只显示系统就绪状态
            log("MAS", LogLevel.INFO, f"✅ 系统就绪，可以开始交互")
            
        except Exception as e:
            log("MAS", LogLevel.ERROR, f"❌ 系统初始化失败: {e}")
            await self._cleanup_on_error()
            raise
            
    async def _cleanup_on_error(self):
        """错误时清理资源"""
        try:
            if hasattr(self, 'agent_manager'):
                await self.agent_manager.stop_all_agents()
            self.agents.clear()
            self._system_ready = False
        except Exception as e:
            log("MAS", LogLevel.ERROR, f"清理资源时出错: {e}")
        
    async def process_message_from_cli(self, message: str, target_dir: Optional[str] = None) -> str:
        """处理来自命令行的消息"""
        # 检查系统状态但允许空消息传递给AI代理
        if not self._system_ready:
            return "❌ 智能体系统未就绪，请稍后重试"
            
        if 'user_comm' not in self.agents:
            return "❌ 用户沟通智能体不可用"
            
        try:
            # 构造消息内容 - 即使是空消息也传递，让AI代理处理
            content = {
                "message": message or "",  # 确保空消息传为空字符串而非None
                "session_id": self._session_id,  # 使用会话唯一 ID
                "target_directory": target_dir,
                "timestamp": asyncio.get_event_loop().time(),
                "wait_for_db": True
            }
            
            # 发送给用户沟通智能体处理
            await self.agents['user_comm'].handle_message(Message(
                id=str(uuid.uuid4()),  # 生成唯一ID
                sender="cli_interface",
                receiver=self.agents['user_comm'].agent_id,  # Use the actual agent ID
                content=content,
                timestamp=asyncio.get_event_loop().time(),
                message_type="user_input"
            ))
            
            return "✅ 消息已处理"
                
        except Exception as e:
            log("MAS", LogLevel.ERROR, f"处理消息时出错: {e}")
            return f"❌ 处理消息失败: {str(e)}"
        
    async def get_agent_status(self) -> Dict[str, Any]:
        """获取所有智能体状态"""
        status = {
            "system_ready": self._system_ready,
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            try:
                # 假设智能体有状态检查方法
                agent_status = getattr(agent, 'get_status', lambda: {"status": "unknown"})()
                status["agents"][name] = agent_status
            except Exception as e:
                status["agents"][name] = {"status": "error", "error": str(e)}
                
        return status
        
    async def shutdown_system(self):
        """关闭系统"""
        if not self._system_ready:
            log("MAS", LogLevel.INFO, "系统未启动，无需关闭")
            return
            
        try:
            log("MAS", LogLevel.INFO, "正在关闭智能体系统...")
            await self.agent_manager.stop_all_agents()
            self.agents.clear()
            self._system_ready = False
            log("MAS", LogLevel.INFO, "✅ 智能体系统已关闭")
        except Exception as e:
            log("MAS", LogLevel.ERROR, f"关闭系统时出错: {e}")
            raise

    async def switch_agent_mode(self, mode: AgentMode):
        """切换智能体运行模式(当前只支持AI驱动模式)"""
        if mode != AgentMode.AI_DRIVEN:
            log("MAS", LogLevel.WARNING, "⚠️ 当前系统只支持AI驱动模式")
            return
            
        if self._system_ready:
            # 需要重新初始化系统
            await self.shutdown_system()
        
        # 更新配置
        self.ai_config.set_agent_mode(mode)
        
        # 重新初始化
        await self.initialize_system()
        log("MAS", LogLevel.INFO, f"✅ 智能体系统已重新初始化")
    
    def get_active_agents(self) -> Dict[str, str]:
        """获取当前活跃的智能体列表"""
        return {name: agent.__class__.__name__ for name, agent in self.agents.items()}
    
    def get_ai_config_status(self) -> Dict[str, Any]:
        """获取AI配置状态"""
        return self.ai_config.get_config_summary()
    
    async def test_ai_agents(self) -> Dict[str, Any]:
        """测试AI智能体的可用性"""
        test_results = {}
        
        ai_agents = {name: agent for name, agent in self.agents.items() if name.startswith('ai_')}
        
        for name, agent in ai_agents.items():
            try:
                # 尝试初始化AI模型
                if hasattr(agent, '_initialize_models'):
                    await agent._initialize_models()
                    test_results[name] = {"status": "available", "ai_ready": True}
                else:
                    test_results[name] = {"status": "available", "ai_ready": "unknown"}
            except Exception as e:
                test_results[name] = {"status": "error", "error": str(e), "ai_ready": False}
        
        return test_results
    
    async def analyze_directory(self, target_directory: str) -> Dict[str, Any]:
        """统一协调: 针对目录触发 静态扫描 / 代码质量 / 安全 / 性能 / 汇总 分析
        调整: run_init 现在在派发任何具体文件任务之前发送, 以确保 SummaryAgent 能正确记录 run_meta
        步骤:
          1. (可选)克隆 GitHub 仓库
          2. 枚举并预先分配 requirement_id & 读取代码内容
          3. 发送 run_init (包含所有 requirement_ids)
          4. 派发每个文件到各分析智能体
          5. 生成 dispatch 摘要报告
        """
        # 检查输入是否为GitHub URL
        github_url_pattern = r"https?://github\.com/[\w-]+/[\w-]+"
        if re.match(github_url_pattern, target_directory):
            log("MAS", LogLevel.INFO, "🔄 检测到GitHub URL，正在克隆仓库...")
            temp_dir = tempfile.mkdtemp()
            try:
                subprocess.run(["git", "clone", target_directory, temp_dir], check=True)
                target_directory = temp_dir
            except subprocess.CalledProcessError as e:
                return {"status": "error", "message": f"克隆GitHub仓库失败: {e}"}

        # 验证目录
        if not os.path.isdir(target_directory):
            return {"status": "error", "message": "目录不存在"}

        # 选取待分析文件（支持多种编程语言）
        supported_extensions = ('.py', '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', 
                                '.js', '.ts', '.java', '.go', '.rs', '.rb', '.php')
        
        # 收集所有符合条件的源文件
        all_source_files = []
        for root, dirs, files in os.walk(target_directory):
            # 跳过常见的非代码目录
            dirs[:] = [d for d in dirs if d not in ('.git', '.svn', 'node_modules', 
                                                     '__pycache__', 'venv', 'env', 
                                                     '.venv', 'dist', 'build', '.idea', '.vscode')]
            for f in files:
                if f.endswith(supported_extensions):
                    file_path = os.path.join(root, f)
                    all_source_files.append(file_path)
        
        if not all_source_files:
            return {"status": "empty", "message": f"未发现支持的源代码文件（支持: {', '.join(supported_extensions)}）"}
        
        # 智能分组策略：
        # 1. 如果文件数量 <= 10，每个文件作为一个独立的 requirement
        # 2. 如果文件数量 > 10，按子目录分组，每个目录作为一个 requirement
        max_files_per_requirement = 10
        max_requirements = 20  # 限制总 requirement 数量避免过载
        
        if len(all_source_files) <= max_files_per_requirement:
            # 策略1：每个文件独立分析
            file_groups = [(os.path.basename(fp), [fp]) for fp in all_source_files]
            log("MAS", LogLevel.INFO, f"[AgentIntegration] 文件数量较少({len(all_source_files)})，采用单文件分析模式")
        else:
            # 策略2：按子目录分组
            from collections import defaultdict
            dir_groups = defaultdict(list)
            for fp in all_source_files:
                rel_path = os.path.relpath(fp, target_directory)
                # 使用第一级子目录作为分组键
                parts = rel_path.split(os.sep)
                group_key = parts[0] if len(parts) > 1 else 'root'
                dir_groups[group_key].append(fp)
            
            file_groups = list(dir_groups.items())
            log("MAS", LogLevel.INFO, f"[AgentIntegration] 文件数量较多({len(all_source_files)})，采用目录分组分析模式，发现 {len(file_groups)} 个分组")
        
        # 限制首批分析数量，避免长阻塞
        if len(file_groups) > max_requirements:
            file_groups = file_groups[:max_requirements]
            log("MAS", LogLevel.INFO, f"[AgentIntegration] 分组数量超过限制，本次分析前 {max_requirements} 个分组")

        run_id = str(uuid.uuid4())
        # 预读文件 + 分配 requirement_id
        prepared = []  # list of (rid, group_name, files_dict)
        for group_name, file_list in file_groups:
            files_content = {}
            for file_path in file_list:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                        files_content[file_path] = fh.read()
                except Exception as e:
                    log("MAS", LogLevel.WARNING, f"[AgentIntegration] 读取文件失败 {file_path}: {e}")
                    files_content[file_path] = ""
            
            if files_content:  # 只添加成功读取的文件
                self.requirement_counter += 1
                rid = self.requirement_counter
                prepared.append((rid, group_name, files_content))
        
        if not prepared:
            return {"status": "empty", "message": "未能成功读取任何源代码文件"}
            
        requirement_ids = [rid for rid, _, _ in prepared]

        # 先发送 run_init
        if 'summary' in self.agents:
            await self.agents['summary'].dispatch_message(
                receiver='summary_agent',
                content={'run_id': run_id, 'requirement_ids': requirement_ids, 'target_directory': target_directory},
                message_type='run_init'
            )
            log("MAS", LogLevel.INFO, f"[AgentIntegration] run_init sent run_id={run_id} requirements={len(requirement_ids)}")

        # 再派发具体分析任务
        dispatched = []
        for rid, group_name, files_content in prepared:
            # 合并所有文件内容，用于分析
            merged_content = "\n\n".join([
                f"/* File: {os.path.basename(fp)} */\n{content}"
                for fp, content in files_content.items()
            ])
            
            # 主文件路径（用于报告命名）
            primary_file = list(files_content.keys())[0] if files_content else group_name
            try:
                rel_path = os.path.relpath(primary_file, target_directory)
            except Exception:
                rel_path = group_name
            
            common_payload = {
                'requirement_id': rid,
                'code_content': merged_content,
                'code_directory': target_directory,
                'file_path': primary_file,
                'run_id': run_id,
                'readable_file': rel_path,
                'files_in_group': list(files_content.keys()),  # 包含的所有文件列表
                'group_name': group_name  # 分组名称（目录名或文件名）
            }
            
            # 收集当前分组的所有派发任务
            dispatch_tasks = []
            
            # 静态扫描
            if 'static_scan' in self.agents:
                dispatch_tasks.append(self.agents['static_scan'].dispatch_message(
                    receiver='static_scan_agent',
                    content=common_payload,
                    message_type='static_scan_request'
                ))
            # 代码质量
            if 'ai_code_quality' in self.agents:
                dispatch_tasks.append(self.agents['ai_code_quality'].dispatch_message(
                    receiver='ai_code_quality_agent',
                    content=common_payload,
                    message_type='quality_analysis_request'
                ))
            # 安全
            if 'ai_security' in self.agents:
                dispatch_tasks.append(self.agents['ai_security'].dispatch_message(
                    receiver='ai_security_agent',
                    content=common_payload,
                    message_type='security_analysis_request'
                ))
            # 性能
            if 'ai_performance' in self.agents:
                dispatch_tasks.append(self.agents['ai_performance'].dispatch_message(
                    receiver='ai_performance_agent',
                    content=common_payload,
                    message_type='performance_analysis_request'
                ))
            
            # 并发派发当前分组的所有任务，并等待所有任务完成入队
            if dispatch_tasks:
                await asyncio.gather(*dispatch_tasks)
                log("MAS", LogLevel.INFO, f"[AgentIntegration] 分组 '{group_name}' (rid={rid}) 的 {len(dispatch_tasks)} 个任务已派发")
            
            dispatched.append({'requirement_id': rid, 'group_name': group_name, 'file': primary_file, 'readable_file': rel_path})

        # 生成初始派发摘要报告(命名为dispatch) - 放在run_id对应的文件夹下
        try:
            from datetime import datetime
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_content = {
                'status': 'dispatched',
                'run_id': run_id,
                'dispatched_file_count': len(dispatched),
                'target_directory': target_directory,
                'tasks': dispatched
            }
            dispatch_filename = f'dispatch_report_{ts}_{run_id}.json'
            report_path = report_manager.generate_run_scoped_report(run_id, report_content, dispatch_filename)
        except Exception as e:
            return {"status": "error", "message": f"报告生成失败: {e}"}
        return {"status": "dispatched", "files": dispatched, "total_files": len(dispatched), "report_path": str(report_path), "run_id": run_id}

    async def wait_for_run_completion(self, run_id: str, timeout: float = 60.0, poll_interval: float = 1.0) -> Dict[str, Any]:
        """等待指定 run_id 的运行级综合报告生成。
        返回: {status: 'completed'|'timeout', 'summary_report': path or None, 'consolidated_reports': [...]}"""
        reports_dir = Path(__file__).parent.parent / 'reports' / 'analysis' / run_id
        end_time = asyncio.get_event_loop().time() + timeout
        summary_path = None
        consolidated = set()
        pattern_summary = re.compile(rf"run_summary_.*_{re.escape(run_id)}\.json$")
        pattern_consolidated = re.compile(rf"consolidated_req_\d+_{re.escape(run_id)}_.*\.json$")
        while asyncio.get_event_loop().time() < end_time:
            if reports_dir.exists():
                for f in reports_dir.rglob('*.json'):  # 递归查找所有JSON文件
                    name = f.name
                    if summary_path is None and pattern_summary.match(name):
                        summary_path = f
                    if pattern_consolidated.match(name):
                        consolidated.add(str(f))
                if summary_path:
                    return {
                        'status': 'completed',
                        'summary_report': str(summary_path),
                        'consolidated_reports': sorted(consolidated)
                    }
            await asyncio.sleep(poll_interval)
        return {
            'status': 'timeout',
            'summary_report': str(summary_path) if summary_path else None,
            'consolidated_reports': sorted(consolidated)
        }

def get_agent_integration_system() -> AgentIntegration:
    """获取智能体集成系统实例(单例)"""
    return AgentIntegration()