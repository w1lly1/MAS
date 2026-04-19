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
from utils.prompt_budgeting import budget_text_segments, prepare_generation_prompt, semantic_truncate_text, resolve_model_max_tokens
from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.database.weaviate.service import WeaviateVectorService
from infrastructure.database.vector_sync import (
    IssuePatternSyncService,
    DefaultKnowledgeEncodingAgent,
)
from utils import log, LogLevel, log_table, tabulate_grouped_items, format_table

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
        self.model = None
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

        # 数据层组件
        self.db_service = DatabaseService()
        self.vector_service = WeaviateVectorService(embed_fn=self._default_embed)
        self.encoding_agent = DefaultKnowledgeEncodingAgent(embed_fn=self._default_embed)
        self.sync_service = IssuePatternSyncService(
            db_service=self.db_service,
            vector_service=self.vector_service,
            agent=self.encoding_agent,
        )

        # 尝试自动连接 Weaviate（优雅降级：连接失败时仅使用 SQLite）
        self._init_weaviate_connection()

    async def initialize_data_manage(self, agent_integration=None):
        """初始化AI模型和代理集成"""
        try:
            self.agent_integration = agent_integration
            if self.model and self.tokenizer:
                self.ai_enabled = True
                log("db_manage_agent", LogLevel.INFO, "🔗 使用共享模型完成初始化")
                return True
            await self._initialize_ai_models()
            return True
        except Exception as e:
            log("db_manage_agent", LogLevel.ERROR, f"AI数据库管理代理初始化错误: {e}")
            return False

    def _init_weaviate_connection(self) -> None:
        """
        尝试连接 Weaviate 向量数据库。
        
        连接失败时优雅降级，仅使用 SQLite 存储。
        """
        try:
            connected = self.vector_service.connect(auto_create_schema=True)
            if connected:
                log(
                    "db_manage_agent",
                    LogLevel.INFO,
                    "✅ Weaviate 向量数据库已连接，支持语义搜索和去重",
                )
            else:
                status = self.vector_service.get_connection_status()
                log(
                    "db_manage_agent",
                    LogLevel.WARNING,
                    f"⚠️ Weaviate 连接失败，将仅使用 SQLite 存储: {status.get('error', '未知错误')}",
                )
        except Exception as e:
            log(
                "db_manage_agent",
                LogLevel.WARNING,
                f"⚠️ Weaviate 初始化异常，将仅使用 SQLite 存储: {e}",
            )

    def set_shared_model(self, model, tokenizer, model_name: Optional[str] = None):
        """注入共享模型与Tokenizer，避免重复加载"""
        if model is None or tokenizer is None:
            return
        self.model = model
        self.tokenizer = tokenizer
        if model_name:
            self.model_name = model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.ai_enabled = True
        log("db_manage_agent", LogLevel.INFO, "✅ 已注入共享模型/Tokenizer")

    async def stop(self):
        """停止智能体并清理资源"""
        # 关闭 Weaviate 连接
        if self.vector_service and self.vector_service.is_connected():
            try:
                self.vector_service.disconnect()
                log("db_manage_agent", LogLevel.INFO, "🔌 Weaviate 连接已关闭")
            except Exception as e:
                log("db_manage_agent", LogLevel.WARNING, f"⚠️ 关闭 Weaviate 连接时出错: {e}")
        
        # 调用父类的 stop 方法
        await super().stop()
    
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

    async def receive_message(self, message: Message):
        """
        覆写接收入口：在入队前做一次“场景分流 + 入参规范化”。

        场景：
        - user_requirement: 期望 content = {"requirement": {...}, "session_id": "..."}
          为了便于调用方，允许直接传 {"db_tasks":[...], "session_id":"..."}，这里会自动包装到 requirement。
        - knowledge_request: 期望 content = {"scan_results": {...}}
        """
        try:
            if message.message_type == "user_requirement":
                if isinstance(message.content, dict) and "requirement" not in message.content:
                    # 允许直接把 requirement 作为 content 传入
                    session_id = message.content.get("session_id", "default")
                    message.content = {
                        "requirement": {k: v for k, v in message.content.items() if k != "session_id"},
                        "session_id": session_id,
                    }
            elif message.message_type == "knowledge_request":
                # 允许 content 直接就是 scan_results
                if isinstance(message.content, dict) and "scan_results" not in message.content:
                    message.content = {"scan_results": message.content}
            else:
                log(
                    "db_manage_agent",
                    LogLevel.WARNING,
                    f"⚠️ 收到未支持的消息类型 {message.message_type}，将忽略该消息",
                )
                return
        except Exception as e:
            log("db_manage_agent", LogLevel.ERROR, f"❌ 规范化消息失败: {e}")
            return

        await super().receive_message(message)

    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体任务 - 实现BaseAgent的抽象方法"""
        # 暂时返回空结果
        return {"status": "success", "message": "Task executed successfully"}

    # === 对外接口方法 ===

    async def user_requirement_interpret(
        self,
        user_requirement: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        用户需求解析接口
        - 入参 user_requirement 中包含自然语言 db_tasks
        - 通过 Qwen + 特定 Prompt 翻译为结构化 DB 操作后执行
        """
        log("db_manage_agent", LogLevel.INFO, f"📝 开始解析用户需求，会话ID: {session_id}")
        
        raw_text = self._extract_raw_text(user_requirement)
        tasks = self._normalize_db_tasks(user_requirement)
        forced_tasks = self._extract_forced_tasks(user_requirement)
        mode = "write"
        if forced_tasks:
            tasks = forced_tasks
            mode = self._infer_mode_from_tasks(tasks, session_id) or "delete"
            log("db_manage_agent", LogLevel.INFO, f"🧭 使用强制任务 mode={mode}")
        else:
            if raw_text:
                mode = await self._classify_db_mode(raw_text)
            log("db_manage_agent", LogLevel.INFO, f"🧭 识别数据库意图 mode={mode}")
        # 稳定性优先：两阶段生成（先 issue_pattern，再 session+issue）
        if tasks and mode == "write" and not forced_tasks:
            issue_pattern_tasks = await self._translate_tasks_with_llm(
                tasks, session_id=session_id, variant="issue_pattern", raw_text=raw_text
            )
            issue_pattern_tasks = self._filter_tasks_by_targets(
                issue_pattern_tasks, {"issue_pattern", "issue_patterns"}
            )
            issue_pattern_task = issue_pattern_tasks[0] if issue_pattern_tasks else None
            if issue_pattern_tasks:
                self._log_llm_tasks("阶段1(issue_pattern)", issue_pattern_tasks)

            session_issue_tasks = await self._translate_tasks_with_llm(
                tasks,
                session_id=session_id,
                variant="session_issue",
                raw_text=raw_text,
                issue_pattern=issue_pattern_task,
            )
            session_issue_tasks = self._filter_tasks_by_targets(
                session_issue_tasks, {"review_session", "review_sessions", "curated_issue", "curated_issues"}
            )
            if session_issue_tasks:
                self._log_llm_tasks("阶段2(session_issue)", session_issue_tasks)

            combined = []
            if issue_pattern_tasks:
                combined.extend(issue_pattern_tasks)
            if session_issue_tasks:
                combined.extend(session_issue_tasks)
            combined = self._dedupe_tasks(combined)

            if combined:
                tasks = combined
                try:
                    pretty_tasks = json.dumps(tasks, ensure_ascii=False, indent=2)
                except Exception:
                    pretty_tasks = str(tasks)
                log("db_manage_agent", LogLevel.INFO, f"📜 LLM 解析后的结构化任务:\n{pretty_tasks}")
            else:
                log("db_manage_agent", LogLevel.WARNING, "⚠️ LLM 未返回可解析的 db_tasks 结构，放弃执行")
        elif tasks and mode in ("query", "delete") and not forced_tasks:
            tasks = await self._translate_tasks_with_llm(
                tasks, session_id=session_id, variant="read_delete", raw_text=raw_text
            )
            tasks = self._dedupe_tasks(tasks)
            if tasks:
                try:
                    pretty_tasks = json.dumps(tasks, ensure_ascii=False, indent=2)
                except Exception:
                    pretty_tasks = str(tasks)
                log("db_manage_agent", LogLevel.INFO, f"📜 LLM 解析后的结构化任务:\n{pretty_tasks}")
            else:
                log("db_manage_agent", LogLevel.WARNING, "⚠️ LLM 未返回可解析的 db_tasks 结构，放弃执行")
            if mode == "query" and not tasks:
                tasks = [
                    {"target": "review_session", "action": "query", "data": {}},
                    {"target": "curated_issue", "action": "query", "data": {}},
                    {"target": "issue_pattern", "action": "query", "data": {}},
                ]

        inferred_mode = self._infer_mode_from_tasks(tasks, session_id)
        if inferred_mode in ("query", "delete") and inferred_mode != mode:
            log(
                "db_manage_agent",
                LogLevel.INFO,
                f"🧭 基于结构化任务推断 mode={inferred_mode}，覆盖原始 mode={mode}",
            )
            mode = inferred_mode
            if mode == "query":
                tasks = self._filter_tasks_by_actions(tasks, {"query"}, session_id)
                if not tasks:
                    tasks = [
                        {"target": "review_session", "action": "query", "data": {}},
                        {"target": "curated_issue", "action": "query", "data": {}},
                        {"target": "issue_pattern", "action": "query", "data": {}},
                    ]
            else:
                tasks = self._filter_tasks_by_actions(tasks, {"delete", "delete_all"}, session_id)

        if mode == "write":
            tasks = self._ensure_three_table_tasks(tasks, raw_text, session_id)
            if not forced_tasks:
                tasks = self._apply_non_llm_defaults(tasks, raw_text, session_id)

        if self._requires_delete_all_confirm(tasks, session_id):
            log(
                "db_manage_agent",
                LogLevel.WARNING,
                "⚠️ 检测到 delete_all 但缺少 confirm=true，已阻止执行",
            )
            return {
                "status": "need_confirm",
                "session_id": session_id,
                "results": [],
                "pending_action": "delete_all",
                "pending_tasks": tasks,
                "message": "删除全部数据需要明确确认，请确认后再执行该操作。",
            }

        if not tasks:
            return {
                "status": "noop",
                "session_id": session_id,
                "results": [],
                "message": "未识别到可执行的数据库任务",
            }

        # 展开 target="all" 的任务为三张表的任务
        tasks = self._expand_all_target_tasks(tasks)

        # 按特定顺序执行：先 issue_pattern，再 review_session，最后 curated_issue
        # 这样可以确保 curated_issue 写入时能关联到刚创建的 pattern_id 和 session_db_id
        ordered_tasks = self._order_tasks_for_execution(tasks)
        created_pattern_id = None
        created_session_db_id = None

        results: List[Dict[str, Any]] = []
        for task in ordered_tasks:
            try:
                # 在执行 curated_issue 前，自动回填已创建的 pattern_id 和 session_db_id
                target = str(task.get("target") or task.get("table") or "").lower()
                target_normalized = {
                    "curated_issues": "curated_issue", "curated_issue": "curated_issue",
                    "issue": "curated_issue",
                }.get(target, target)
                if target_normalized == "curated_issue" and isinstance(task.get("data"), dict):
                    if created_pattern_id and not task["data"].get("pattern_id"):
                        task["data"]["pattern_id"] = created_pattern_id
                    if created_session_db_id and not task["data"].get("session_id"):
                        task["data"]["session_id"] = created_session_db_id

                result = await self._handle_single_db_task(task, session_id)
                results.append({"task": task, "status": "success", "result": result})

                # 收集已创建的 pattern_id 和 session_db_id，供后续 curated_issue 关联
                if isinstance(result, dict):
                    if result.get("pattern_id") and not created_pattern_id:
                        created_pattern_id = result["pattern_id"]
                    if result.get("session_db_id") and not created_session_db_id:
                        created_session_db_id = result["session_db_id"]
            except Exception as e:
                log("db_manage_agent", LogLevel.ERROR, f"❌ 处理数据库任务失败: {e}")
                results.append({"task": task, "status": "failed", "error": str(e)})

        overall_status = (
            "success"
            if all(item["status"] == "success" for item in results)
            else "partial"
        )
        summary_table = ""
        if any(
            self._normalize_task(
                str(item.get("task", {}).get("action", "")),
                str(item.get("task", {}).get("target", "")),
                item.get("task", {}).get("data", {}) if isinstance(item.get("task", {}).get("data"), dict) else {},
                session_id,
            )[0]
            != "query"
            for item in results
        ):
            summary_table = self._build_db_summary_table(results)
            if summary_table:
                # use log_table so multi-line tables get a timestamped prefix per line
                # print the table raw (no date/level/agent prefix) to preserve alignment
                log_table("db_manage_agent", LogLevel.INFO, summary_table, title="📋 数据库写入摘要:", use_logger_prefix=False)

        # 使用 tabulate 优化终端展示查询结果
        query_tables = self._build_db_query_tables_tabulate(results)
        if query_tables:
            for table_name, table_text in query_tables.items():
                # print the query table raw so borders align correctly
                log_table("db_manage_agent", LogLevel.INFO, table_text, title=f"🔎 数据库查询结果 ({table_name}):", use_logger_prefix=False)

        return {
            "status": overall_status,
            "session_id": session_id,
            "results": results,
            "message": "数据库任务执行完成",
            "summary_table": summary_table,
            "query_tables": query_tables,
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

    # ================== 内部辅助方法 ================== #
    def _normalize_db_tasks(
        self,
        user_requirement: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        仅从 user_requirement 提取自然语言 db_tasks。
        """
        if isinstance(user_requirement, dict):
            raw_text = user_requirement.get("raw_text")
            if isinstance(raw_text, str) and raw_text.strip():
                return [{"description": raw_text.strip()}]
        return []

    def _extract_raw_text(self, user_requirement: Optional[Dict[str, Any]]) -> str:
        if isinstance(user_requirement, dict):
            raw_text = user_requirement.get("raw_text")
            if isinstance(raw_text, str) and raw_text.strip():
                return raw_text.strip()
            tasks = user_requirement.get("db_tasks") or []
            if isinstance(tasks, list) and tasks:
                desc = tasks[0].get("description") if isinstance(tasks[0], dict) else ""
                if isinstance(desc, str) and desc.strip():
                    return desc.strip()
        return ""

    def _extract_forced_tasks(self, user_requirement: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(user_requirement, dict):
            return []
        forced = user_requirement.get("forced_tasks")
        if not forced:
            return []
        if not isinstance(forced, list):
            forced = [forced]
        return [task for task in forced if isinstance(task, dict)]

    def _filter_tasks_by_targets(
        self, tasks: List[Dict[str, Any]], targets: set
    ) -> List[Dict[str, Any]]:
        filtered = []
        for task in tasks or []:
            if not isinstance(task, dict):
                continue
            target = str(task.get("target") or task.get("table") or "").lower()
            if target in targets:
                filtered.append(task)
        return filtered

    def _dedupe_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for task in tasks or []:
            if not isinstance(task, dict):
                continue
            key = json.dumps(task, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(task)
        return deduped

    def _requires_delete_all_confirm(
        self, tasks: List[Dict[str, Any]], session_id: str
    ) -> bool:
        for task in tasks or []:
            if not isinstance(task, dict):
                continue
            action = str(task.get("action") or task.get("op") or task.get("type") or "").lower()
            target = str(task.get("target") or task.get("table") or task.get("object") or "").lower()
            data = task.get("data") if isinstance(task.get("data"), dict) else {}
            normalized_action, _, normalized_data = self._normalize_task(
                action, target, data, session_id
            )
            if normalized_action == "delete_all" and normalized_data.get("confirm") is not True:
                return True
            if normalized_action == "delete" and self._is_delete_all_candidate(
                normalized_data, target
            ):
                return True
        return False

    def _is_delete_all_candidate(self, data: Dict[str, Any], target: str) -> bool:
        if not data:
            return True
        id_keys = {"id", "pattern_id", "issue_id", "session_db_id"}
        if any(data.get(k) for k in id_keys):
            return False
        # delete 操作目前不支持按字段过滤，缺少 id 等同于全删意图
        return True

    def _infer_mode_from_tasks(self, tasks: List[Dict[str, Any]], session_id: str) -> str:
        if not tasks:
            return "write"
        has_query = False
        has_delete = False
        for task in tasks:
            if not isinstance(task, dict):
                continue
            action = str(task.get("action") or task.get("op") or task.get("type") or "").lower()
            target = str(task.get("target") or task.get("table") or task.get("object") or "").lower()
            data = task.get("data") if isinstance(task.get("data"), dict) else {}
            normalized_action, _, _ = self._normalize_task(action, target, data, session_id)
            if normalized_action in ("delete", "delete_all"):
                has_delete = True
            elif normalized_action == "query":
                has_query = True
        if has_delete:
            return "delete"
        if has_query:
            return "query"
        return "write"

    def _filter_tasks_by_actions(
        self,
        tasks: List[Dict[str, Any]],
        allowed_actions: set,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for task in tasks or []:
            if not isinstance(task, dict):
                continue
            action = str(task.get("action") or task.get("op") or task.get("type") or "").lower()
            target = str(task.get("target") or task.get("table") or task.get("object") or "").lower()
            data = task.get("data") if isinstance(task.get("data"), dict) else {}
            normalized_action, _, _ = self._normalize_task(action, target, data, session_id)
            if normalized_action in allowed_actions:
                filtered.append(task)
        return filtered

    def _ensure_three_table_tasks(
        self, tasks: List[Dict[str, Any]], raw_text: str, session_id: str
    ) -> List[Dict[str, Any]]:
        """确保每次都包含 review_session / curated_issue / issue_pattern 三类任务。"""
        normalized = []
        seen = set()
        target_map = {
            "review_sessions": "review_session",
            "review_session": "review_session",
            "curated_issues": "curated_issue",
            "curated_issue": "curated_issue",
            "issue_patterns": "issue_pattern",
            "issue_pattern": "issue_pattern",
        }
        for task in tasks or []:
            if not isinstance(task, dict):
                continue
            target = str(task.get("target") or task.get("table") or "").lower()
            target = target_map.get(target, target)
            if target:
                seen.add(target)
            normalized.append(task)

        def _default_review_session():
            return {
                "target": "review_session",
                "action": "upsert",
                "data": {
                    "session_id": session_id,
                    "user_message": raw_text or "",
                    "code_directory": "",
                    "status": "open",
                    "code_patch": "",
                    "git_commit": "",
                },
            }

        def _default_curated_issue():
            return {
                "target": "curated_issue",
                "action": "upsert",
                "data": {
                    "session_id": session_id,
                    "pattern_id": "",
                    "project_path": "",
                    "file_path": "",
                    "start_line": 0,
                    "end_line": 0,
                    "code_snippet": "",
                    "problem_phenomenon": raw_text or "",
                    "root_cause": "",
                    "solution": "",
                    "severity": "medium",
                    "status": "open",
                },
            }

        def _default_issue_pattern():
            return {
                "target": "issue_pattern",
                "action": "upsert",
                "data": {
                    "error_type": "general",
                    "severity": "medium",
                    "language": "",
                    "framework": "",
                    "error_description": raw_text or "",
                    "problematic_pattern": raw_text or "",
                    "solution": "",
                    "file_pattern": "",
                    "class_pattern": "",
                    "tags": "",
                    "status": "active",
                },
            }

        if "review_session" not in seen:
            normalized.append(_default_review_session())
        if "curated_issue" not in seen:
            normalized.append(_default_curated_issue())
        if "issue_pattern" not in seen:
            normalized.append(_default_issue_pattern())
        return normalized

    def _apply_non_llm_defaults(
        self, tasks: List[Dict[str, Any]], raw_text: str, session_id: str
    ) -> List[Dict[str, Any]]:
        """填充系统自管理字段（不依赖大模型推断），覆盖/补齐写入数据。

        系统自管理字段规则（参考 todoList.txt 定义）：
        - ReviewSession: id(自增), session_id(系统), user_message(系统), status(系统,默认open), created_at, updated_at
        - CuratedIssue: id(自增), session_id(系统), pattern_id(系统), severity(系统,默认medium), status(系统,默认open), created_at, updated_at
        - IssuePattern: id(自增), severity(系统,默认medium), status(系统,默认active), created_at, updated_at
        """
        if not tasks:
            return tasks
        for task in tasks:
            if not isinstance(task, dict):
                continue
            action = str(task.get("action") or task.get("op") or task.get("type") or "").lower()
            if action in ("query", "delete", "delete_all"):
                continue
            target = str(task.get("target") or task.get("table") or "").lower()
            data = task.get("data") if isinstance(task.get("data"), dict) else {}

            # --- 移除 LLM 不应输出的系统自管理字段（防止 LLM 错误覆盖） ---
            for sys_field in ("id", "created_at", "updated_at"):
                data.pop(sys_field, None)

            if target in ("review_session", "review_sessions", "session"):
                # 系统自管理字段：session_id, user_message, status
                data["session_id"] = session_id
                data["user_message"] = raw_text or ""
                data["status"] = data.get("status", "open")
                # LLM 推断字段兜底
                data.setdefault("code_directory", "")
                data.setdefault("code_patch", "")
                data.setdefault("git_commit", "")
                task["data"] = data
                continue

            if target in ("curated_issue", "curated_issues", "issue"):
                # 系统自管理字段：session_id, pattern_id, severity, status
                data["session_id"] = session_id
                # pattern_id 将在执行阶段自动关联（见 _link_pattern_id_to_curated_issues）
                data.setdefault("pattern_id", "")
                data["severity"] = data.get("severity", "medium")
                data["status"] = data.get("status", "open")
                # LLM 推断字段兜底
                data.setdefault("project_path", "")
                data.setdefault("file_path", "")
                data.setdefault("start_line", 0)
                data.setdefault("end_line", 0)
                data.setdefault("code_snippet", "")
                data.setdefault("problem_phenomenon", raw_text or "")
                data.setdefault("root_cause", "")
                data.setdefault("solution", "")
                task["data"] = data
                continue

            if target in ("issue_pattern", "issue_patterns", "issuepattern", "pattern"):
                # 系统自管理字段：severity, status
                data["severity"] = data.get("severity", "medium")
                data["status"] = data.get("status", "active")
                # LLM 推断字段兜底
                data.setdefault("error_type", "general")
                data.setdefault("error_description", raw_text or "")
                data.setdefault("problematic_pattern", raw_text or "")
                data.setdefault("solution", "")
                data.setdefault("title", "")
                data.setdefault("language", "")
                data.setdefault("framework", "")
                data.setdefault("file_pattern", "")
                data.setdefault("class_pattern", "")
                data.setdefault("tags", "")
                task["data"] = data
                continue
        return tasks

    def _expand_all_target_tasks(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """展开 target="all" 的任务为三张表的任务。
        
        当 LLM 输出 {"target": "all", "action": "delete_all", ...} 时，
        自动展开为三条任务，分别针对 issue_pattern、review_session、curated_issue。
        """
        expanded: List[Dict[str, Any]] = []
        all_tables = ["issue_pattern", "review_session", "curated_issue"]
        
        for task in tasks or []:
            if not isinstance(task, dict):
                continue
            target = str(task.get("target") or task.get("table") or "").lower()
            
            if target in ("all", "*", "all_tables"):
                # 展开为三张表的任务
                action = task.get("action", "")
                data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
                
                log(
                    "db_manage_agent",
                    LogLevel.INFO,
                    f"🔄 展开 target=all 为三张表的任务 action={action}",
                )
                
                for table in all_tables:
                    expanded.append({
                        "target": table,
                        "action": action,
                        "data": dict(data),  # 复制 data，避免共享引用
                    })
            else:
                expanded.append(task)
        
        return expanded

    def _order_tasks_for_execution(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """按执行优先级排序任务：issue_pattern > review_session > curated_issue。
        
        这样可以确保：
        1. issue_pattern 先创建，获得 pattern_id
        2. review_session 创建，获得 session_db_id
        3. curated_issue 最后创建，可以关联 pattern_id 和 session_db_id
        """
        target_priority = {
            "issue_pattern": 0, "issue_patterns": 0, "issuepattern": 0, "pattern": 0,
            "review_session": 1, "review_sessions": 1, "session": 1,
            "curated_issue": 2, "curated_issues": 2, "issue": 2,
        }

        def _sort_key(task: Dict[str, Any]) -> int:
            if not isinstance(task, dict):
                return 99
            target = str(task.get("target") or task.get("table") or "").lower()
            return target_priority.get(target, 50)

        return sorted(tasks, key=_sort_key)

    def _sanitize_json_like_text(self, text: str) -> str:
        """清理 LLM 输出中的噪声，尽量提取 JSON."""
        if not isinstance(text, str):
            return ""
        cleaned = text.strip()
        # 去除 ```json / ``` 包裹
        cleaned = re.sub(r"```json", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "")
        # 移除注释
        cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        # 移除尾随逗号
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
        return cleaned.strip()

    async def _translate_tasks_with_llm(
        self,
        raw_tasks: List[Dict[str, Any]],
        session_id: str,
        variant: Optional[str] = None,
        raw_text: Optional[str] = None,
        issue_pattern: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 Qwen 将自然语言 db_tasks 翻译为结构化 DB 操作。
        输入示例（prompts.py 106-112）：
        {
          "db_tasks": [
            {"project": "...", "description": "数据库操作的自然语言描述"}
          ]
        }

        期望输出 JSON 数组元素：
        {
          "target": "issue_pattern|curated_issue|review_session",
          "action": "create|update|delete|sync",
          "data": { ... 与 models.py 字段对齐 ... }
        }
        """
        if not self.ai_enabled or not self.tokenizer or not hasattr(self, "model"):
            log("db_manage_agent", LogLevel.WARNING, "⚠️ AI 未启用，跳过 LLM 翻译")
            return []

        # 使用独立 Prompt 指令化翻译行为
        try:
            from infrastructure.config.prompts import get_prompt
            system_prompt = get_prompt(
                "db_task_translation",
                model_name=self.model_name,
                variant=variant,
            )
            log("db_manage_agent", LogLevel.INFO, f"✅ 成功加载 prompt variant={variant}, 长度={len(system_prompt)}")
        except Exception as e:
            log("db_manage_agent", LogLevel.WARNING, f"⚠️ Prompt 加载失败，使用兜底 prompt: {e}")
            system_prompt = (
                "你是数据库管理代理。根据 db_tasks 的 project 和 description，"
                "将需求翻译为 SQLite 的结构化操作，字段对齐 review_sessions/curated_issues/issue_patterns。"
                "输出 JSON 数组，仅包含 target, action, data，禁止附加说明。"
            )

        payload = {"db_tasks": raw_tasks}
        if raw_text:
            payload["raw_text"] = raw_text
        if issue_pattern:
            payload["issue_pattern"] = issue_pattern
        try:
            user_content = json.dumps(payload, ensure_ascii=False)
        except Exception:
            user_content = str(payload)

        # 增强日志：打印实际接收的内容和发送给模型的内容
        log("db_manage_agent", LogLevel.INFO, f"📥 DB Agent 接收到的 db_tasks: {user_content[:500]}")
        log("db_manage_agent", LogLevel.DEBUG, f"🧠 System Prompt (前300字): {system_prompt[:300]}")


        # 使用 chat template 明确角色（与 user_comm_agent 保持一致）
        if hasattr(self.tokenizer, "apply_chat_template") and self.model_name.startswith("Qwen/"):
            base_max = resolve_model_max_tokens(self.tokenizer, fallback=4096)
            budgets = budget_text_segments(
                self.tokenizer,
                [system_prompt, user_content],
                max_tokens=max(512, base_max - 96),
            )
            if budgets:
                system_prompt = budgets[0]
            if len(budgets) > 1:
                user_content = budgets[1]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if self.used_device == "gpu":
                input_ids = input_ids.to("cuda")

            log("db_manage_agent", LogLevel.INFO, "⏳ 正在调用模型生成翻译结果，请稍候...")

            effective_max_new_tokens = 512
            _, _, effective_max_new_tokens = prepare_generation_prompt(
                self.tokenizer,
                f"{system_prompt}\n\n{user_content}",
                max_new_tokens=effective_max_new_tokens,
                fallback_model_max=base_max,
                safety_margin=64,
            )
            
            # 把同步阻塞的 model.generate 放到线程池执行，避免阻塞事件循环
            def _generate_sync():
                with torch.no_grad():
                    return self.model.generate(
                        input_ids,
                        max_new_tokens=effective_max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            
            try:
                outputs = await asyncio.to_thread(_generate_sync)
                generated = self.tokenizer.decode(
                    outputs[0][input_ids.shape[1]:], skip_special_tokens=True
                )
            except Exception as e:
                log("db_manage_agent", LogLevel.ERROR, f"❌ 模型生成失败: {e}")
                return []
        else:
            # 兜底：旧式拼接（保留兼容性）
            prompt = f"{system_prompt}\n\n用户输入：{user_content}\n\n请输出 JSON 数组："
            base_max = resolve_model_max_tokens(self.tokenizer, fallback=4096)
            prompt = semantic_truncate_text(self.tokenizer, prompt, max(512, base_max - 128))
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.used_device == "gpu":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            log("db_manage_agent", LogLevel.WARNING, "⏳ 正在调用模型生成翻译结果（兜底模式），请稍候...")
            _, _, effective_max_new_tokens = prepare_generation_prompt(
                self.tokenizer,
                prompt,
                max_new_tokens=512,
                fallback_model_max=base_max,
                safety_margin=64,
            )
            
            def _generate_sync():
                with torch.no_grad():
                    return self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=effective_max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            
            try:
                outputs = await asyncio.to_thread(_generate_sync)
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                log("db_manage_agent", LogLevel.ERROR, f"❌ 模型生成失败: {e}")
                return []

        parsed = self._extract_structured_tasks_from_text(generated)
        if parsed:
            variant_text = f" variant={variant}" if variant else ""
            log("db_manage_agent", LogLevel.INFO, f"✅ LLM 翻译得到 {len(parsed)} 条任务{variant_text}")
        else:
            # 输出原始 LLM 响应以便调试
            log("db_manage_agent", LogLevel.WARNING, f"⚠️ 未能解析 LLM 输出，返回空任务")
            # 同时用 INFO 级别输出，确保能看到
            log("db_manage_agent", LogLevel.INFO, f"📝 LLM 原始响应 (解析失败):\n{generated[:500]}..." if len(generated) > 500 else f"📝 LLM 原始响应 (解析失败):\n{generated}")
        return parsed

    async def _classify_db_mode(self, raw_text: str) -> str:
        """使用 LLM 判断用户请求属于 write/query/delete。"""
        if not self.ai_enabled or not self.tokenizer or not hasattr(self, "model"):
            return "write"
        try:
            from infrastructure.config.prompts import get_prompt
            system_prompt = get_prompt(
                "db_task_intent",
                model_name=self.model_name,
                variant="default",
            )
        except Exception:
            system_prompt = "判断用户请求属于 write/query/delete，仅输出 JSON: {\"mode\":\"write\"}"

        payload = {"raw_text": raw_text}
        try:
            user_content = json.dumps(payload, ensure_ascii=False)
        except Exception:
            user_content = str(payload)

        if hasattr(self.tokenizer, "apply_chat_template") and self.model_name.startswith("Qwen/"):
            base_max = resolve_model_max_tokens(self.tokenizer, fallback=4096)
            budgets = budget_text_segments(
                self.tokenizer,
                [system_prompt, user_content],
                max_tokens=max(256, base_max - 64),
            )
            if budgets:
                system_prompt = budgets[0]
            if len(budgets) > 1:
                user_content = budgets[1]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if self.used_device == "gpu":
                input_ids = input_ids.to("cuda")

            _, _, effective_max_new_tokens = prepare_generation_prompt(
                self.tokenizer,
                f"{system_prompt}\n\n{user_content}",
                max_new_tokens=64,
                fallback_model_max=base_max,
                safety_margin=32,
            )

            def _generate_sync():
                with torch.no_grad():
                    return self.model.generate(
                        input_ids,
                        max_new_tokens=effective_max_new_tokens,
                        temperature=0.2,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

            try:
                outputs = await asyncio.to_thread(_generate_sync)
                generated = self.tokenizer.decode(
                    outputs[0][input_ids.shape[1]:], skip_special_tokens=True
                )
            except Exception as e:
                log("db_manage_agent", LogLevel.WARNING, f"⚠️ 模式判断失败，回退 write: {e}")
                return "write"
        else:
            prompt = f"{system_prompt}\n\n用户输入：{user_content}\n\n只输出 JSON："
            base_max = resolve_model_max_tokens(self.tokenizer, fallback=4096)
            prompt = semantic_truncate_text(self.tokenizer, prompt, max(256, base_max - 128))
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.used_device == "gpu":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            _, _, effective_max_new_tokens = prepare_generation_prompt(
                self.tokenizer,
                prompt,
                max_new_tokens=64,
                fallback_model_max=base_max,
                safety_margin=32,
            )

            def _generate_sync():
                with torch.no_grad():
                    return self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=effective_max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

            try:
                outputs = await asyncio.to_thread(_generate_sync)
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                log("db_manage_agent", LogLevel.WARNING, f"⚠️ 模式判断失败，回退 write: {e}")
                return "write"

        cleaned = self._sanitize_json_like_text(generated)
        try:
            data = json.loads(cleaned)
            mode = str(data.get("mode", "")).lower()
            if mode in ("write", "query", "delete"):
                return mode
        except Exception:
            pass
        for key in ("write", "query", "delete"):
            if key in cleaned.lower():
                return key
        return "write"

    def _extract_structured_tasks_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        从模型输出中提取 JSON 数组；容错：截取首尾的 JSON 片段。
        """
        cleaned = self._sanitize_json_like_text(text)

        # 1) 直接解析
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, list) else ( [data] if isinstance(data, dict) else [])
        except Exception:
            pass

        # 2) 尝试修复常见问题：单引号 -> 双引号
        repaired = cleaned
        if "'" in repaired and '"' not in repaired:
            repaired = re.sub(r"'(.*?)'", r'"\1"', repaired)

        # 3) 逐字扫描第一个 JSON 数组片段（允许末尾缺失闭合括号）
        try:
            start = repaired.find("[")
            if start != -1:
                depth = 0
                for idx in range(start, len(repaired)):
                    ch = repaired[idx]
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            snippet = repaired[start : idx + 1]
                            try:
                                data = json.loads(snippet)
                                return data if isinstance(data, list) else []
                            except Exception:
                                break
                # 如果走到末尾仍未闭合，尝试补齐 ']' 并解析
                if depth > 0:
                    snippet = repaired[start:]
                    snippet += "]" * depth
                    try:
                        data = json.loads(snippet)
                        return data if isinstance(data, list) else []
                    except Exception:
                        pass

        except Exception:
            pass

        # 4) 逐字扫描 JSON 对象片段，尝试将单个对象包装为数组
        try:
            obj_start = repaired.find("{")
            if obj_start != -1:
                depth = 0
                for idx in range(obj_start, len(repaired)):
                    ch = repaired[idx]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            snippet = repaired[obj_start : idx + 1]
                            try:
                                data = json.loads(snippet)
                                return [data] if isinstance(data, dict) else []
                            except Exception:
                                break
                # 如果对象未闭合，尝试补齐 '}' 并解析
                if depth > 0:
                    snippet = repaired[obj_start:]
                    snippet += "}" * depth
                    try:
                        data = json.loads(snippet)
                        return [data] if isinstance(data, dict) else []
                    except Exception:
                        pass
        except Exception:
            pass

        # 5) 最后尝试宽松模式：替换单引号、删除多余换行和尾部省略号，再补齐括号
        try:
            loose = re.sub(r"\.\.\.\s*$", "", cleaned)
            loose = re.sub(r"\n+", " ", loose)
            if "'" in loose:
                loose = re.sub(r"'(.*?)'", r'"\1"', loose)

            # 补齐中括号/花括号
            open_sq = loose.count("[") - loose.count("]")
            open_cur = loose.count("{") - loose.count("}")
            if open_sq > 0:
                loose = loose + ("]" * open_sq)
            if open_cur > 0:
                loose = loose + ("}" * open_cur)

            data = json.loads(loose)
            return data if isinstance(data, list) else ( [data] if isinstance(data, dict) else [])
        except Exception:
            pass

        return []

    async def _handle_single_db_task(
        self, task: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        """
        执行单条数据库任务。
        预期字段：
            - action: create/update/delete/sync
            - target/table: review_session/curated_issue/issue_pattern
            - data: 具体字段
        """
        action = str(task.get("action") or task.get("op") or task.get("type") or "").lower()
        target = str(task.get("target") or task.get("table") or task.get("object") or "").lower()
        data = task.get("data") if isinstance(task.get("data"), dict) else {}
        action, target, data = self._normalize_task(action, target, data, session_id)
        if action or target:
            log(
                "db_manage_agent",
                LogLevel.INFO,
                f"🧩 执行数据库任务 target={target or '未提供'} action={action or '未提供'} keys={list(data.keys())}",
            )

        # 如果缺少 action/target，尝试根据字段推断（常见场景：新增 IssuePattern）
        if not action and task:
            if "error_type" in task:
                action = "create"
                target = target or "issue_pattern"
                data = {**task}
        if not target and "table" in task:
            target = str(task["table"]).lower()

        if target in ("issue_pattern", "issuepattern", "pattern"):
            return await self._handle_issue_pattern_task(action, data)
        if target in ("curated_issue", "curatedissue", "curated"):
            return await self._handle_curated_issue_task(action, data)
        if target in ("review_session", "reviewsession", "session"):
            return await self._handle_review_session_task(action, data, session_id)

        raise ValueError(
            f"未知的数据库任务目标: {target or '未提供'} (允许: issue_pattern/curated_issue/review_session)"
        )

    def _normalize_task(
        self,
        action: str,
        target: str,
        data: Dict[str, Any],
        session_id: str,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """规范化 LLM 输出的 target/action/data，避免表名/动作/字段不匹配。"""
        if action:
            action = action.strip().lower()
            # 兼容 SQL/自然语义动作
            if "select" in action or "query" in action or "read" in action:
                action = "query"
            elif "show" in action or "list" in action or "print" in action:
                action = "query"
            elif "delete" in action and "all" in action:
                action = "delete_all"
            elif "truncate" in action or "clear" in action:
                action = "delete_all"
            elif action.startswith("delete"):
                action = "delete"
        target_map = {
            "review_sessions": "review_session",
            "review_session": "review_session",
            "curated_issues": "curated_issue",
            "curated_issue": "curated_issue",
            "issue_patterns": "issue_pattern",
            "issue_pattern": "issue_pattern",
            "issuepattern": "issue_pattern",
            "pattern": "issue_pattern",
            "session": "review_session",
            "issue": "curated_issue",
        }
        action_map = {
            "insert": "create",
            "create": "create",
            "add": "create",
            "update": "update",
            "modify": "update",
            "upsert": "upsert",
            "create_or_update": "upsert",
            "delete": "delete",
            "remove": "delete",
            "delete_all": "delete_all",
            "truncate": "delete_all",
            "clear": "delete_all",
            "sync": "sync",
            "select": "query",
            "query": "query",
            "read": "query",
            "fetch": "query",
            "list": "query",
            "show": "query",
        }
        target = target_map.get(target, target)
        action = action_map.get(action, action)
        if action == "delete" and data.get("confirm") is True and data.get("scope") in ("all", "*"):
            action = "delete_all"
        if action not in ("create", "update", "delete", "sync", "upsert", "query", "delete_all"):
            action = "upsert"

        # 写入场景下，LLM 生成的 update 如果缺少 id，自动回退为 upsert
        # 因为 LLM 不负责管理主键 id，update 无法执行，应回退为 create/upsert
        if action == "update":
            # 根据 target 类型判断主键字段（区分主键和外键）
            if target == "issue_pattern":
                has_id = bool(data.get("id") or data.get("pattern_id"))
            elif target == "curated_issue":
                # 注意：pattern_id 和 session_id 是外键，不能作为更新的标识
                has_id = bool(data.get("id") or data.get("issue_id"))
            elif target == "review_session":
                has_id = bool(data.get("id") or data.get("session_db_id"))
            else:
                has_id = False
            
            if not has_id:
                log(
                    "db_manage_agent",
                    LogLevel.INFO,
                    f"🔄 LLM 输出 update 但缺少 id，自动回退为 upsert (target={target})",
                )
                action = "upsert"

        # 字段白名单，过滤 LLM 自造字段
        if target == "issue_pattern":
            allowed = {
                # LLM 推断字段
                "title",
                "error_type",
                "language",
                "framework",
                "error_description",
                "problematic_pattern",
                "solution",
                "file_pattern",
                "class_pattern",
                "tags",
                # 系统自管理字段（由 _apply_non_llm_defaults 填充）
                "severity",
                "status",
                # 操作控制字段
                "layers",
                "id",
                "pattern_id",
                "limit",
                "offset",
                "confirm",
                "scope",
            }
        elif target == "curated_issue":
            allowed = {
                "session_id",
                "pattern_id",
                "project_path",
                "file_path",
                "start_line",
                "end_line",
                "code_snippet",
                "problem_phenomenon",
                "root_cause",
                "solution",
                "severity",
                "status",
                "id",
                "issue_id",
                "session_db_id",
                "limit",
                "offset",
                "confirm",
                "scope",
            }
        elif target == "review_session":
            allowed = {
                "session_id",
                "user_message",
                "code_directory",
                "status",
                "code_patch",
                "git_commit",
                "id",
                "session_db_id",
                "limit",
                "offset",
                "confirm",
                "scope",
            }
        else:
            return action, target, data

        filtered = {k: v for k, v in data.items() if k in allowed}
        # 基本兜底：review_session 写入时需要 session_id
        if target == "review_session" and action in ("create", "update", "upsert"):
            if not filtered.get("session_id"):
                filtered["session_id"] = session_id
        return action, target, filtered

    async def _handle_issue_pattern_task(
        self, action: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理 IssuePattern 相关任务：
        - create / update / delete / sync
        """
        action = action or "create"
        action = action.lower()

        if action == "query":
            pattern_id = data.get("id") or data.get("pattern_id")
            if pattern_id:
                item = await self.db_service.get_issue_pattern_by_id(pattern_id)
                return {"items": [item] if item else [], "count": 1 if item else 0}
            status = data.get("status")
            limit = data.get("limit")
            items = await self.db_service.get_issue_patterns(status=status)
            if isinstance(limit, int) and limit > 0:
                items = items[:limit]
            return {"items": items, "count": len(items)}

        if action in ("create", "add", "insert", "upsert"):
            payload = self._fill_issue_pattern_defaults(data)
            
            # 去重逻辑：在 Weaviate 中查找相似的 IssuePattern
            similar = self._find_similar_issue_pattern(payload)
            if similar:
                existing_id = similar["sqlite_id"]
                similarity = similar["similarity"]
                log(
                    "db_manage_agent",
                    LogLevel.INFO,
                    f"🔄 发现相似记录 id={existing_id} (相似度={similarity:.2%})，执行更新而非新增",
                )
                # 更新已有记录
                updated = await self.db_service.update_issue_pattern(
                    pattern_id=existing_id,
                    error_type=payload.get("error_type"),
                    error_description=payload.get("error_description"),
                    problematic_pattern=payload.get("problematic_pattern"),
                    file_pattern=payload.get("file_pattern"),
                    class_pattern=payload.get("class_pattern"),
                    solution=payload.get("solution"),
                    severity=payload.get("severity"),
                    status=payload.get("status"),
                )
                layers = data.get("layers") if isinstance(data.get("layers"), list) else ["full"]
                sync_info = await self._sync_issue_pattern_if_possible(existing_id, layers)
                return {
                    "pattern_id": existing_id,
                    "action": "updated_existing",
                    "similarity": similarity,
                    "updated": updated,
                    "weaviate_sync": sync_info,
                }
            
            # 无相似记录，新增
            log(
                "db_manage_agent",
                LogLevel.INFO,
                f"🗄️ 写入 issue_patterns: keys={list(payload.keys())}",
            )
            pattern_id = await self.db_service.create_issue_pattern(**payload)
            layers = data.get("layers") if isinstance(data.get("layers"), list) else ["full"]
            sync_info = await self._sync_issue_pattern_if_possible(pattern_id, layers)
            return {"pattern_id": pattern_id, "action": "created_new", "weaviate_sync": sync_info}

        if action in ("update", "modify"):
            pattern_id = data.get("id") or data.get("pattern_id")
            if not pattern_id:
                if action == "update":
                    raise ValueError("更新 IssuePattern 需要提供 id")
                payload = self._fill_issue_pattern_defaults(data)
                pattern_id = await self.db_service.create_issue_pattern(**payload)
                layers = data.get("layers") if isinstance(data.get("layers"), list) else ["full"]
                sync_info = await self._sync_issue_pattern_if_possible(pattern_id, layers)
                return {"pattern_id": pattern_id, "weaviate_sync": sync_info}
            updated = await self.db_service.update_issue_pattern(
                pattern_id=pattern_id,
                error_type=data.get("error_type"),
                error_description=data.get("error_description"),
                problematic_pattern=data.get("problematic_pattern"),
                file_pattern=data.get("file_pattern"),
                class_pattern=data.get("class_pattern"),
                solution=data.get("solution"),
                severity=data.get("severity"),
                status=data.get("status"),
            )
            log("db_manage_agent", LogLevel.INFO, f"🗄️ 更新 issue_patterns id={pattern_id}")
            layers = data.get("layers") if isinstance(data.get("layers"), list) else ["full"]
            sync_info = await self._sync_issue_pattern_if_possible(pattern_id, layers)
            return {"pattern_id": pattern_id, "updated": updated, "weaviate_sync": sync_info}

        if action in ("delete", "remove"):
            pattern_id = data.get("id") or data.get("pattern_id")
            if not pattern_id:
                raise ValueError("删除 IssuePattern 需要提供 id")
            deleted = await self.db_service.delete_issue_pattern(pattern_id)
            log("db_manage_agent", LogLevel.INFO, f"🗄️ 删除 issue_patterns id={pattern_id}")
            weaviate_deleted = self._delete_weaviate_items(pattern_id)
            return {"pattern_id": pattern_id, "deleted": deleted, "weaviate_deleted": weaviate_deleted}

        if action == "delete_all":
            if data.get("confirm") is not True:
                raise ValueError("删除全部 IssuePattern 需要 confirm=true")
            patterns = await self.db_service.get_issue_patterns()
            pattern_ids = [p.get("id") for p in patterns if p.get("id") is not None]
            deleted_count = await self.db_service.delete_all_issue_patterns()
            weaviate_deleted = 0
            for pid in pattern_ids:
                weaviate_deleted += self._delete_weaviate_items(pid)
            # 如果按-id 删除结果少于 SQLite 删除的记录数，尝试清空整个 Weaviate collection 作为兜底
            if weaviate_deleted < (deleted_count or 0):
                try:
                    fallback_deleted = self.vector_service.delete_all_knowledge_items()
                    log(
                        "db_manage_agent",
                        LogLevel.INFO,
                        f"🔄 Weaviate 批量删除回退: 删除对象数={fallback_deleted}",
                    )
                    weaviate_deleted += fallback_deleted
                except Exception as e:
                    log(
                        "db_manage_agent",
                        LogLevel.WARNING,
                        f"⚠️ Weaviate 回退删除失败: {e}",
                    )

            log("db_manage_agent", LogLevel.INFO, f"🗄️ 批量删除 issue_patterns count={deleted_count}")
            return {"deleted_count": deleted_count, "weaviate_deleted": weaviate_deleted}

        if action == "sync":
            pattern_id = data.get("id") or data.get("pattern_id")
            if not pattern_id:
                raise ValueError("同步 IssuePattern 需要提供 id")
            layers = data.get("layers") if isinstance(data.get("layers"), list) else ["full"]
            sync_info = await self._sync_issue_pattern_if_possible(pattern_id, layers)
            return {"pattern_id": pattern_id, "weaviate_sync": sync_info}

        raise ValueError(f"不支持的 IssuePattern 操作: {action}")

    async def _handle_curated_issue_task(
        self, action: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        action = action.lower()
        if action == "query":
            issue_id = data.get("id") or data.get("issue_id")
            if issue_id:
                item = await self.db_service.get_curated_issue_by_id(issue_id)
                return {"items": [item] if item else [], "count": 1 if item else 0}
            session_db_id = data.get("session_db_id") or data.get("session_id")
            file_path = data.get("file_path")
            status = data.get("status")
            limit = data.get("limit")
            items = await self.db_service.get_curated_issues(
                session_db_id=session_db_id,
                file_path=file_path,
                status=status,
            )
            if isinstance(limit, int) and limit > 0:
                items = items[:limit]
            return {"items": items, "count": len(items)}
        if action in ("create", "add", "insert", "upsert"):
            data = self._fill_curated_issue_defaults(data)
            
            # 去重逻辑：检查同一 session_id + pattern_id 是否已有记录
            pattern_id = data.get("pattern_id")
            target_session_id = data.get("session_id")
            if pattern_id and target_session_id:
                existing = await self.db_service.get_curated_issue_by_session_and_pattern(
                    session_id=target_session_id,
                    pattern_id=pattern_id,
                )
                if existing:
                    log(
                        "db_manage_agent",
                        LogLevel.INFO,
                        f"🔄 发现已有 curated_issue (session_id={target_session_id}, pattern_id={pattern_id})，更新而非新增",
                    )
                    # 更新现有记录的状态
                    await self.db_service.update_curated_issue_status(
                        issue_id=existing["id"],
                        status=data.get("status", existing.get("status", "open")),
                    )
                    return {"issue_id": existing["id"], "action": "updated_existing"}
            
            log("db_manage_agent", LogLevel.INFO, "🗄️ 写入 curated_issues")
            issue_id = await self.db_service.create_curated_issue(
                session_id=data["session_id"],
                file_path=data["file_path"],
                start_line=data["start_line"],
                end_line=data["end_line"],
                code_snippet=data["code_snippet"],
                problem_phenomenon=data["problem_phenomenon"],
                root_cause=data["root_cause"],
                solution=data["solution"],
                severity=data.get("severity", "medium"),
                status=data.get("status", "open"),
                project_path=data.get("project_path"),
                pattern_id=data.get("pattern_id"),
            )
            return {"issue_id": issue_id}

        if action in ("update", "modify"):
            issue_id = data.get("id") or data.get("issue_id")
            if not issue_id:
                if action == "update":
                    raise ValueError("更新 CuratedIssue 需要提供 id")
                data = self._fill_curated_issue_defaults(data)
                issue_id = await self.db_service.create_curated_issue(
                    session_id=data["session_id"],
                    file_path=data["file_path"],
                    start_line=data["start_line"],
                    end_line=data["end_line"],
                    code_snippet=data["code_snippet"],
                    problem_phenomenon=data["problem_phenomenon"],
                    root_cause=data["root_cause"],
                    solution=data["solution"],
                    severity=data.get("severity", "medium"),
                    status=data.get("status", "open"),
                    project_path=data.get("project_path"),
                    pattern_id=data.get("pattern_id"),
                )
                return {"issue_id": issue_id}
            updated = await self.db_service.update_curated_issue_status(
                issue_id=issue_id,
                status=data.get("status", "open"),
            )
            log("db_manage_agent", LogLevel.INFO, f"🗄️ 更新 curated_issues id={issue_id}")
            return {"issue_id": issue_id, "updated": updated}

        if action in ("delete", "remove"):
            issue_id = data.get("id") or data.get("issue_id")
            if not issue_id:
                raise ValueError("删除 CuratedIssue 需要提供 id")
            deleted = await self.db_service.delete_curated_issue(issue_id)
            log("db_manage_agent", LogLevel.INFO, f"🗄️ 删除 curated_issues id={issue_id}")
            return {"issue_id": issue_id, "deleted": deleted}

        if action == "delete_all":
            if data.get("confirm") is not True:
                raise ValueError("删除全部 CuratedIssue 需要 confirm=true")
            deleted_count = await self.db_service.delete_all_curated_issues()
            log("db_manage_agent", LogLevel.INFO, f"🗄️ 批量删除 curated_issues count={deleted_count}")
            return {"deleted_count": deleted_count}

        raise ValueError(f"不支持的 CuratedIssue 操作: {action}")

    async def _handle_review_session_task(
        self, action: str, data: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        action = action.lower()
        if action == "query":
            db_id = data.get("id") or data.get("session_db_id")
            if db_id:
                item = await self.db_service.get_review_session_by_id(db_id)
                return {"items": [item] if item else [], "count": 1 if item else 0}
            status = data.get("status")
            limit = data.get("limit")
            offset = data.get("offset", 0)
            items = await self.db_service.get_review_sessions(
                status=status,
                limit=limit if isinstance(limit, int) and limit > 0 else None,
                offset=offset if isinstance(offset, int) and offset > 0 else 0,
            )
            return {"items": items, "count": len(items)}
        if action in ("create", "add", "insert", "upsert"):
            # 去重逻辑：检查同一 session_id 是否已有记录
            target_session_id = data.get("session_id", session_id)
            existing = await self.db_service.get_review_session_by_session_id(target_session_id)
            if existing:
                log(
                    "db_manage_agent",
                    LogLevel.INFO,
                    f"🔄 发现已有 review_session (session_id={target_session_id})，更新而非新增",
                )
                # 更新现有记录：刷新 status 与 时间戳；若有新的 user_message/code info，则同步更新
                await self.db_service.update_review_session_status(
                    db_id=existing["id"],
                    status=data.get("status", existing.get("status", "open")),
                )
                # 尝试同步 user_message / code_directory / code_patch / git_commit
                try:
                    # 如果传入了更详细的信息，更新对应字段并刷新 updated_at
                    if any(data.get(k) for k in ("user_message", "code_directory", "code_patch", "git_commit")):
                        with_exception = False
                        try:
                            with_db = self.db_service.get_session()
                            # perform a lightweight update
                            with with_db as db:
                                from infrastructure.database.sqlite.models import ReviewSession as _RS
                                obj = db.query(_RS).filter(_RS.id == existing["id"]).one_or_none()
                                if obj:
                                    if data.get("user_message"):
                                        obj.user_message = data.get("user_message")
                                    if data.get("code_directory"):
                                        obj.code_directory = data.get("code_directory")
                                    if data.get("code_patch") is not None:
                                        obj.code_patch = data.get("code_patch")
                                    if data.get("git_commit") is not None:
                                        obj.git_commit = data.get("git_commit")
                                    obj.updated_at = __import__("datetime").datetime.datetime.now()
                                    db.commit()
                        except Exception:
                            # 如果低层直接更新失败，忽略不影响主流程
                            pass
                except Exception:
                    pass
                return {"session_db_id": existing["id"], "action": "updated_existing"}
            
            log("db_manage_agent", LogLevel.INFO, "🗄️ 写入 review_sessions")
            session_db_id = await self.db_service.create_review_session(
                session_id=target_session_id,
                user_message=data.get("user_message", ""),
                code_directory=data.get("code_directory", ""),
                code_patch=data.get("code_patch"),
                git_commit=data.get("git_commit"),
                status=data.get("status", "open"),
            )
            return {"session_db_id": session_db_id}

        if action in ("update", "modify"):
            db_id = data.get("id") or data.get("session_db_id")
            if not db_id:
                if action == "update":
                    raise ValueError("更新 ReviewSession 需要提供 id")
                session_db_id = await self.db_service.create_review_session(
                    session_id=data.get("session_id", session_id),
                    user_message=data.get("user_message", ""),
                    code_directory=data.get("code_directory", ""),
                    code_patch=data.get("code_patch"),
                    git_commit=data.get("git_commit"),
                    status=data.get("status", "open"),
                )
                return {"session_db_id": session_db_id}
            updated = await self.db_service.update_review_session_status(
                db_id=db_id, status=data.get("status", "open")
            )
            log("db_manage_agent", LogLevel.INFO, f"🗄️ 更新 review_sessions id={db_id}")
            return {"session_db_id": db_id, "updated": updated}

        if action in ("delete", "remove"):
            db_id = data.get("id") or data.get("session_db_id")
            if not db_id:
                raise ValueError("删除 ReviewSession 需要提供 id")
            deleted = await self.db_service.delete_review_session(db_id)
            log("db_manage_agent", LogLevel.INFO, f"🗄️ 删除 review_sessions id={db_id}")
            return {"session_db_id": db_id, "deleted": deleted}

        if action == "delete_all":
            if data.get("confirm") is not True:
                raise ValueError("删除全部 ReviewSession 需要 confirm=true")
            deleted_count = await self.db_service.delete_all_review_sessions()
            log("db_manage_agent", LogLevel.INFO, f"🗄️ 批量删除 review_sessions count={deleted_count}")
            return {"deleted_count": deleted_count}

        raise ValueError(f"不支持的 ReviewSession 操作: {action}")

    async def _sync_issue_pattern_if_possible(
        self, pattern_id: int, layers: List[str]
    ) -> Dict[str, Any]:
        if not self.vector_service.client:
            log(
                "db_manage_agent",
                LogLevel.WARNING,
                "⚠️ 未配置 Weaviate client，跳过向量同步",
            )
            return {"skipped": True, "reason": "weaviate_client_not_configured"}
        sync_info = await self.sync_service.sync_issue_pattern(pattern_id, layers)
        log("db_manage_agent", LogLevel.INFO, f"🔄 同步 issue_patterns id={pattern_id} layers={layers} sync_info={sync_info}")
        return sync_info

    def _delete_weaviate_items(self, pattern_id: int) -> int:
        if not self.vector_service.client:
            return 0
        deleted_count = self.vector_service.delete_knowledge_items_by_sqlite_id(pattern_id)
        log("db_manage_agent", LogLevel.INFO, f"🔄 同步删除 weaviate items id={pattern_id} deleted_count={deleted_count}")
        return deleted_count

    def _fill_issue_pattern_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw_text = data.get("error_description") or data.get("problematic_pattern") or ""
        return {
            "error_type": data.get("error_type", "general"),
            "error_description": data.get("error_description", raw_text or ""),
            "problematic_pattern": data.get("problematic_pattern", raw_text or ""),
            "solution": data.get("solution", ""),
            "severity": data.get("severity", "medium"),
            "title": data.get("title"),
            "language": data.get("language"),
            "framework": data.get("framework"),
            "file_pattern": data.get("file_pattern", ""),
            "class_pattern": data.get("class_pattern", ""),
            "tags": data.get("tags"),
            "status": data.get("status", "active"),
        }

    def _fill_curated_issue_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw_text = data.get("problem_phenomenon") or ""
        return {
            "session_id": data.get("session_id", ""),
            "pattern_id": data.get("pattern_id"),
            "project_path": data.get("project_path"),
            "file_path": data.get("file_path", ""),
            "start_line": data.get("start_line", 0),
            "end_line": data.get("end_line", 0),
            "code_snippet": data.get("code_snippet", ""),
            "problem_phenomenon": data.get("problem_phenomenon", raw_text or ""),
            "root_cause": data.get("root_cause", ""),
            "solution": data.get("solution", ""),
            "severity": data.get("severity", "medium"),
            "status": data.get("status", "open"),
        }

    def _default_embed(self, text: str) -> List[float]:
        """
        轻量级嵌入函数，用于在缺少真实模型时提供稳定向量。
        """
        if text is None:
            text = ""
        total = float(sum(ord(c) for c in text))
        length = float(len(text) or 1)
        return [
            length,
            (total % 991) / 991.0,
            (total % 313) / 313.0,
        ]

    def _build_semantic_text(self, data: Dict[str, Any]) -> str:
        """
        构建用于语义匹配的文本，与 Weaviate 的 semantic 层保持一致。
        
        包含字段：error_type, severity, language, framework, error_description
        """
        parts = [
            f"[error_type] {data.get('error_type') or ''}",
            f"[severity] {data.get('severity') or ''}",
            f"[language] {data.get('language') or ''}",
            f"[framework] {data.get('framework') or ''}",
            f"[description] {data.get('error_description') or ''}",
        ]
        return "\n".join(parts)

    def _find_similar_issue_pattern(
        self, data: Dict[str, Any], similarity_threshold: float = 0.85
    ) -> Optional[Dict[str, Any]]:
        """
        在 Weaviate 中查找与给定数据相似的 IssuePattern。
        
        Args:
            data: 待匹配的 IssuePattern 数据
            similarity_threshold: 相似度阈值（0-1），越高越严格
            
        Returns:
            如果找到相似记录，返回包含 sqlite_id 和 distance 的字典；否则返回 None
        """
        if not self.vector_service.client:
            log(
                "db_manage_agent",
                LogLevel.DEBUG,
                "⚠️ Weaviate 未配置，跳过相似度查询",
            )
            return None
        
        # 构建语义文本并生成向量
        semantic_text = self._build_semantic_text(data)
        query_vector = self._default_embed(semantic_text)
        
        # 在 Weaviate 中搜索相似项
        try:
            results = self.vector_service.search_knowledge_items(
                query_vector=query_vector,
                limit=1,
                layer="semantic",
            )
            
            if not results:
                return None
            
            top_result = results[0]
            # Weaviate 返回的是 distance（距离），越小越相似
            # distance = 0 表示完全相同，distance = 2 表示完全不同（余弦距离范围 0-2）
            distance = top_result.get("_additional", {}).get("distance", 2.0)
            
            # 将 distance 转换为 similarity (1 - distance/2)
            similarity = 1.0 - (distance / 2.0)
            
            log(
                "db_manage_agent",
                LogLevel.INFO,
                f"🔍 相似度查询结果: sqlite_id={top_result.get('sqlite_id')}, "
                f"distance={distance:.4f}, similarity={similarity:.4f}",
            )
            
            if similarity >= similarity_threshold:
                return {
                    "sqlite_id": top_result.get("sqlite_id"),
                    "distance": distance,
                    "similarity": similarity,
                }
            
            return None
            
        except Exception as e:
            log(
                "db_manage_agent",
                LogLevel.WARNING,
                f"⚠️ 相似度查询失败: {e}",
            )
            return None

    def _format_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Render a table. Delegate to utils.format_table for consistent tabulate output when available.

        Keeps a fallback to the simple ASCII renderer if the helper is unavailable.
        """
        if not headers or not rows:
            return ""

        try:
            # prefer the shared helper for consistent wrapping/encoding
            return format_table(rows, headers=headers, tablefmt="grid")
        except Exception:
            # fallback: simple ASCII renderer
            def _col_width(idx: int) -> int:
                values = [headers[idx]] + [r[idx] for r in rows]
                return max(len(v) for v in values)

            widths = [_col_width(i) for i in range(len(headers))]

            def _fmt_row(cols: List[str]) -> str:
                return "| " + " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols)) + " |"

            sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
            lines = [sep, _fmt_row(headers), sep]
            for row in rows:
                lines.append(_fmt_row(row))
            lines.append(sep)
            return "\n".join(lines)

    def _summarize_tasks_for_log(self, tasks: List[Dict[str, Any]]) -> str:
        """按表/动作汇总 LLM 结构化任务，便于日志展示。"""
        if not tasks:
            return ""
        grouped: Dict[str, Dict[str, int]] = {}
        for task in tasks:
            raw_target = str(task.get("target") or task.get("table") or "")
            raw_action = str(task.get("action") or task.get("op") or "")
            data = task.get("data") if isinstance(task.get("data"), dict) else {}
            action, target, _ = self._normalize_task(raw_action, raw_target, data, "summary")
            target_key = target or "unknown"
            action_key = action or "unknown"
            grouped.setdefault(target_key, {})
            grouped[target_key][action_key] = grouped[target_key].get(action_key, 0) + 1

        headers = ["target", "action", "count"]
        rows: List[List[str]] = []
        for target, actions in grouped.items():
            for action, count in actions.items():
                rows.append([target, action, str(count)])
        return self._format_table(headers, rows)

    def _log_llm_tasks(self, stage: str, tasks: List[Dict[str, Any]]) -> None:
        """打印单次 LLM 输出的结构化任务及摘要。"""
        if not tasks:
            return
        try:
            pretty_tasks = json.dumps(tasks, ensure_ascii=False, indent=2)
        except Exception:
            pretty_tasks = str(tasks)
        log("db_manage_agent", LogLevel.INFO, f"📜 LLM 结构化任务({stage}):\n{pretty_tasks}")
        summary_table = self._summarize_tasks_for_log(tasks)
        if summary_table:
            log_table("db_manage_agent", LogLevel.INFO, summary_table, title=f"📌 LLM 任务摘要({stage}):", use_logger_prefix=False)

    def _build_db_summary_table(self, results: List[Dict[str, Any]]) -> str:
        """生成数据库写入摘要表格（ASCII）。"""
        if not results:
            return ""
        grouped: Dict[str, List[List[str]]] = {}
        for item in results:
            task = item.get("task") if isinstance(item.get("task"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            raw_target = str(task.get("target") or task.get("table") or "")
            raw_action = str(task.get("action") or task.get("op") or "")
            data = task.get("data") if isinstance(task.get("data"), dict) else {}
            action, target, norm_data = self._normalize_task(raw_action, raw_target, data, "summary")
            status = str(item.get("status", ""))
            row_id = (
                result.get("pattern_id")
                or result.get("issue_id")
                or result.get("session_db_id")
                or ""
            )
            rows = grouped.setdefault(target or "unknown", [])
            if norm_data:
                for key, value in norm_data.items():
                    try:
                        value_text = json.dumps(value, ensure_ascii=False)
                    except Exception:
                        value_text = str(value)
                    rows.append([action, status, str(key), value_text, str(row_id)])
            else:
                rows.append([action, status, "", "", str(row_id)])

        tables: List[str] = []
        headers = ["action", "status", "field", "value", "id"]
        for target, rows in grouped.items():
            table_text = self._format_table(headers, rows)
            if table_text:
                tables.append(f"[{target}]\n{table_text}")
        return "\n\n".join(tables)

    def _build_db_query_tables(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """生成查询结果表格（按表名分组）。"""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in results:
            if item.get("status") != "success":
                continue
            task = item.get("task") if isinstance(item.get("task"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            raw_target = str(task.get("target") or task.get("table") or "")
            raw_action = str(task.get("action") or task.get("op") or "")
            data = task.get("data") if isinstance(task.get("data"), dict) else {}
            action, target, _ = self._normalize_task(raw_action, raw_target, data, "summary")
            if action != "query":
                continue
            items = result.get("items") if isinstance(result.get("items"), list) else []
            if items:
                grouped.setdefault(target, []).extend(items)

        tables: Dict[str, str] = {}
        for target, items in grouped.items():
            if not items:
                continue
            # prefer model-defined column order: primary keys, foreign keys, then attributes
            try:
                from utils.pretty_db import _preferred_column_order

                headers = _preferred_column_order(target, sorted({k for item in items for k in item.keys()}))
            except Exception:
                headers = sorted({k for item in items for k in item.keys()})
            rows = []
            for item in items:
                rows.append([str(item.get(h, "")) for h in headers])
            table_text = self._format_table(headers, rows)
            if table_text:
                tables[target] = table_text
        return tables

    def _build_db_query_tables_tabulate(self, results: List[Dict[str, Any]], max_col_width: int = 120) -> Dict[str, str]:
        """
        使用 tabulate 渲染查询结果（按表名分组），返回 target->table_text 映射。
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in results:
            if item.get("status") != "success":
                continue
            task = item.get("task") if isinstance(item.get("task"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            raw_target = str(task.get("target") or task.get("table") or "")
            raw_action = str(task.get("action") or task.get("op") or "")
            data = task.get("data") if isinstance(task.get("data"), dict) else {}
            action, target, _ = self._normalize_task(raw_action, raw_target, data, "summary")
            if action != "query":
                continue
            items = result.get("items") if isinstance(result.get("items"), list) else []
            if items:
                grouped.setdefault(target, []).extend(items)

        # 调用 core.utils.pretty_db.tabulate_grouped_items
        try:
            tables = tabulate_grouped_items(grouped, max_col_width=max_col_width)
        except Exception:
            # 回退到原有 ASCII 表格
            tables = {}
            for target, items in grouped.items():
                if not items:
                    continue
                headers = sorted({k for item in items for k in item.keys()})
                rows = []
                for item in items:
                    rows.append([str(item.get(h, "")) for h in headers])
                table_text = self._format_table(headers, rows)
                if table_text:
                    tables[target] = table_text
        return tables