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
from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.database.weaviate.service import WeaviateVectorService
from infrastructure.database.vector_sync import (
    IssuePatternSyncService,
    DefaultKnowledgeEncodingAgent,
)
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

        # 数据层组件
        self.db_service = DatabaseService()
        self.vector_service = WeaviateVectorService(embed_fn=self._default_embed)
        self.encoding_agent = DefaultKnowledgeEncodingAgent(embed_fn=self._default_embed)
        self.sync_service = IssuePatternSyncService(
            db_service=self.db_service,
            vector_service=self.vector_service,
            agent=self.encoding_agent,
        )
    
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

        tasks = self._normalize_db_tasks(user_requirement)
        # 统一通过 LLM 翻译自然语言 db_tasks -> 结构化任务
        if tasks:
            llm_plan = await self._translate_tasks_with_llm(tasks, session_id=session_id)
            if llm_plan:
                tasks = llm_plan

        if not tasks:
            return {
                "status": "noop",
                "session_id": session_id,
                "results": [],
                "message": "未识别到可执行的数据库任务",
            }

        results: List[Dict[str, Any]] = []
        for task in tasks:
            try:
                result = await self._handle_single_db_task(task, session_id)
                results.append({"task": task, "status": "success", "result": result})
            except Exception as e:
                log("db_manage_agent", LogLevel.ERROR, f"❌ 处理数据库任务失败: {e}")
                results.append({"task": task, "status": "failed", "error": str(e)})

        overall_status = (
            "success"
            if all(item["status"] == "success" for item in results)
            else "partial"
        )

        return {
            "status": overall_status,
            "session_id": session_id,
            "results": results,
            "message": "数据库任务执行完成",
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
            tasks = user_requirement.get("db_tasks") or []
            if isinstance(tasks, list):
                return [t for t in tasks if isinstance(t, dict)]
        return []

    async def _translate_tasks_with_llm(
        self, raw_tasks: List[Dict[str, Any]], session_id: str
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
            )
        except Exception:
            system_prompt = (
                "你是数据库管理代理。根据 db_tasks 的 project 和 description，"
                "将需求翻译为 SQLite 的结构化操作，字段对齐 review_sessions/curated_issues/issue_patterns。"
                "输出 JSON 数组，仅包含 target, action, data，禁止附加说明。"
            )

        try:
            user_content = json.dumps({"db_tasks": raw_tasks}, ensure_ascii=False)
        except Exception:
            user_content = str(raw_tasks)

        prompt = f"{system_prompt}\n用户输入：{user_content}\n输出 JSON："
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.used_device == "gpu":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed = self._extract_structured_tasks_from_text(generated)
        if parsed:
            log("db_manage_agent", LogLevel.INFO, f"✅ LLM 翻译得到 {len(parsed)} 条任务")
        else:
            log("db_manage_agent", LogLevel.WARNING, "⚠️ 未能解析 LLM 输出，返回空任务")
        return parsed

    def _extract_structured_tasks_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        从模型输出中提取 JSON 数组；容错：截取首尾的 JSON 片段。
        """
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except Exception:
            pass
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                snippet = text[start : end + 1]
                data = json.loads(snippet)
                return data if isinstance(data, list) else []
        except Exception:
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
        action = str(
            task.get("action") or task.get("op") or task.get("type") or ""
        ).lower()
        target = str(
            task.get("target") or task.get("table") or task.get("object") or ""
        ).lower()
        data = task.get("data") if isinstance(task.get("data"), dict) else {}

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
        if target in ("curated_issue", "issue", "curated"):
            return await self._handle_curated_issue_task(action, data)
        if target in ("review_session", "session"):
            return await self._handle_review_session_task(action, data, session_id)

        raise ValueError(f"未知的数据库任务目标: {target or '未提供'}")

    async def _handle_issue_pattern_task(
        self, action: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理 IssuePattern 相关任务：
        - create / update / delete / sync
        """
        action = action or "create"
        action = action.lower()

        if action in ("create", "add", "insert", "upsert"):
            payload = self._fill_issue_pattern_defaults(data)
            pattern_id = await self.db_service.create_issue_pattern(**payload)
            layers = data.get("layers") if isinstance(data.get("layers"), list) else ["full"]
            sync_info = await self._sync_issue_pattern_if_possible(pattern_id, layers)
            return {"pattern_id": pattern_id, "weaviate_sync": sync_info}

        if action in ("update", "modify", "upsert"):
            pattern_id = data.get("id") or data.get("pattern_id")
            if not pattern_id:
                raise ValueError("更新 IssuePattern 需要提供 id")
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
            layers = data.get("layers") if isinstance(data.get("layers"), list) else ["full"]
            sync_info = await self._sync_issue_pattern_if_possible(pattern_id, layers)
            return {"pattern_id": pattern_id, "updated": updated, "weaviate_sync": sync_info}

        if action in ("delete", "remove"):
            pattern_id = data.get("id") or data.get("pattern_id")
            if not pattern_id:
                raise ValueError("删除 IssuePattern 需要提供 id")
            deleted = await self.db_service.delete_issue_pattern(pattern_id)
            weaviate_deleted = self._delete_weaviate_items(pattern_id)
            return {"pattern_id": pattern_id, "deleted": deleted, "weaviate_deleted": weaviate_deleted}

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
        if action in ("create", "add", "insert"):
            required_fields = ["session_id", "file_path", "start_line", "end_line", "code_snippet", "problem_phenomenon", "root_cause", "solution"]
            missing = [f for f in required_fields if f not in data]
            if missing:
                raise ValueError(f"创建 CuratedIssue 缺少必填字段: {missing}")
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
                raise ValueError("更新 CuratedIssue 需要提供 id")
            updated = await self.db_service.update_curated_issue_status(
                issue_id=issue_id,
                status=data.get("status", "open"),
            )
            return {"issue_id": issue_id, "updated": updated}

        if action in ("delete", "remove"):
            issue_id = data.get("id") or data.get("issue_id")
            if not issue_id:
                raise ValueError("删除 CuratedIssue 需要提供 id")
            deleted = await self.db_service.delete_curated_issue(issue_id)
            return {"issue_id": issue_id, "deleted": deleted}

        raise ValueError(f"不支持的 CuratedIssue 操作: {action}")

    async def _handle_review_session_task(
        self, action: str, data: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        action = action.lower()
        if action in ("create", "add", "insert"):
            session_db_id = await self.db_service.create_review_session(
                session_id=data.get("session_id", session_id),
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
                raise ValueError("更新 ReviewSession 需要提供 id")
            updated = await self.db_service.update_review_session_status(
                db_id=db_id, status=data.get("status", "open")
            )
            return {"session_db_id": db_id, "updated": updated}

        if action in ("delete", "remove"):
            db_id = data.get("id") or data.get("session_db_id")
            if not db_id:
                raise ValueError("删除 ReviewSession 需要提供 id")
            deleted = await self.db_service.delete_review_session(db_id)
            return {"session_db_id": db_id, "deleted": deleted}

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
        return await self.sync_service.sync_issue_pattern(pattern_id, layers)

    def _delete_weaviate_items(self, pattern_id: int) -> int:
        if not self.vector_service.client:
            return 0
        return self.vector_service.delete_knowledge_items_by_sqlite_id(pattern_id)

    def _fill_issue_pattern_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        required = ["error_type", "error_description", "problematic_pattern", "solution"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            raise ValueError(f"创建 IssuePattern 缺少必填字段: {missing}")

        return {
            "error_type": data["error_type"],
            "error_description": data.get("error_description", ""),
            "problematic_pattern": data.get("problematic_pattern", ""),
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