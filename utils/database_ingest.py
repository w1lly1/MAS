import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.database.vector_sync import IssuePatternSyncService
from core.agents.ai_driven_database_manage_agent import DefaultKnowledgeEncodingAgent
from infrastructure.database.weaviate.service import WeaviateVectorService
from infrastructure.database.sqlite.models import IssuePattern, CuratedIssue
from utils import log, LogLevel

class DatabaseIngestTool:
    """
    硬编码数据库导入工具 - 不经过 LLM，直接解析结构化 JSON 并更新 SQLite + Weaviate。
    支持 ReviewSession -> IssuePattern -> CuratedIssue 的层级关联。
    """

    def __init__(self):
        self.db_service = DatabaseService()
        self.vector_service = WeaviateVectorService(embed_fn=self._default_embed)
        self.encoding_agent = DefaultKnowledgeEncodingAgent(embed_fn=self._default_embed)
        self.sync_service = IssuePatternSyncService(
            db_service=self.db_service,
            vector_service=self.vector_service,
            agent=self.encoding_agent,
        )
        # 尝试连接向量数据库
        self.vector_service.connect(auto_create_schema=True)

    def _default_embed(self, text: str) -> List[float]:
        """降级向量生成（与 Agent 内部逻辑一致）"""
        if not text:
            text = ""
        total = float(sum(ord(c) for c in text))
        length = float(len(text) or 1)
        return [length, (total % 997) / 997.0, (total % 389) / 389.0]

    def _norm_text(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        return " ".join(text.split())

    def _build_pattern_fingerprint(self, pattern_data: Dict[str, Any]) -> str:
        parts = [
            self._norm_text(pattern_data.get("title")),
            self._norm_text(pattern_data.get("error_type")),
            self._norm_text(pattern_data.get("language")),
            self._norm_text(pattern_data.get("framework")),
            self._norm_text(pattern_data.get("error_description"))[:280],
        ]
        return "|".join(parts)

    def _build_issue_fingerprint(self, issue_data: Dict[str, Any], session_id: int, pattern_id: int) -> str:
        parts = [
            str(session_id),
            str(pattern_id),
            self._norm_text(issue_data.get("project_path")),
            self._norm_text(issue_data.get("file_path")),
            str(issue_data.get("start_line") or ""),
            str(issue_data.get("end_line") or ""),
            self._norm_text(issue_data.get("problem_phenomenon"))[:220],
            self._norm_text(issue_data.get("root_cause"))[:220],
        ]
        return "|".join(parts)

    async def _find_existing_pattern_id(self, pattern_data: Dict[str, Any]) -> Optional[int]:
        """按标题 + 关键语义字段匹配已有 Pattern，避免重复导入。"""
        incoming_fp = self._build_pattern_fingerprint(pattern_data)
        with self.db_service.get_session() as db_session:
            rows = db_session.query(IssuePattern).all()
            for row in rows:
                existing_fp = self._build_pattern_fingerprint(
                    {
                        "title": row.title,
                        "error_type": row.error_type,
                        "language": row.language,
                        "framework": row.framework,
                        "error_description": row.error_description,
                    }
                )
                if existing_fp == incoming_fp:
                    return int(row.id)
        return None

    async def _find_existing_curated_issue_id(
        self,
        issue_data: Dict[str, Any],
        session_db_id: int,
        pattern_id: int,
    ) -> Optional[int]:
        """按核心定位与语义字段匹配已有 CuratedIssue，避免重复导入。"""
        incoming_fp = self._build_issue_fingerprint(issue_data, session_db_id, pattern_id)
        with self.db_service.get_session() as db_session:
            rows = (
                db_session.query(CuratedIssue)
                .filter(CuratedIssue.session_id == session_db_id)
                .filter(CuratedIssue.pattern_id == pattern_id)
                .all()
            )
            for row in rows:
                existing_fp = self._build_issue_fingerprint(
                    {
                        "project_path": row.project_path,
                        "file_path": row.file_path,
                        "start_line": row.start_line,
                        "end_line": row.end_line,
                        "problem_phenomenon": row.problem_phenomenon,
                        "root_cause": row.root_cause,
                    },
                    session_db_id,
                    pattern_id,
                )
                if existing_fp == incoming_fp:
                    return int(row.id)
        return None

    async def process_file(self, file_path: str):
        """解析 JSON 文件并按顺序执行导入"""
        path = Path(file_path)
        if not path.exists():
            log("ingest_tool", LogLevel.ERROR, f"文件未找到: {file_path}")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
        except Exception as e:
            log("ingest_tool", LogLevel.ERROR, f"JSON 解析失败: {e}")
            return

        data_list = content.get("data", [])
        log("ingest_tool", LogLevel.INFO, f"🚀 开始处理 {len(data_list)} 组结构化数据...")

        for entry in data_list:
            await self._process_entry(entry)

        log("ingest_tool", LogLevel.INFO, "✅ 所有数据导入完成。")

    async def _process_entry(self, entry: Dict[str, Any]):
        """处理单组关联数据 (Pattern + Instances)"""
        pattern_data = entry.get("pattern")
        instances = entry.get("instances", [])

        if not pattern_data:
            log("ingest_tool", LogLevel.WARNING, "跳过无效条目：缺失 pattern 定义")
            return

        # 1. 处理 IssuePattern (SQLite + Weaviate)
        # 按标题+关键语义字段去重，不依赖 get_issue_patterns 返回字段。
        pattern_id = await self._find_existing_pattern_id(pattern_data)
        if pattern_id:
            log("ingest_tool", LogLevel.INFO, f"🔄 匹配到已有 Pattern，执行更新: {pattern_data.get('title')}")
            await self.db_service.update_issue_pattern(
                pattern_id=pattern_id,
                error_type=pattern_data.get("error_type"),
                error_description=pattern_data.get("error_description"),
                problematic_pattern=pattern_data.get("problematic_pattern"),
                file_pattern=pattern_data.get("file_pattern", ""),
                class_pattern=pattern_data.get("class_pattern", ""),
                solution=pattern_data.get("solution"),
                severity=pattern_data.get("severity"),
                status=pattern_data.get("status")
            )

        if not pattern_id:
            log("ingest_tool", LogLevel.INFO, f"➕ 创建新 Pattern: {pattern_data.get('title')}")
            # create_issue_pattern 同样是异步方法
            pattern_id = await self.db_service.create_issue_pattern(
                title=pattern_data.get("title"),
                error_type=pattern_data.get("error_type"),
                error_description=pattern_data.get("error_description"),
                problematic_pattern=pattern_data.get("problematic_pattern"),
                solution=pattern_data.get("solution"),
                severity=pattern_data.get("severity"),
                language=pattern_data.get("language"),
                framework=pattern_data.get("framework"),
                file_pattern=pattern_data.get("file_pattern", ""),
                class_pattern=pattern_data.get("class_pattern", ""),
                tags=pattern_data.get("tags"),
                status=pattern_data.get("status")
            )

        # 强制同步向量库
        try:
            await self.sync_service.sync_issue_pattern(pattern_id)
            log("ingest_tool", LogLevel.INFO, f"🧠 向量同步成功 (ID: {pattern_id})")
        except Exception as e:
            log("ingest_tool", LogLevel.ERROR, f"❌ 向量同步失败: {e}")

        # 2. 处理每个代码实例及其关联会话
        for inst in instances:
            session_meta = inst.get("session_meta", {})
            issue_data = inst.get("issue", {})

            # 2.1 处理 ReviewSession
            # 如果没有 session_id，生成一个默认的
            sid = session_meta.get("session_id", "manual-import-session")
            
            # 使用同步方法获取 Session (DatabaseService 中的 get_review_sessions 等方法定义并没有 async)
            # 通过阅读源码发现 DatabaseService.get_review_sessions 实际上定义为 async，报错可能来自其它处
            # 但为了稳妥，我们使用同步搜索逻辑（如果确实是 async 则配合 await）
            existing_session = await self.db_service.get_review_session_by_session_id(sid)
            session_db_id = existing_session.get("id") if existing_session else None
            
            if not session_db_id:
                session_db_id = await self.db_service.create_review_session(
                    session_id=sid,
                    user_message=session_meta.get("user_message", "Ingested from JSON"),
                    code_directory=session_meta.get("code_directory", ""),
                    status="completed"
                )
                log("ingest_tool", LogLevel.INFO, f"📁 创建 Session (ID: {sid})")

            # 2.2 处理 CuratedIssue（先去重）
            issue_data["session_id"] = session_db_id
            issue_data["pattern_id"] = pattern_id
            existing_issue_id = await self._find_existing_curated_issue_id(
                issue_data,
                session_db_id=session_db_id,
                pattern_id=pattern_id,
            )
            if existing_issue_id:
                log(
                    "ingest_tool",
                    LogLevel.INFO,
                    f"⏭️ 跳过重复问题实例: file={issue_data.get('file_path')} existing_id={existing_issue_id}",
                )
                continue
            
            # 创建问题实例 (SQLite)
            try:
                # 假设 DatabaseService 有这个方法，或者使用 underlying session
                with self.db_service.get_session() as db_session:
                    from infrastructure.database.sqlite.models import CuratedIssue
                    new_issue = CuratedIssue(**issue_data)
                    db_session.add(new_issue)
                    db_session.commit()
                log("ingest_tool", LogLevel.INFO, f"📍 记录问题实例: {issue_data.get('file_path')}")
            except Exception as e:
                log("ingest_tool", LogLevel.ERROR, f"❌ 无法创建问题实例: {e}")

if __name__ == "__main__":
    # 简易本地测试
    import sys
    if len(sys.argv) > 1:
        tool = DatabaseIngestTool()
        asyncio.run(tool.process_file(sys.argv[1]))
    else:
        print("Usage: python utils/database_ingest.py <json_file>")
