from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.database.weaviate import WeaviateVectorService


@dataclass
class AgentResult:
    """
    知识编码 Agent 的返回值：包含文本、向量以及写入 Weaviate 所需的结构化字段。
    """

    text_payload: str
    vector: List[float]
    weaviate_payload: Dict[str, Any]


class KnowledgeEncodingAgent(Protocol):
    """
    负责将 IssuePattern 的结构化字段转换为向量与 Weaviate payload 的 Agent。
    """

    def encode_issue_pattern(self, record: Dict[str, Any]) -> AgentResult:
        ...


@dataclass
class IssuePatternRecord:
    """
    SQLite 中 IssuePattern 的抽象视图，便于同步与测试。
    """

    id: int
    kb_code: Optional[str]
    error_type: str
    severity: str
    status: str
    language: Optional[str]
    framework: Optional[str]
    error_description: str
    problematic_pattern: str
    solution: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IssuePatternRecord":
        return cls(
            id=data["id"],
            kb_code=data.get("kb_code"),
            error_type=data["error_type"],
            severity=data["severity"],
            status=data.get("status", "active"),
            language=data.get("language"),
            framework=data.get("framework"),
            error_description=data.get("error_description") or "",
            problematic_pattern=data.get("problematic_pattern") or "",
            solution=data.get("solution") or "",
        )

    def to_agent_payload(self) -> Dict[str, Any]:
        return {
            "sqlite_id": self.id,
            "kb_code": self.kb_code,
            "error_type": self.error_type,
            "severity": self.severity,
            "status": self.status,
            "language": self.language,
            "framework": self.framework,
            "error_description": self.error_description,
            "problematic_pattern": self.problematic_pattern,
            "solution": self.solution,
        }


class IssuePatternSyncService:
    """
    负责将 SQLite 中的 IssuePattern 同步到 Weaviate。
    """

    def __init__(
        self,
        db_service: DatabaseService,
        vector_service: WeaviateVectorService,
        agent: KnowledgeEncodingAgent,
    ) -> None:
        self.db_service = db_service
        self.vector_service = vector_service
        self.agent = agent

    async def sync_issue_pattern(self, pattern_id: int) -> str:
        """
        同步单条 IssuePattern：读取 SQLite -> Agent 编码 -> 写入 Weaviate。
        """
        record_dict = await self.db_service.get_issue_pattern_by_id(pattern_id)
        if not record_dict:
            raise ValueError(f"IssuePattern {pattern_id} not found")
        record = IssuePatternRecord.from_dict(record_dict)
        return self._write_to_weaviate(record)

    async def sync_all_issue_patterns(self, status: Optional[str] = "active") -> List[str]:
        """
        同步满足条件的所有 IssuePattern，返回已写入的 Weaviate UUID 列表。
        """
        patterns = await self.db_service.get_issue_patterns(status=status)
        uuids: List[str] = []
        for pattern in patterns:
            record = IssuePatternRecord.from_dict(pattern)
            uuids.append(self._write_to_weaviate(record))
        return uuids

    def _write_to_weaviate(self, record: IssuePatternRecord) -> str:
        agent_result = self.agent.encode_issue_pattern(record.to_agent_payload())
        payload = agent_result.weaviate_payload
        return self.vector_service.create_knowledge_item(
            sqlite_id=payload["sqlite_id"],
            kb_code=payload.get("kb_code"),
            error_type=payload["error_type"],
            severity=payload["severity"],
            status=payload.get("status", "active"),
            language=payload.get("language"),
            framework=payload.get("framework"),
            error_description=payload.get("error_description"),
            problematic_pattern=payload.get("problematic_pattern"),
            solution=payload.get("solution"),
            vector=agent_result.vector,
        )


