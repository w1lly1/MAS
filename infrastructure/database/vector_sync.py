from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

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
    # 可选的分层向量，若为空则上层可复用单一向量
    layer_vectors: Optional[Dict[str, List[float]]] = None


class KnowledgeEncodingAgent(Protocol):
    """
    负责将 IssuePattern 的结构化字段转换为向量与 Weaviate payload 的 Agent。
    """

    def encode_issue_pattern(self, record: Dict[str, Any]) -> AgentResult:
        ...


class DefaultKnowledgeEncodingAgent(KnowledgeEncodingAgent):
    """
    默认的知识编码 Agent。

    - 复用 Weaviate 分层文本构建思路，生成多分层向量
    - 若未提供 embed_fn，则使用轻量级可重复的降级向量生成
    """

    def __init__(self, embed_fn: Optional[Callable[[str], List[float]]] = None) -> None:
        self.embed_fn = embed_fn or self._fallback_embed

    def encode_issue_pattern(self, record: Dict[str, Any]) -> AgentResult:
        payload = {
            "sqlite_id": record["sqlite_id"],
            "error_type": record.get("error_type"),
            "severity": record.get("severity"),
            "status": record.get("status", "active"),
            "language": record.get("language"),
            "framework": record.get("framework"),
            "error_description": record.get("error_description"),
            "problematic_pattern": record.get("problematic_pattern"),
            "solution": record.get("solution"),
            "file_pattern": record.get("file_pattern"),
            "class_pattern": record.get("class_pattern"),
        }

        layer_texts = self._build_layer_texts(payload)
        layer_vectors: Dict[str, List[float]] = {
            layer: self.embed_fn(text) for layer, text in layer_texts.items()
        }
        full_vector = layer_vectors.get("full") or next(iter(layer_vectors.values()))

        return AgentResult(
            text_payload=layer_texts.get("full") or "",
            vector=full_vector,
            layer_vectors=layer_vectors,
            weaviate_payload=payload,
        )

    # --- helpers --------------------------------------------------------- #
    def _fallback_embed(self, text: str) -> List[float]:
        """
        简单的可重复降级向量生成，避免依赖真实大模型。
        """
        if text is None:
            text = ""
        total = float(sum(ord(c) for c in text))
        length = float(len(text) or 1)
        # 生成一个稳定但非零的三维向量
        return [
            length,
            (total % 997) / 997.0,
            (total % 389) / 389.0,
        ]

    def _build_layer_texts(self, props: Dict[str, Any]) -> Dict[str, str]:
        """
        构建多分层文本，与 Weaviate 分层逻辑保持一致。
        """
        semantic = "\n".join(
            [
                f"[error_type] {props.get('error_type') or ''}",
                f"[severity] {props.get('severity') or ''}",
                f"[language] {props.get('language') or ''}",
                f"[framework] {props.get('framework') or ''}",
                f"[description] {props.get('error_description') or ''}",
            ]
        )
        code_pattern = "\n".join(
            [
                f"[problematic_pattern] {props.get('problematic_pattern') or ''}",
                f"[file_pattern] {props.get('file_pattern') or ''}",
                f"[class_pattern] {props.get('class_pattern') or ''}",
                f"[language] {props.get('language') or ''}",
            ]
        )
        solution = "\n".join(
            [
                f"[solution] {props.get('solution') or ''}",
                f"[error_description] {props.get('error_description') or ''}",
                f"[severity] {props.get('severity') or ''}",
            ]
        )
        full = "\n".join(
            [
                f"[error_type] {props.get('error_type') or ''}",
                f"[severity] {props.get('severity') or ''}",
                f"[language] {props.get('language') or ''}",
                f"[framework] {props.get('framework') or ''}",
                f"[description] {props.get('error_description') or ''}",
                f"[pattern] {props.get('problematic_pattern') or ''}",
                f"[solution] {props.get('solution') or ''}",
                f"[file_pattern] {props.get('file_pattern') or ''}",
                f"[class_pattern] {props.get('class_pattern') or ''}",
            ]
        )
        return {
            "semantic": semantic,
            "code_pattern": code_pattern,
            "solution": solution,
            "full": full,
        }


@dataclass
class IssuePatternRecord:
    """
    SQLite 中 IssuePattern 的抽象视图，便于同步与测试。
    """

    id: int
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

    async def sync_issue_pattern(self, pattern_id: int, layers: List[str] = ["full"]) -> Dict[str, str]:
        """
        同步单条 IssuePattern 并支持多分层向量存储
        
        Args:
            pattern_id: IssuePattern ID
            layers: 分层列表，可选值：["semantic", "code_pattern", "solution", "full"]
        
        Returns:
            各分层对应的对象ID字典
        """
            
        record_dict = await self.db_service.get_issue_pattern_by_id(pattern_id)
        if not record_dict:
            raise ValueError(f"IssuePattern {pattern_id} not found")
        record = IssuePatternRecord.from_dict(record_dict)
        return self._write_to_weaviate(record, layers)

    async def sync_all_issue_patterns(self, status: Optional[str] = "active", layers: List[str] = ["full"]) -> List[Dict[str, str]]:
        """
        同步满足条件的所有 IssuePattern 并支持多分层向量存储
        
        Args:
            status: 状态过滤条件
            layers: 分层列表
        
        Returns:
            各IssuePattern的分层对象ID字典列表
        """

        patterns = await self.db_service.get_issue_patterns(status=status)
        results: List[Dict[str, str]] = []
        for pattern in patterns:
            record = IssuePatternRecord.from_dict(pattern)
            results.append(self._write_to_weaviate(record, layers))
        return results

    def _write_to_weaviate(self, record: IssuePatternRecord, layers: List[str]) -> Dict[str, str]:
        """
        将IssuePattern写入Weaviate并支持多分层向量存储
        """
        agent_result = self.agent.encode_issue_pattern(record.to_agent_payload())
        payload = agent_result.weaviate_payload
        
        # 为每个分层生成向量，优先使用 Agent 提供的分层向量
        vectors = agent_result.layer_vectors.copy() if agent_result.layer_vectors else {}
        for layer in layers:
            if layer not in vectors:
                vectors[layer] = agent_result.vector
        
        return self.vector_service.create_knowledge_item_with_layered_vectors(
            sqlite_id=payload["sqlite_id"],
            error_type=payload["error_type"],
            severity=payload["severity"],
            status=payload.get("status", "active"),
            language=payload.get("language"),
            framework=payload.get("framework"),
            error_description=payload.get("error_description"),
            problematic_pattern=payload.get("problematic_pattern"),
            solution=payload.get("solution"),
            vectors=vectors,
        )