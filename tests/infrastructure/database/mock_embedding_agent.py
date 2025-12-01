from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

EmbedFn = Callable[[str], List[float]]


@dataclass
class AgentResult:
    """
    代表一次“知识编码”任务的结果：

    - text_payload：准备交给 embedding 模型的文本
    - vector：embedding 模型返回的向量
    - weaviate_payload：写入 Weaviate 所需的结构化字段
    """

    text_payload: str
    vector: List[float]
    weaviate_payload: Dict[str, Any]


class MockDBEmbeddingAgent:
    """
    用于测试的“数据库 embedding Agent”占位实现。

    真实系统中，这个角色应由大模型负责理解记录并生成向量；
    此 mock 版本只是在测试中验证接口和数据流是否合理。
    """

    def __init__(self, embed_fn: EmbedFn) -> None:
        self.embed_fn = embed_fn

    def encode_issue_pattern(self, record: Dict[str, Any]) -> AgentResult:
        """
        将 IssuePattern / KnowledgeBase 的结构化字段转换为 Weaviate 写入所需信息。

        record 期望包含至少以下键：
        - sqlite_id
        - error_type / severity / status
        - 可选字段：kb_code / language / framework / error_description / problematic_pattern / solution
        """
        text = self._build_issue_pattern_text(record)
        vector = self.embed_fn(text)
        weaviate_payload = {
            "sqlite_id": record["sqlite_id"],
            "kb_code": record.get("kb_code"),
            "error_type": record.get("error_type"),
            "severity": record.get("severity"),
            "status": record.get("status", "active"),
            "language": record.get("language"),
            "framework": record.get("framework"),
            "error_description": record.get("error_description"),
            "problematic_pattern": record.get("problematic_pattern"),
            "solution": record.get("solution"),
        }
        return AgentResult(
            text_payload=text,
            vector=vector,
            weaviate_payload=weaviate_payload,
        )

    def _build_issue_pattern_text(self, props: Dict[str, Any]) -> str:
        """
        与 WeaviateVectorService 中相同的兜底拼接策略，
        便于未来替换为真正的“大模型 Agent”时做对比。
        """
        parts = [
            f"[kb_code] {props.get('kb_code') or ''}",
            f"[error_type] {props.get('error_type') or ''}",
            f"[severity] {props.get('severity') or ''}",
            f"[language] {props.get('language') or ''}",
            f"[framework] {props.get('framework') or ''}",
            f"[description] {props.get('error_description') or ''}",
            f"[pattern] {props.get('problematic_pattern') or ''}",
            f"[solution] {props.get('solution') or ''}",
        ]
        return "\n".join(parts)


