from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from weaviate import WeaviateClient

EmbedFn = Callable[[str], List[float]]


@dataclass
class WeaviateConfig:
    """
    Weaviate 连接配置。

    - url: Weaviate 实例的完整 URL，例如 http://localhost:8080
    - api_key: 如果开启了 API Key 认证，则在此提供；否则为 None
    - timeout: 请求超时时间（秒）
    """

    url: str
    api_key: Optional[str] = None
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "WeaviateConfig":
        """
        从环境变量构造配置：
        - WEAVIATE_URL（默认 http://localhost:8080）
        - WEAVIATE_API_KEY（可选）
        - WEAVIATE_TIMEOUT（可选，秒）
        """
        url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        api_key = os.getenv("WEAVIATE_API_KEY") or None
        timeout_str = os.getenv("WEAVIATE_TIMEOUT", "30")
        try:
            timeout = int(timeout_str)
        except ValueError:
            timeout = 30
        return cls(url=url, api_key=api_key, timeout=timeout)


class WeaviateVectorService:
    """
    Weaviate 向量索引服务。

    当前主要面向 IssuePattern（知识模式），使用 Weaviate 中的 "KnowledgeItem" class：
    - sqlite_id: 对应 SQLite 中 IssuePattern 的主键 id
    - kb_code, error_type, severity, status 等用于过滤
    - 向量字段由 Weaviate 维护（vectorizer: none + 手动传 vector）
    """

    KNOWLEDGE_CLASS = "KnowledgeItem"

    def __init__(
        self,
        config: Optional[WeaviateConfig] = None,
        embed_fn: Optional[EmbedFn] = None,
    ) -> None:
        self.config = config or WeaviateConfig.from_env()
        self.embed_fn = embed_fn

        # 延迟初始化 WeaviateClient：
        # - 在单元测试中会直接用 fake client 覆盖 self.client
        # - 在实际运行中，外部可以注入已经配置好的 WeaviateClient 实例
        # 后续如需自动初始化，可在这里或单独的 connect() 方法中补充。
        self.client: Optional[WeaviateClient] = None

    # --------------------------------------------------------------------- #
    # schema 管理
    # --------------------------------------------------------------------- #
    def ensure_knowledge_schema(self) -> None:
        """
        确保 KnowledgeItem class 存在，如不存在则创建。

        Schema 设计（简化版）：
        - class: KnowledgeItem
        - properties:
          - sqlite_id: int
          - kb_code: text
          - error_type: text
          - severity: text
          - status: text
          - language: text
          - framework: text
        - vectorizer: none （向量由外部传入）
        """
        schema = self.client.schema.get()
        classes = {c["class"] for c in schema.get("classes", [])}
        if self.KNOWLEDGE_CLASS in classes:
            return

        class_obj = {
            "class": self.KNOWLEDGE_CLASS,
            "description": "IssuePattern knowledge items used for semantic search",
            "vectorizer": "none",
            "properties": [
                {"name": "sqlite_id", "dataType": ["int"]},
                {"name": "kb_code", "dataType": ["text"]},
                {"name": "error_type", "dataType": ["text"]},
                {"name": "severity", "dataType": ["text"]},
                {"name": "status", "dataType": ["text"]},
                {"name": "language", "dataType": ["text"]},
                {"name": "framework", "dataType": ["text"]},
            ],
        }
        self.client.schema.create_class(class_obj)

    # --------------------------------------------------------------------- #
    # IssuePattern / KnowledgeItem CRUD + 向量写入
    # --------------------------------------------------------------------- #
    def _build_issue_pattern_text(self, props: Dict[str, Any]) -> str:
        """
        将 IssuePattern 结构化字段拼接成用于 embedding 的文本。
        仅在未显式提供向量且配置了 embed_fn 时使用。
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

    def _build_props(
        self,
        sqlite_id: int,
        kb_code: Optional[str],
        error_type: str,
        severity: str,
        status: str = "active",
        language: Optional[str] = None,
        framework: Optional[str] = None,
        error_description: Optional[str] = None,
        problematic_pattern: Optional[str] = None,
        solution: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "sqlite_id": sqlite_id,
            "kb_code": kb_code,
            "error_type": error_type,
            "severity": severity,
            "status": status,
            "language": language,
            "framework": framework,
            "error_description": error_description,
            "problematic_pattern": problematic_pattern,
            "solution": solution,
        }

    def _ensure_vector(self, props: Dict[str, Any], vector: Optional[List[float]]) -> List[float]:
        if vector is not None:
            return vector
        if not self.embed_fn:
            raise RuntimeError(
                "No vector provided and no embed_fn configured for WeaviateVectorService"
            )
        text = self._build_issue_pattern_text(props)
        return self.embed_fn(text)

    def _get_object_ids_by_sqlite_id(self, sqlite_id: int) -> List[str]:
        where_filter = {
            "path": ["sqlite_id"],
            "operator": "Equal",
            "valueInt": sqlite_id,
        }
        result = (
            self.client.query.get(self.KNOWLEDGE_CLASS, ["_additional { id }"])
            .with_where(where_filter)
            .do()
        )
        objs = result.get("data", {}).get("Get", {}).get(self.KNOWLEDGE_CLASS, [])
        ids: List[str] = []
        for obj in objs:
            uuid = obj.get("_additional", {}).get("id")
            if uuid:
                ids.append(uuid)
        return ids

    def create_knowledge_item(
        self,
        sqlite_id: int,
        kb_code: Optional[str],
        error_type: str,
        severity: str,
        status: str = "active",
        language: Optional[str] = None,
        framework: Optional[str] = None,
        error_description: Optional[str] = None,
        problematic_pattern: Optional[str] = None,
        solution: Optional[str] = None,
        vector: Optional[List[float]] = None,
    ) -> str:
        """
        在 Weaviate 中为一条 IssuePattern 创建对应的 KnowledgeItem object。

        - sqlite_id: SQLite 中 IssuePattern.id
        - 其余字段与 IssuePattern 的主要属性对应
        - vector: 如未提供且配置了 embed_fn，将使用拼接文本自动生成

        返回 Weaviate object 的 UUID。
        """
        self.ensure_knowledge_schema()

        props = self._build_props(
            sqlite_id=sqlite_id,
            kb_code=kb_code,
            error_type=error_type,
            severity=severity,
            status=status,
            language=language,
            framework=framework,
            error_description=error_description,
            problematic_pattern=problematic_pattern,
            solution=solution,
        )
        computed_vector = self._ensure_vector(props, vector)

        uuid = self.client.data_object.create(
            data_object={
                "sqlite_id": sqlite_id,
                "kb_code": kb_code,
                "error_type": error_type,
                "severity": severity,
                "status": status,
                "language": language,
                "framework": framework,
            },
            class_name=self.KNOWLEDGE_CLASS,
            vector=computed_vector,
        )
        return uuid

    def update_knowledge_item(
        self,
        sqlite_id: int,
        kb_code: Optional[str],
        error_type: str,
        severity: str,
        status: str = "active",
        language: Optional[str] = None,
        framework: Optional[str] = None,
        error_description: Optional[str] = None,
        problematic_pattern: Optional[str] = None,
        solution: Optional[str] = None,
        vector: Optional[List[float]] = None,
    ) -> bool:
        """
        根据 sqlite_id 更新现有的 KnowledgeItem。

        返回是否成功更新（未找到对象时返回 False）。
        """
        self.ensure_knowledge_schema()
        object_ids = self._get_object_ids_by_sqlite_id(sqlite_id)
        if not object_ids:
            return False

        props = self._build_props(
            sqlite_id=sqlite_id,
            kb_code=kb_code,
            error_type=error_type,
            severity=severity,
            status=status,
            language=language,
            framework=framework,
            error_description=error_description,
            problematic_pattern=problematic_pattern,
            solution=solution,
        )
        computed_vector = self._ensure_vector(props, vector)

        data_object = {
            "sqlite_id": sqlite_id,
            "kb_code": kb_code,
            "error_type": error_type,
            "severity": severity,
            "status": status,
            "language": language,
            "framework": framework,
            "error_description": error_description,
            "problematic_pattern": problematic_pattern,
            "solution": solution,
        }

        for object_id in object_ids:
            self.client.data_object.update(
                data_object=data_object,
                class_name=self.KNOWLEDGE_CLASS,
                uuid=object_id,
                vector=computed_vector,
            )
        return True

    def delete_knowledge_items_by_sqlite_id(self, sqlite_id: int) -> int:
        """
        根据 sqlite_id 删除对应的 KnowledgeItem 对象。

        返回删除的对象数量。
        """
        self.ensure_knowledge_schema()
        where_filter = {
            "path": ["sqlite_id"],
            "operator": "Equal",
            "valueInt": sqlite_id,
        }
        # 查询符合条件的对象 UUID
        result = (
            self.client.query.get(self.KNOWLEDGE_CLASS, ["_additional { id }"])
            .with_where(where_filter)
            .do()
        )
        objs = result.get("data", {}).get("Get", {}).get(self.KNOWLEDGE_CLASS, [])
        count = 0
        for obj in objs:
            uuid = obj.get("_additional", {}).get("id")
            if uuid:
                self.client.data_object.delete(uuid=uuid, class_name=self.KNOWLEDGE_CLASS)
                count += 1
        return count

    def get_knowledge_items(
        self,
        sqlite_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        查询 KnowledgeItem 对象列表。

        - 如提供 sqlite_id，则按该字段过滤
        - limit 控制返回数量
        """
        self.ensure_knowledge_schema()
        query = self.client.query.get(
            self.KNOWLEDGE_CLASS,
            [
                "sqlite_id",
                "kb_code",
                "error_type",
                "severity",
                "status",
                "language",
                "framework",
            ],
        )
        if sqlite_id is not None:
            where_filter = {
                "path": ["sqlite_id"],
                "operator": "Equal",
                "valueInt": sqlite_id,
            }
            query = query.with_where(where_filter)
        query = query.with_limit(limit)
        result = query.do()
        objs = result.get("data", {}).get("Get", {}).get(self.KNOWLEDGE_CLASS, [])
        return objs


