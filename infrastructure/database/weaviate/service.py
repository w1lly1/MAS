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
    - error_type, severity, status 等用于过滤
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
    def _build_semantic_layer_text(self, props: Dict[str, Any]) -> str:
        """
        构建语义层文本：专注于问题的本质特征和语义含义
        """
        parts = [
            f"[error_type] {props.get('error_type') or ''}",
            f"[severity] {props.get('severity') or ''}",
            f"[language] {props.get('language') or ''}",
            f"[framework] {props.get('framework') or ''}",
            f"[description] {props.get('error_description') or ''}",
        ]
        return "\n".join(parts)

    def _build_code_pattern_layer_text(self, props: Dict[str, Any]) -> str:
        """
        构建代码模式层文本：专注于代码实现特征和结构模式
        """
        parts = [
            f"[problematic_pattern] {props.get('problematic_pattern') or ''}",
            f"[file_pattern] {props.get('file_pattern') or ''}",
            f"[class_pattern] {props.get('class_pattern') or ''}",
            f"[language] {props.get('language') or ''}",
        ]
        return "\n".join(parts)

    def _build_solution_layer_text(self, props: Dict[str, Any]) -> str:
        """
        构建解决方案层文本：专注于修复策略和实施方法
        """
        parts = [
            f"[solution] {props.get('solution') or ''}",
            f"[error_description] {props.get('error_description') or ''}",
            f"[severity] {props.get('severity') or ''}",
        ]
        return "\n".join(parts)

    def _build_full_layer_text(self, props: Dict[str, Any]) -> str:
        """
        构建完整层文本：包含所有可用信息的完整上下文
        """
        parts = [
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
        return "\n".join(parts)

    def _build_props(
        self,
        sqlite_id: int,
        error_type: str,
        severity: str,
        status: str = "active",
        language: Optional[str] = None,
        framework: Optional[str] = None,
        error_description: Optional[str] = None,
        problematic_pattern: Optional[str] = None,
        solution: Optional[str] = None,
        file_pattern: Optional[str] = None,
        class_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "sqlite_id": sqlite_id,
            "error_type": error_type,
            "severity": severity,
            "status": status,
            "language": language,
            "framework": framework,
            "error_description": error_description,
            "problematic_pattern": problematic_pattern,
            "solution": solution,
            "file_pattern": file_pattern,
            "class_pattern": class_pattern,
        }

    def _build_enhanced_issue_pattern_text(self, props: Dict[str, Any], layer: str = "full") -> str:
        """
        增强的文本构建方法：支持四种分层策略
        
        Args:
            props: 属性字典
            layer: 分层策略，可选值："semantic" | "code_pattern" | "solution" | "full"
        """
        layer_mapping = {
            "semantic": self._build_semantic_layer_text,
            "code_pattern": self._build_code_pattern_layer_text,
            "solution": self._build_solution_layer_text,
            "full": self._build_full_layer_text,
        }
        
        if layer not in layer_mapping:
            raise ValueError(f"不支持的层类型: {layer}，支持的类型: {list(layer_mapping.keys())}")
        
        return layer_mapping[layer](props)

    def _ensure_vector_with_layer(self, props: Dict[str, Any], vector: Optional[List[float]], 
                                 layer: str = "full") -> List[float]:
        """
        增强的向量生成方法：支持分层策略
        """
        if vector is not None:
            return vector
        if not self.embed_fn:
            raise RuntimeError(
                "No vector provided and no embed_fn configured for WeaviateVectorService"
            )
        text = self._build_enhanced_issue_pattern_text(props, layer)
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
        error_type: str,
        severity: str,
        status: str = "active",
        language: Optional[str] = None,
        framework: Optional[str] = None,
        error_description: Optional[str] = None,
        problematic_pattern: Optional[str] = None,
        solution: Optional[str] = None,
        file_pattern: Optional[str] = None,
        class_pattern: Optional[str] = None,
        vector: Optional[List[float]] = None,
        layer: str = "full",
    ) -> str:
        """
        在 Weaviate 中为一条 IssuePattern 创建对应的 KnowledgeItem object。
        
        - layer: 分层策略，可选值："semantic" | "code_pattern" | "solution" | "full"
        """
        self.ensure_knowledge_schema()

        props = self._build_props(
            sqlite_id=sqlite_id,
            error_type=error_type,
            severity=severity,
            status=status,
            language=language,
            framework=framework,
            error_description=error_description,
            problematic_pattern=problematic_pattern,
            solution=solution,
            file_pattern=file_pattern,
            class_pattern=class_pattern,
        )
        computed_vector = self._ensure_vector_with_layer(props, vector, layer)

        uuid = self.client.data_object.create(
            data_object={
                "sqlite_id": sqlite_id,
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

    def create_knowledge_item_with_layered_vectors(
        self,
        sqlite_id: int,
        error_type: str,
        severity: str,
        status: str = "active",
        language: Optional[str] = None,
        framework: Optional[str] = None,
        error_description: Optional[str] = None,
        problematic_pattern: Optional[str] = None,
        solution: Optional[str] = None,
        file_pattern: Optional[str] = None,
        class_pattern: Optional[str] = None,
        vectors: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, str]:
        """
        创建知识项并支持多向量分层存储
        
        Args:
            vectors: 分层向量字典，key为分层类型，value为向量
        
        Returns:
            各分层对应的对象ID字典
        """
        self.ensure_knowledge_schema()
        
        props = self._build_props(
            sqlite_id=sqlite_id,
            error_type=error_type,
            severity=severity,
            status=status,
            language=language,
            framework=framework,
            error_description=error_description,
            problematic_pattern=problematic_pattern,
            solution=solution,
            file_pattern=file_pattern,
            class_pattern=class_pattern,
        )
        
        # 如果没有提供向量，为所有分层生成向量
        if vectors is None:
            vectors = {}
            for layer in ["semantic", "code_pattern", "solution", "full"]:
                vectors[layer] = self._ensure_vector_with_layer(props, None, layer)
        
        # 为每个分层创建独立的对象
        object_ids = {}
        for layer, vector in vectors.items():
            # 为每个分层添加分层标识
            data_object = {
                "sqlite_id": sqlite_id,
                "error_type": error_type,
                "severity": severity,
                "status": status,
                "language": language,
                "framework": framework,
                "vector_layer": layer,  # 添加分层标识
            }
            
            uuid = self.client.data_object.create(
                data_object=data_object,
                class_name=self.KNOWLEDGE_CLASS,
                vector=vector,
            )
            object_ids[layer] = uuid
        
        return object_ids

    def update_knowledge_item(
        self,
        uuid: str,
        sqlite_id: Optional[int] = None,
        error_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        language: Optional[str] = None,
        framework: Optional[str] = None,
        error_description: Optional[str] = None,
        problematic_pattern: Optional[str] = None,
        solution: Optional[str] = None,
        file_pattern: Optional[str] = None,
        class_pattern: Optional[str] = None,
        vector: Optional[List[float]] = None,
        layer: str = "full",
    ) -> bool:
        """
        更新 Weaviate 中的 KnowledgeItem object。
        
        - layer: 分层策略，可选值："semantic" | "code_pattern" | "solution" | "full"
        """
        # 获取现有对象以构建完整属性
        existing = self.get_knowledge_item(uuid)
        if not existing:
            return False

        # 构建更新后的属性
        props = self._build_props(
            sqlite_id=sqlite_id if sqlite_id is not None else existing.get("sqlite_id"),
            error_type=error_type if error_type is not None else existing.get("error_type"),
            severity=severity if severity is not None else existing.get("severity"),
            status=status if status is not None else existing.get("status"),
            language=language if language is not None else existing.get("language"),
            framework=framework if framework is not None else existing.get("framework"),
            error_description=error_description if error_description is not None else existing.get("error_description"),
            problematic_pattern=problematic_pattern if problematic_pattern is not None else existing.get("problematic_pattern"),
            solution=solution if solution is not None else existing.get("solution"),
            file_pattern=file_pattern if file_pattern is not None else existing.get("file_pattern"),
            class_pattern=class_pattern if class_pattern is not None else existing.get("class_pattern"),
        )

        # 计算新的向量
        computed_vector = self._ensure_vector_with_layer(props, vector, layer)

        # 构建更新数据
        update_data = {}
        if sqlite_id is not None:
            update_data["sqlite_id"] = sqlite_id
        if error_type is not None:
            update_data["error_type"] = error_type
        if severity is not None:
            update_data["severity"] = severity
        if status is not None:
            update_data["status"] = status
        if language is not None:
            update_data["language"] = language
        if framework is not None:
            update_data["framework"] = framework

        try:
            self.client.data_object.update(
                uuid=uuid,
                data_object=update_data,
                class_name=self.KNOWLEDGE_CLASS,
                vector=computed_vector,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update knowledge item {uuid}: {e}")
            return False

    def search_knowledge_items(
        self,
        query_vector: List[float],
        limit: int = 10,
        layer: str = "full",
        additional_filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        基于向量搜索知识项，支持分层搜索
        
        Args:
            query_vector: 查询向量
            limit: 返回结果数量
            layer: 分层策略，可选值："semantic" | "code_pattern" | "solution" | "full"
            additional_filters: 额外的过滤条件
        
        Returns:
            搜索结果列表
        """
        # 构建基础查询
        query = {
            "class": self.KNOWLEDGE_CLASS,
            "vector": query_vector,
            "limit": limit,
            "with_additional": ["distance"],
        }

        # 添加分层过滤条件
        if layer != "full":
            if additional_filters is None:
                additional_filters = {}
            additional_filters["vector_layer"] = layer

        # 添加过滤条件
        if additional_filters:
            where_clause = {"operator": "And", "operands": []}
            for key, value in additional_filters.items():
                where_clause["operands"].append({
                    "path": [key],
                    "operator": "Equal",
                    "valueString": value
                })
            query["where"] = where_clause

        try:
            result = self.client.query.get(
                self.KNOWLEDGE_CLASS,
                ["sqlite_id", "error_type", "severity", "status",
                 "language", "framework", "vector_layer"]
            ).with_near_vector({
                "vector": query_vector,
                "distance": 0.6
            }).with_limit(limit).with_additional(["distance"]).do()

            if "errors" in result:
                logger.error(f"Search error: {result['errors']}")
                return []

            items = result.get("data", {}).get("Get", {}).get(self.KNOWLEDGE_CLASS, [])
            return items

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

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