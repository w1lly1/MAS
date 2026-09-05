from __future__ import annotations

import logging
import os
import uuid as uuid_lib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import weaviate
from weaviate import WeaviateClient
from weaviate.exceptions import WeaviateConnectionError
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter

EmbedFn = Callable[[str], List[float]]
logger = logging.getLogger(__name__)


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
    grpc_port: int = 50051

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
        grpc_port_str = os.getenv("WEAVIATE_GRPC_PORT", "50051")
        
        # 自动补全协议前缀
        if url and not url.startswith(("http://", "https://")):
            # 如果有 API Key 或者是云端域名，使用 https
            if api_key or ".weaviate.cloud" in url:
                url = f"https://{url}"
            else:
                url = f"http://{url}"
        
        try:
            timeout = int(timeout_str)
        except ValueError:
            timeout = 30
        try:
            grpc_port = int(grpc_port_str)
        except ValueError:
            grpc_port = 50051
        return cls(url=url, api_key=api_key, timeout=timeout, grpc_port=grpc_port)


class WeaviateVectorService:
    """
    Weaviate 向量索引服务（兼容 weaviate-client v4）。

    当前主要面向 IssuePattern（知识模式），使用 Weaviate 中的 "KnowledgeItem" collection：
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
        # - 也可以调用 connect() 方法自动连接
        self.client: Optional[WeaviateClient] = None
        self._connection_attempted = False
        self._connection_error: Optional[str] = None

    def connect(self, auto_create_schema: bool = True) -> bool:
        """
        连接到 Weaviate 实例。
        
        Args:
            auto_create_schema: 连接成功后是否自动创建 schema
            
        Returns:
            True 如果连接成功，False 如果连接失败
        """
        if self.client is not None:
            # 如果已有连接，先尝试断开再重新连接（处理程序重启/热重载场景）
            try:
                logger.info("🔁 检测到已有 Weaviate 连接，断开旧连接")
                self.disconnect()
            except Exception as e:
                logger.warning(f"⚠️ 关闭旧 Weaviate 连接时出错: {e}")
        
        if self._connection_attempted:
            # 已经尝试过连接但失败了，避免重复尝试
            return False
        
        self._connection_attempted = True
        
        try:
            parsed = urlparse(self.config.url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8080
            is_secure = parsed.scheme == "https"
            
            logger.info(f"🔗 正在连接 Weaviate: {self.config.url}")
            
            # 尝试连接到 Weaviate
            if host in ("localhost", "127.0.0.1") and not is_secure:
                # 本地连接
                self.client = weaviate.connect_to_local(
                    host=host,
                    port=port,
                    grpc_port=self.config.grpc_port,
                )
            elif ".weaviate.cloud" in host:
                # Weaviate Cloud 连接
                auth_credentials = None
                if self.config.api_key:
                    auth_credentials = weaviate.auth.AuthApiKey(self.config.api_key)
                
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.config.url,
                    auth_credentials=auth_credentials,
                    skip_init_checks=True,  # 跳过 gRPC 健康检查，解决防火墙问题
                )
            else:
                # 自定义连接（支持远程自建实例）
                grpc_port = self.config.grpc_port
                
                auth_credentials = None
                if self.config.api_key:
                    auth_credentials = weaviate.auth.AuthApiKey(self.config.api_key)
                
                self.client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=is_secure,
                    grpc_host=host,
                    grpc_port=grpc_port,
                    grpc_secure=is_secure,
                    auth_credentials=auth_credentials,
                    skip_init_checks=True,
                )
            
            # 验证连接
            if self.client.is_ready():
                logger.info(f"✅ Weaviate 连接成功: {self.config.url}")
                
                # 自动创建 schema
                if auto_create_schema:
                    try:
                        self.ensure_knowledge_schema()
                        logger.info("✅ Weaviate schema 已就绪")
                    except Exception as e:
                        logger.warning(f"⚠️ 创建 schema 失败: {e}")
                
                return True
            else:
                self._connection_error = "Weaviate 实例未就绪"
                logger.warning(f"⚠️ {self._connection_error}")
                self.client = None
                return False
                
        except WeaviateConnectionError as e:
            self._connection_error = f"连接失败: {e}"
            logger.warning(f"⚠️ Weaviate {self._connection_error}")
            self.client = None
            return False
        except Exception as e:
            self._connection_error = f"连接异常: {e}"
            logger.warning(f"⚠️ Weaviate {self._connection_error}")
            self.client = None
            return False

    def disconnect(self) -> None:
        """断开 Weaviate 连接"""
        if self.client is not None:
            try:
                self.client.close()
                logger.info("🔌 Weaviate 连接已断开")
            except Exception as e:
                logger.warning(f"⚠️ 断开连接时出错: {e}")
            finally:
                self.client = None
                self._connection_attempted = False
                self._connection_error = None

    def is_connected(self) -> bool:
        """检查是否已连接到 Weaviate"""
        return self.client is not None

    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态信息"""
        return {
            "connected": self.is_connected(),
            "url": self.config.url,
            "attempted": self._connection_attempted,
            "error": self._connection_error,
        }

    # --------------------------------------------------------------------- #
    # schema 管理 (v4 API)
    # --------------------------------------------------------------------- #
    def ensure_knowledge_schema(self) -> None:
        """
        确保 KnowledgeItem collection 存在，如不存在则创建。

        Schema 设计（v4 API）：
        - collection: KnowledgeItem
        - properties:
          - sqlite_id: int
          - error_type: text
          - severity: text
          - status: text
          - language: text
          - framework: text
                    - error_description: text
                    - problematic_pattern: text
                    - solution: text
                    - file_pattern: text
                    - class_pattern: text
                    - layer_text: text
          - vector_layer: text
        - vectorizer: none （向量由外部传入）
        """
        # v4 API: 使用 collections.exists() 检查
        if self.client.collections.exists(self.KNOWLEDGE_CLASS):
            return

        # 创建 collection（v4 API）
        # 不指定 vectorizer，因为我们手动传入向量
        self.client.collections.create(
            name=self.KNOWLEDGE_CLASS,
            description="IssuePattern knowledge items used for semantic search",
            properties=[
                Property(name="sqlite_id", data_type=DataType.INT),
                Property(name="error_type", data_type=DataType.TEXT),
                Property(name="severity", data_type=DataType.TEXT),
                Property(name="status", data_type=DataType.TEXT),
                Property(name="language", data_type=DataType.TEXT),
                Property(name="framework", data_type=DataType.TEXT),
                Property(name="error_description", data_type=DataType.TEXT),
                Property(name="problematic_pattern", data_type=DataType.TEXT),
                Property(name="solution", data_type=DataType.TEXT),
                Property(name="file_pattern", data_type=DataType.TEXT),
                Property(name="class_pattern", data_type=DataType.TEXT),
                Property(name="layer_text", data_type=DataType.TEXT),
                Property(name="vector_layer", data_type=DataType.TEXT),
            ],
        )

    def _get_collection(self):
        """获取 KnowledgeItem collection 对象"""
        return self.client.collections.get(self.KNOWLEDGE_CLASS)

    # --------------------------------------------------------------------- #
    # IssuePattern / KnowledgeItem CRUD + 向量写入 (v4 API)
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
        layer_text: Optional[str] = None,
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
            "layer_text": layer_text,
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
        """根据 sqlite_id 获取所有对象的 UUID（v4 API）"""
        collection = self._get_collection()
        result = collection.query.fetch_objects(
            filters=Filter.by_property("sqlite_id").equal(sqlite_id),
            limit=100,
        )
        return [str(obj.uuid) for obj in result.objects]

    def _get_object_ids_by_sqlite_id_and_layer(self, sqlite_id: int, layer: str) -> List[str]:
        """根据 sqlite_id 和 vector_layer 获取所有对象的 UUID（v4 API）"""
        collection = self._get_collection()
        result = collection.query.fetch_objects(
            filters=Filter.by_property("sqlite_id").equal(sqlite_id),
            limit=100,
        )
        object_ids: List[str] = []
        for obj in result.objects:
            if obj.properties.get("vector_layer") == layer:
                object_ids.append(str(obj.uuid))
        return object_ids

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
        在 Weaviate 中为一条 IssuePattern 创建对应的 KnowledgeItem object（v4 API）。
        
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
            layer_text=None,
        )
        computed_vector = self._ensure_vector_with_layer(props, vector, layer)
        layer_text = self._build_enhanced_issue_pattern_text(props, layer)

        self.delete_knowledge_items_by_sqlite_id_and_layer(sqlite_id, layer)

        # v4 API: 使用 collection.data.insert()
        collection = self._get_collection()
        obj_uuid = collection.data.insert(
            properties={
                "sqlite_id": sqlite_id,
                "error_type": error_type or "",
                "severity": severity or "",
                "status": status or "",
                "language": language or "",
                "framework": framework or "",
                "error_description": error_description or "",
                "problematic_pattern": problematic_pattern or "",
                "solution": solution or "",
                "file_pattern": file_pattern or "",
                "class_pattern": class_pattern or "",
                "layer_text": layer_text,
                "vector_layer": layer,
            },
            vector=computed_vector,
        )
        return str(obj_uuid)

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
        layer_texts: Optional[Dict[str, str]] = None,
        layer_payloads: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """
        创建知识项并支持多向量分层存储（v4 API）
        
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

        if layer_texts is None:
            layer_texts = {
                layer: self._build_enhanced_issue_pattern_text(props, layer)
                for layer in vectors.keys()
            }

        if layer_payloads is None:
            layer_payloads = {}
            for layer in vectors.keys():
                layer_payloads[layer] = self._build_layer_payload(
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
                    layer=layer,
                    layer_text=layer_texts.get(layer) or self._build_enhanced_issue_pattern_text(props, layer),
                )
        
        # v4 API: 为每个分层创建独立的对象
        collection = self._get_collection()
        object_ids = {}
        for layer, vector in vectors.items():
            self.delete_knowledge_items_by_sqlite_id_and_layer(sqlite_id, layer)
            payload_for_layer = layer_payloads.get(layer) or self._build_layer_payload(
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
                layer=layer,
                layer_text=layer_texts.get(layer) or self._build_enhanced_issue_pattern_text(props, layer),
            )
            obj_uuid = collection.data.insert(
                properties=payload_for_layer,
                vector=vector,
            )
            object_ids[layer] = str(obj_uuid)
        
        return object_ids

    def get_knowledge_item(self, uuid: str) -> Optional[Dict[str, Any]]:
        """根据 UUID 获取单个知识项（v4 API）"""
        try:
            collection = self._get_collection()
            obj = collection.query.fetch_object_by_id(uuid_lib.UUID(uuid))
            if obj:
                return obj.properties
            return None
        except Exception as e:
            logger.error(f"Failed to get knowledge item {uuid}: {e}")
            return None

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
        更新 Weaviate 中的 KnowledgeItem object（v4 API）。
        
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
            # v4 API: 使用 collection.data.update()
            collection = self._get_collection()
            collection.data.update(
                uuid=uuid_lib.UUID(uuid),
                properties=update_data,
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
        基于向量搜索知识项，支持分层搜索（v4 API）
        
        Args:
            query_vector: 查询向量
            limit: 返回结果数量
            layer: 分层策略，可选值："semantic" | "code_pattern" | "solution" | "full"
            additional_filters: 额外的过滤条件
        
        Returns:
            搜索结果列表
        """
        try:
            collection = self._get_collection()
            
            # 构建过滤器：始终按 layer 过滤，确保传入 layer 时只返回该 layer 的对象
            filters = Filter.by_property("vector_layer").equal(layer)
            
            # v4 API: 使用 near_vector 查询
            result = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=filters,
                return_metadata=wvc.query.MetadataQuery(distance=True),
            )

            items = []
            for obj in result.objects:
                item = dict(obj.properties)
                if obj.metadata and obj.metadata.distance is not None:
                    item["_additional"] = {"distance": obj.metadata.distance}
                items.append(item)
            
            return items

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_knowledge_items_bm25(
        self,
        query_text: str,
        limit: int = 10,
        layer: str = "full",
        query_properties: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        基于 BM25 的关键词稀疏检索（v4 API）。

        BM25 是词项级精确匹配 + 词频/逆文档频率加权，属于"稀疏精确匹配"，
        与向量相似度（稠密模糊）互补。返回值结构与 near_vector 一致，
        但 `_additional` 中携带的是 `score`（越高越相关），而非 `distance`。

        Args:
            query_text: 查询文本（通常是错误代码 / 描述）。
            limit: 返回结果数量。
            layer: 分层过滤（semantic | code_pattern | solution | full）。
            query_properties: 参与 BM25 检索的属性名列表；None 表示所有可检索文本属性。
        """
        try:
            collection = self._get_collection()
            filters = Filter.by_property("vector_layer").equal(layer)
            result = collection.query.bm25(
                query=query_text,
                query_properties=query_properties,
                limit=limit,
                filters=filters,
                return_metadata=wvc.query.MetadataQuery(score=True),
            )
            items = []
            for obj in result.objects:
                item = dict(obj.properties)
                if obj.metadata and obj.metadata.score is not None:
                    item["_additional"] = {"score": obj.metadata.score}
                items.append(item)
            return items
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def search_knowledge_items_hybrid(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        alpha: float = 0.5,
        limit: int = 10,
        layer: str = "full",
        query_properties: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        混合检索：BM25 关键词 + 向量相似度融合（v4 API）。

        alpha 控制 BM25 与向量的权重：
        - alpha=0：纯 BM25
        - alpha=1：纯向量
        - 0<alpha<1：两者按 RRF/ranked 融合（默认 RANKED fusion）

        返回结构与 bm25 一致，`_additional` 中携带融合后的 `score`。

        Args:
            query_text: 查询文本（供 BM25 使用）。
            query_vector: 查询向量（供向量部分使用）；None 则用 embed_fn 生成。
            alpha: BM25 权重（1-alpha 为向量权重）。
            limit: 返回结果数量。
            layer: 分层过滤。
            query_properties: 参与 BM25 检索的属性名列表。
        """
        try:
            collection = self._get_collection()
            filters = Filter.by_property("vector_layer").equal(layer)
            result = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=alpha,
                query_properties=query_properties,
                limit=limit,
                filters=filters,
                return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
            )
            items = []
            for obj in result.objects:
                item = dict(obj.properties)
                if obj.metadata and obj.metadata.score is not None:
                    item["_additional"] = {"score": obj.metadata.score}
                items.append(item)
            return items
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def delete_knowledge_items_by_sqlite_id(self, sqlite_id: int) -> int:
        """
        根据 sqlite_id 删除对应的 KnowledgeItem 对象（v4 API）。

        返回删除的对象数量。
        """
        self.ensure_knowledge_schema()
        
        # 获取所有匹配的对象 UUID
        object_ids = self._get_object_ids_by_sqlite_id(sqlite_id)
        
        if not object_ids:
            return 0
        
        # v4 API: 逐个删除
        collection = self._get_collection()
        count = 0
        for obj_id in object_ids:
            try:
                collection.data.delete_by_id(uuid_lib.UUID(obj_id))
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete object {obj_id}: {e}")
        
        return count

    def delete_knowledge_items_by_sqlite_id_and_layer(self, sqlite_id: int, layer: str) -> int:
        """
        根据 sqlite_id + vector_layer 删除对应的 KnowledgeItem 对象（v4 API）。

        返回删除的对象数量。
        """
        self.ensure_knowledge_schema()
        collection = self._get_collection()
        count = 0
        while True:
            object_ids = self._get_object_ids_by_sqlite_id_and_layer(sqlite_id, layer)
            if not object_ids:
                break

            deleted_this_round = 0
            for obj_id in object_ids:
                try:
                    collection.data.delete_by_id(uuid_lib.UUID(obj_id))
                    deleted_this_round += 1
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete object {obj_id} for layer {layer}: {e}")

            if deleted_this_round == 0:
                break

        return count

    def _build_layer_payload(
        self,
        sqlite_id: int,
        error_type: str,
        severity: str,
        status: str,
        language: Optional[str],
        framework: Optional[str],
        error_description: Optional[str],
        problematic_pattern: Optional[str],
        solution: Optional[str],
        file_pattern: Optional[str],
        class_pattern: Optional[str],
        layer: str,
        layer_text: str,
    ) -> Dict[str, Any]:
        base = {
            "sqlite_id": sqlite_id,
            "error_type": error_type or "",
            "severity": severity or "",
            "status": status or "",
            "language": language or "",
            "framework": framework or "",
            "error_description": error_description or "",
            "problematic_pattern": problematic_pattern or "",
            "solution": solution or "",
            "file_pattern": file_pattern or "",
            "class_pattern": class_pattern or "",
            "layer_text": layer_text,
            "vector_layer": layer,
        }

        if layer == "semantic":
            return {
                **base,
                "problematic_pattern": "",
                "solution": "",
                "file_pattern": "",
                "class_pattern": "",
            }
        if layer == "code_pattern":
            return {
                **base,
                "error_description": "",
                "solution": "",
            }
        if layer == "solution":
            return {
                **base,
                "problematic_pattern": "",
                "file_pattern": "",
                "class_pattern": "",
            }
        return base

    def delete_collection(self) -> bool:
        """删除 KnowledgeItem collection（用于换向量维度后重建）。"""
        try:
            if self.client is not None and self.client.collections.exists(self.KNOWLEDGE_CLASS):
                self.client.collections.delete(self.KNOWLEDGE_CLASS)
                logger.info(f"🗑️ 已删除 collection: {self.KNOWLEDGE_CLASS}")
                return True
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to delete collection: {e}")
            return False

    def delete_all_knowledge_items(self) -> int:
        """
        删除 KnowledgeItem collection 中的所有对象（批量删除，v4 API）。

        返回删除的对象数量。
        """
        try:
            self.ensure_knowledge_schema()
            collection = self._get_collection()
            # 分页获取所有对象 UUID 并逐个删除以兼容 v4 API
            total_deleted = 0
            batch_size = 100
            while True:
                result = collection.query.fetch_objects(limit=batch_size)
                objs = result.objects
                if not objs:
                    break
                for obj in objs:
                    try:
                        collection.data.delete_by_id(uuid_lib.UUID(str(obj.uuid)))
                        total_deleted += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete object {obj.uuid}: {e}")
                # loop until no objects left
            return total_deleted
        except Exception as e:
            logger.warning(f"Failed to delete all knowledge items: {e}")
            return 0

    def get_knowledge_items(
        self,
        sqlite_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        查询 KnowledgeItem 对象列表（v4 API）。

        - 如提供 sqlite_id，则按该字段过滤
        - limit 控制返回数量
        """
        self.ensure_knowledge_schema()
        
        collection = self._get_collection()
        
        # 构建过滤器
        filters = None
        if sqlite_id is not None:
            filters = Filter.by_property("sqlite_id").equal(sqlite_id)
        
        # v4 API: 使用 fetch_objects
        result = collection.query.fetch_objects(
            filters=filters,
            limit=limit,
        )
        
        return [dict(obj.properties) for obj in result.objects]
