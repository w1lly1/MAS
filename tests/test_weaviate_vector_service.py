import types
from typing import Any, Dict, List

from unittest.mock import Mock

from infrastructure.database import WeaviateVectorService, WeaviateConfig


class FakeSchema:
    def __init__(self, classes: List[Dict[str, Any]]):
        self._classes = classes
        self.create_class = Mock()

    def get(self) -> Dict[str, Any]:
        return {"classes": self._classes}


class FakeQuery:
    def __init__(self, result: Dict[str, Any]):
        self._result = result
        self.with_where = Mock(return_value=self)
        self.with_limit = Mock(return_value=self)

    def do(self) -> Dict[str, Any]:
        return self._result


class FakeDataObject:
    def __init__(self):
        self.create = Mock(return_value="fake-uuid")
        self.delete = Mock()


def make_service_with_fake_client(result_for_query: Dict[str, Any]):
    """
    构造一个带有 fake client 的 WeaviateVectorService 实例，
    避免在测试中真正访问网络。
    """
    cfg = WeaviateConfig(url="http://fake", api_key=None, timeout=5)
    svc = WeaviateVectorService(config=cfg, embed_fn=lambda text: [0.1, 0.2, 0.3])

    fake_schema = FakeSchema(classes=[])
    fake_query = types.SimpleNamespace(
        get=Mock(return_value=FakeQuery(result_for_query))
    )
    fake_data_object = FakeDataObject()

    svc.client = types.SimpleNamespace(
        schema=fake_schema,
        query=fake_query,
        data_object=fake_data_object,
    )
    return svc


def test_ensure_knowledge_schema_creates_class_when_missing():
    # 初始 schema 中没有 KnowledgeItem，应创建
    svc = make_service_with_fake_client(result_for_query={"data": {"Get": {}}})

    # 覆盖 schema.get 让其返回不含 KnowledgeItem 的 classes
    svc.client.schema._classes = [{"class": "OtherClass"}]
    svc.ensure_knowledge_schema()

    svc.client.schema.create_class.assert_called_once()
    args, _ = svc.client.schema.create_class.call_args
    class_obj = args[0]
    assert class_obj["class"] == svc.KNOWLEDGE_CLASS


def test_create_knowledge_item_with_explicit_vector():
    # schema 中已经存在 KnowledgeItem，不应再次创建
    result = {
        "data": {
            "Get": {
                "KnowledgeItem": [],
            }
        }
    }
    svc = make_service_with_fake_client(result_for_query=result)
    svc.client.schema._classes = [{"class": svc.KNOWLEDGE_CLASS}]

    uuid = svc.create_knowledge_item(
        sqlite_id=1,
        kb_code="KB-TEST",
        error_type="SQLInjection",
        severity="high",
        status="active",
        language="python",
        framework="django",
        error_description="desc",
        problematic_pattern="pattern",
        solution="solution",
        vector=[0.5, 0.6, 0.7],
    )

    assert uuid == "fake-uuid"
    # 确认 create 被调用且使用了提供的向量
    svc.client.data_object.create.assert_called_once()
    _, kwargs = svc.client.data_object.create.call_args
    assert kwargs["class_name"] == svc.KNOWLEDGE_CLASS
    assert kwargs["vector"] == [0.5, 0.6, 0.7]


def test_delete_knowledge_items_by_sqlite_id():
    # 预设查询返回两个对象，delete 应该被调用两次
    query_result = {
        "data": {
            "Get": {
                "KnowledgeItem": [
                    {"_additional": {"id": "uuid-1"}},
                    {"_additional": {"id": "uuid-2"}},
                ]
            }
        }
    }
    svc = make_service_with_fake_client(result_for_query=query_result)
    svc.client.schema._classes = [{"class": svc.KNOWLEDGE_CLASS}]

    deleted = svc.delete_knowledge_items_by_sqlite_id(sqlite_id=123)
    assert deleted == 2
    # 确认 delete 调用两次
    assert svc.client.data_object.delete.call_count == 2


def test_get_knowledge_items_with_and_without_filter():
    # 准备一个包含单个 KnowledgeItem 的查询结果
    query_result = {
        "data": {
            "Get": {
                "KnowledgeItem": [
                    {
                        "sqlite_id": 10,
                        "kb_code": "KB-001",
                        "error_type": "Perf",
                        "severity": "medium",
                        "status": "active",
                        "language": "python",
                        "framework": "fastapi",
                    }
                ]
            }
        }
    }
    svc = make_service_with_fake_client(result_for_query=query_result)
    svc.client.schema._classes = [{"class": svc.KNOWLEDGE_CLASS}]

    # 带 sqlite_id 过滤
    items = svc.get_knowledge_items(sqlite_id=10, limit=5)
    assert len(items) == 1
    assert items[0]["sqlite_id"] == 10
    # 验证过滤方法被调用
    # get() 返回的 FakeQuery 已经包装了 with_where/with_limit 的调用计数
    q = svc.client.query.get.return_value
    q.with_where.assert_called_once()
    q.with_limit.assert_called_once()


