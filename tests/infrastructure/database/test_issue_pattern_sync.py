from pathlib import Path
import sys
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.database.weaviate.service import WeaviateVectorService
from infrastructure.database.vector_sync import (
    AgentResult,
    IssuePatternSyncService,
    KnowledgeEncodingAgent,
)
from tests.infrastructure.database.weaviate_in_memory import InMemoryWeaviateClient


class DummyEmbeddingAgent(KnowledgeEncodingAgent):
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def encode_issue_pattern(self, record: Dict[str, Any]) -> AgentResult:
        self.calls.append(record)
        text = "\n".join(f"{k}:{record.get(k) or ''}" for k in sorted(record.keys()))
        vector = [float(len(text)), 0.0, 1.0]
        return AgentResult(
            text_payload=text,
            vector=vector,
            weaviate_payload=record,
        )


def make_db_service(tmp_path: Path) -> DatabaseService:
    db_file = tmp_path / "sync.db"
    return DatabaseService(database_url=f"sqlite:///{db_file}")


def make_weaviate_service():
    svc = WeaviateVectorService(embed_fn=lambda text: [0.1, 0.2, 0.3])
    client = InMemoryWeaviateClient()
    svc.client = client
    return svc, client


@pytest.mark.asyncio
async def test_sync_issue_pattern_creates_weaviate_entry(tmp_path):
    """测试同步单条 IssuePattern 到 Weaviate（默认 full 模式）"""
    db_service = make_db_service(tmp_path)
    pattern_id = await db_service.create_issue_pattern(
        error_type="Security",
        error_description="desc",
        problematic_pattern="pattern",
        solution="solution",
        severity="high",
        language="python",
        framework="fastapi",
    )

    weaviate_service, client = make_weaviate_service()
    agent = DummyEmbeddingAgent()
    sync_service = IssuePatternSyncService(
        db_service=db_service,
        vector_service=weaviate_service,
        agent=agent,
    )

    # sync_issue_pattern 现在返回 Dict[str, str]（分层 UUID 字典）
    object_ids = await sync_service.sync_issue_pattern(pattern_id)
    assert isinstance(object_ids, dict)
    assert "full" in object_ids
    
    uuid = object_ids["full"]
    assert uuid in client.storage
    stored = client.storage[uuid]["data_object"]
    assert stored["sqlite_id"] == pattern_id
    assert stored["error_type"] == "Security"


@pytest.mark.asyncio
async def test_sync_issue_pattern_with_custom_layers(tmp_path):
    """测试同步单条 IssuePattern 使用自定义分层"""
    db_service = make_db_service(tmp_path)
    pattern_id = await db_service.create_issue_pattern(
        error_type="Performance",
        error_description="desc",
        problematic_pattern="pattern",
        solution="solution",
        severity="medium",
        language="python",
    )

    weaviate_service, client = make_weaviate_service()
    agent = DummyEmbeddingAgent()
    sync_service = IssuePatternSyncService(
        db_service=db_service,
        vector_service=weaviate_service,
        agent=agent,
    )

    # 使用自定义分层
    object_ids = await sync_service.sync_issue_pattern(
        pattern_id, 
        layers=["semantic", "full"]
    )
    assert isinstance(object_ids, dict)
    assert "semantic" in object_ids
    assert "full" in object_ids
    assert len(object_ids) == 2


@pytest.mark.asyncio
async def test_sync_all_issue_patterns_handles_multiple_records(tmp_path):
    """测试同步多条 IssuePattern 到 Weaviate"""
    db_service = make_db_service(tmp_path)
    ids = []
    for idx in range(2):
        pattern_id = await db_service.create_issue_pattern(
            error_type=f"Type{idx}",
            error_description=f"desc{idx}",
            problematic_pattern=f"pattern{idx}",
            solution=f"solution{idx}",
            severity="medium",
        )
        ids.append(pattern_id)

    weaviate_service, client = make_weaviate_service()
    agent = DummyEmbeddingAgent()
    sync_service = IssuePatternSyncService(
        db_service=db_service,
        vector_service=weaviate_service,
        agent=agent,
    )

    # sync_all_issue_patterns 现在返回 List[Dict[str, str]]
    results = await sync_service.sync_all_issue_patterns(status="active")
    assert len(results) == len(ids)
    
    for pattern_id in ids:
        items = weaviate_service.get_knowledge_items(sqlite_id=pattern_id, limit=5)
        assert len(items) >= 1
        assert items[0]["error_type"].startswith("Type")


@pytest.mark.asyncio
async def test_sync_all_issue_patterns_with_custom_layers(tmp_path):
    """测试同步多条 IssuePattern 使用自定义分层"""
    db_service = make_db_service(tmp_path)
    ids = []
    for idx in range(2):
        pattern_id = await db_service.create_issue_pattern(
            error_type=f"Type{idx}",
            error_description=f"desc{idx}",
            problematic_pattern=f"pattern{idx}",
            solution=f"solution{idx}",
            severity="medium",
        )
        ids.append(pattern_id)

    weaviate_service, client = make_weaviate_service()
    agent = DummyEmbeddingAgent()
    sync_service = IssuePatternSyncService(
        db_service=db_service,
        vector_service=weaviate_service,
        agent=agent,
    )

    # 使用自定义分层
    results = await sync_service.sync_all_issue_patterns(
        status="active",
        layers=["semantic", "code_pattern", "full"]
    )
    assert len(results) == len(ids)
    
    # 每个结果应该有 3 个分层
    for result in results:
        assert isinstance(result, dict)
        assert "semantic" in result
        assert "code_pattern" in result
        assert "full" in result
        assert len(result) == 3


@pytest.mark.asyncio
async def test_sync_issue_pattern_not_found(tmp_path):
    """测试同步不存在的 IssuePattern 应该抛出异常"""
    db_service = make_db_service(tmp_path)
    weaviate_service, _ = make_weaviate_service()
    agent = DummyEmbeddingAgent()
    sync_service = IssuePatternSyncService(
        db_service=db_service,
        vector_service=weaviate_service,
        agent=agent,
    )

    with pytest.raises(ValueError, match="not found"):
        await sync_service.sync_issue_pattern(99999)
