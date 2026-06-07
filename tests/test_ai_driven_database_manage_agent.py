from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agents.ai_driven_database_manage_agent import AIDrivenDatabaseManageAgent


@pytest.mark.asyncio
async def test_get_knowledge_from_database_returns_weaviate_items():
    agent = AIDrivenDatabaseManageAgent.__new__(AIDrivenDatabaseManageAgent)
    agent.vector_service = Mock()
    agent.vector_service.client = object()
    agent.vector_service.get_knowledge_items = Mock(
        return_value=[
            {"sqlite_id": 1, "vector_layer": "semantic", "error_type": "dos"},
            {"sqlite_id": 1, "vector_layer": "code_pattern", "error_type": "dos"},
        ]
    )

    result = await agent.get_knowledge_from_database({})

    assert result["status"] == "success"
    assert result["count"] == 2
    assert len(result["knowledge_data"]) == 2
    assert set(result["grouped_by_layer"].keys()) == {"semantic", "code_pattern"}


@pytest.mark.asyncio
async def test_delete_all_issue_patterns_clears_weaviate_collection():
    agent = AIDrivenDatabaseManageAgent.__new__(AIDrivenDatabaseManageAgent)
    agent.db_service = Mock()
    agent.db_service.get_issue_patterns = AsyncMock(return_value=[{"id": 1}, {"id": 2}])
    agent.db_service.delete_all_issue_patterns = AsyncMock(return_value=2)
    agent.vector_service = Mock()
    agent.vector_service.client = object()
    agent.vector_service.delete_all_knowledge_items = Mock(return_value=11)
    agent._delete_weaviate_items = Mock(return_value=4)

    result = await agent._handle_issue_pattern_task(
        "delete_all",
        {"confirm": True},
    )

    assert result["deleted_count"] == 2
    assert result["weaviate_deleted"] == 19
    agent.vector_service.delete_all_knowledge_items.assert_called_once()
