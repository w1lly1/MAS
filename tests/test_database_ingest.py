import pytest
from unittest.mock import AsyncMock, Mock

from utils.database_ingest import DatabaseIngestTool


@pytest.mark.asyncio
async def test_database_ingest_reuses_existing_weaviate_layers():
    tool = DatabaseIngestTool.__new__(DatabaseIngestTool)
    tool.weaviate_layers = ["semantic", "code_pattern", "solution", "full"]
    tool.db_service = Mock()
    tool.sync_service = Mock()
    tool.sync_service.sync_issue_pattern = AsyncMock(return_value={"semantic": "ok"})

    tool._find_existing_pattern_id = AsyncMock(return_value=None)
    tool._find_existing_curated_issue_id = AsyncMock(return_value=None)

    tool.db_service.create_issue_pattern = AsyncMock(return_value=101)
    tool.db_service.get_review_session_by_session_id = AsyncMock(return_value=None)
    tool.db_service.create_review_session = AsyncMock(return_value=201)
    tool.db_service.get_session = Mock()

    entry = {
        "pattern": {
            "title": "CVE-TEST-1",
            "error_type": "dos",
            "severity": "medium",
            "language": "C",
            "framework": "linux",
            "error_description": "demo",
            "problematic_pattern": "demo",
            "solution": "demo",
            "file_pattern": "",
            "class_pattern": "",
            "tags": "DoS",
            "status": "active",
        },
        "instances": [],
    }

    await tool._process_entry(entry)

    tool.sync_service.sync_issue_pattern.assert_awaited_once_with(
        101,
        layers=["semantic", "code_pattern", "solution", "full"],
    )
