import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.infrastructure.database.mock_embedding_agent import (
    MockDBEmbeddingAgent,
)


def fake_embed_factory(collected: List[str]):
    def _embed(text: str) -> List[float]:
        collected.append(text)
        return [float(len(text)), 1.0, 2.0]

    return _embed


def make_record(**overrides: Any) -> Dict[str, Any]:
    record = {
        "sqlite_id": 101,
        "kb_code": "KB-AGENT-01",
        "error_type": "SQLInjection",
        "severity": "high",
        "status": "active",
        "language": "python",
        "framework": "fastapi",
        "error_description": "User input is concatenated directly into SQL.",
        "problematic_pattern": "f\"SELECT * FROM users WHERE name = '{name}'\"",
        "solution": "Use parameterized queries.",
    }
    record.update(overrides)
    return record


def test_mock_agent_builds_text_and_vector():
    captured_texts: List[str] = []
    agent = MockDBEmbeddingAgent(embed_fn=fake_embed_factory(captured_texts))

    record = make_record()
    result = agent.encode_issue_pattern(record)

    # 确认 embed 函数被调用且返回值被透传
    assert len(captured_texts) == 1
    assert result.text_payload == captured_texts[0]
    assert result.vector == [float(len(result.text_payload)), 1.0, 2.0]

    # 文本中应包含关键字段
    assert "[kb_code] KB-AGENT-01" in result.text_payload
    assert "[error_type] SQLInjection" in result.text_payload
    assert "[severity] high" in result.text_payload
    assert "[language] python" in result.text_payload
    assert "[framework] fastapi" in result.text_payload
    assert "[description] User input is concatenated directly into SQL." in result.text_payload
    assert "[pattern] f\"SELECT * FROM users WHERE name = '{name}'\"" in result.text_payload
    assert "[solution] Use parameterized queries." in result.text_payload

    # weaviate payload 应该包含结构化字段，供后续写入 Weaviate
    payload = result.weaviate_payload
    assert payload["sqlite_id"] == 101
    assert payload["kb_code"] == "KB-AGENT-01"
    assert payload["error_type"] == "SQLInjection"
    assert payload["severity"] == "high"
    assert payload["status"] == "active"
    assert payload["language"] == "python"
    assert payload["framework"] == "fastapi"
    assert payload["error_description"].startswith("User input")
    assert "parameterized queries" in payload["solution"]


def test_mock_agent_handles_missing_optional_fields():
    captured_texts: List[str] = []
    agent = MockDBEmbeddingAgent(embed_fn=fake_embed_factory(captured_texts))

    record = make_record(
        kb_code=None,
        language=None,
        framework=None,
        error_description="",
        problematic_pattern=None,
        solution=None,
    )
    result = agent.encode_issue_pattern(record)

    # 文本中的可选字段应优雅降级为空字符串
    assert "[kb_code] " in result.text_payload  # 空值也应保留标签
    assert "[language] " in result.text_payload
    assert "[framework] " in result.text_payload
    assert "[pattern] " in result.text_payload
    assert "[solution] " in result.text_payload

    # weaviate payload 中对应字段可以为 None 或空字符串
    payload = result.weaviate_payload
    assert payload["kb_code"] is None
    assert payload["language"] is None
    assert payload["framework"] is None
    assert payload["error_description"] == ""
    assert payload["problematic_pattern"] is None
    assert payload["solution"] is None


