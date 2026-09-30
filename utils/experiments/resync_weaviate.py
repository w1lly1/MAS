# -*- coding: utf-8 -*-
"""从 SQLite(mas.db) 重新同步 IssuePattern 到 Weaviate（克隆后重建向量库）。"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.database.vector_sync import IssuePatternSyncService
from core.agents.ai_driven_database_manage_agent import DefaultKnowledgeEncodingAgent
from infrastructure.database.weaviate.service import WeaviateVectorService


def _embed(text: str, layer=None):
    from infrastructure.embeddings.codebert_embedder import embed_text
    return embed_text(text, layer)


async def main() -> None:
    db = DatabaseService()
    vec = WeaviateVectorService(embed_fn=_embed)
    agent = DefaultKnowledgeEncodingAgent(embed_fn=_embed)
    sync = IssuePatternSyncService(db_service=db, vector_service=vec, agent=agent)
    vec.connect(auto_create_schema=True)
    layers = ["semantic", "code_pattern", "solution", "full"]
    try:
        results = await sync.sync_all_issue_patterns(status="active", layers=layers)
        print(f"RE-SYNC DONE: {len(results)} patterns, layers={layers}")
    finally:
        vec.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
