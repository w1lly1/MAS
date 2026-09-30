# -*- coding: utf-8 -*-
"""清库：删除 SQLite 的 curated_issues + issue_patterns，以及 Weaviate 集合。"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.database.weaviate.service import WeaviateVectorService


async def main() -> None:
    db = DatabaseService()
    n_ci = await db.delete_all_curated_issues()
    n_ip = await db.delete_all_issue_patterns()
    print(f"SQLite cleared: curated_issues={n_ci}, issue_patterns={n_ip}")

    vec = WeaviateVectorService()
    vec.connect()
    ok = vec.delete_collection()
    print(f"Weaviate collection deleted: {ok}")
    vec.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
