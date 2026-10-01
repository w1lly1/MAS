# -*- coding: utf-8 -*-
"""`llm_semantic`（大模型语义理解）字段的不变量测试。

## 这个字段是什么

知识库里新增的一列，存**大模型从代码出发写出的语义理解**（英文，与索引侧同一套
分类与语域：功能 + 风险）。它和分析时的查询文本是**唯一能对齐的一对**
（依据见《02》第十六节的规则对齐实验：代码查询进 top-5 只有 34%，同语域散文 100%）。

## 三条必须锁住的不变量

1. **只进 semantic 与 full 两层**：code_pattern / solution 两层的索引文本里
   绝不能出现它 —— 那两层的语域是"代码模式/文件名/类名"与"修复前后代码"，
   混进散文会破坏它们原本的模态一致性（这是本字段的设计前提，不是可以随手改的细节）。
2. **空值必须等价于"没有这个字段"**：字段留空时，四层索引文本要与加字段**之前逐字节相同**。
   否则一上线就把既有向量作废了 —— 而"加字段"这一步本身不该有这种副作用，
   只有真正填了内容才应该付出重建索引的代价。
3. **两份实现必须一致**：`weaviate/service.py` 与 `vector_sync.py` 各有一份层文本构造器
   （历史上两份曾经不一致过），改动必须在两处同时成立。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from infrastructure.database.vector_sync import (  # noqa: E402
    DefaultKnowledgeEncodingAgent, IssuePatternRecord,
)
from infrastructure.database.weaviate.service import WeaviateVectorService  # noqa: E402

LAYERS = ("semantic", "code_pattern", "solution", "full")

BASE_PROPS = {
    "sqlite_id": 7,
    "id": 7,          # IssuePatternRecord.from_dict 用 "id"（Weaviate 属性里叫 sqlite_id）
    "error_type": "memory_overflow",
    "severity": "medium",
    "status": "active",
    "language": "c",
    "framework": "linux",
    "error_description": "fs/f2fs/segment.c allows local users to cause a denial of service.",
    "problematic_pattern": "Unchecked arithmetic or index usage may cause out-of-bounds access.",
    "solution": "Remove incorrect logic: if (a > b) {. Ensure corrected path: if (a >= b) {",
    "file_pattern": "fs/f2fs/segment.c",
    "class_pattern": "f2fs_balance_fs",
}

LLM_TEXT = (
    "[error_type] memory_overflow\n"
    "function f2fs_balance_fs walks the segment array and indexes it with a value that "
    "is only checked for being positive; a negative or oversized value leads to an "
    "out-of-bounds read of the segment array."
)


def _svc() -> WeaviateVectorService:
    return WeaviateVectorService()


def _agent() -> DefaultKnowledgeEncodingAgent:
    return DefaultKnowledgeEncodingAgent()


def _layer_texts_weaviate(props: dict) -> dict:
    svc = _svc()
    return {L: svc._build_enhanced_issue_pattern_text(props, L) for L in LAYERS}


def _layer_texts_vector_sync(props: dict) -> dict:
    return _agent()._build_layer_texts(props)


def test_llm_semantic_only_enters_semantic_and_full():
    """核心约束：只有 semantic / full 两层带它，另两层一个字都不能沾。"""
    props = dict(BASE_PROPS, llm_semantic=LLM_TEXT)
    svc_texts = _layer_texts_weaviate(props)
    vs_texts = _layer_texts_vector_sync(props)

    for name, texts in (("weaviate", svc_texts), ("vector_sync", vs_texts)):
        assert "[llm_semantic]" in texts["semantic"], name
        assert "[llm_semantic]" in texts["full"], name
        assert "f2fs_balance_fs walks the segment array" in texts["semantic"], name
        assert "f2fs_balance_fs walks the segment array" in texts["full"], name
        # 关键否定断言：这两层绝不能出现
        assert "llm_semantic" not in texts["code_pattern"], (
            "%s 的 code_pattern 层混进了 llm_semantic —— 会破坏该层的代码/模式语域" % name)
        assert "llm_semantic" not in texts["solution"], (
            "%s 的 solution 层混进了 llm_semantic —— 会破坏该层的代码语域" % name)
        assert "f2fs_balance_fs walks the segment array" not in texts["code_pattern"], name
        assert "f2fs_balance_fs walks the segment array" not in texts["solution"], name


def test_empty_llm_semantic_is_byte_identical_to_pre_change_text():
    """留空 ≡ 没有这个字段：四层文本与"加字段之前"逐字节相同（不会作废既有向量）。

    这里的"加字段之前"是**照原实现抄写的参考版本**（下方 OLD_*）。抄一份而不是
    调被测函数，是为了让这条断言真的能发现"不小心改变了旧层文本"。
    """
    expected = {
        "semantic": "\n".join([
            f"[error_type] {BASE_PROPS['error_type']}",
            f"[severity] {BASE_PROPS['severity']}",
            f"[language] {BASE_PROPS['language']}",
            f"[framework] {BASE_PROPS['framework']}",
            f"[description] {BASE_PROPS['error_description']}",
        ]),
        "code_pattern": "\n".join([
            f"[problematic_pattern] {BASE_PROPS['problematic_pattern']}",
            f"[file_pattern] {BASE_PROPS['file_pattern']}",
            f"[class_pattern] {BASE_PROPS['class_pattern']}",
            f"[language] {BASE_PROPS['language']}",
        ]),
        "solution": "\n".join([
            f"[solution] {BASE_PROPS['solution']}",
            f"[error_description] {BASE_PROPS['error_description']}",
            f"[severity] {BASE_PROPS['severity']}",
        ]),
        "full": "\n".join([
            f"[error_type] {BASE_PROPS['error_type']}",
            f"[severity] {BASE_PROPS['severity']}",
            f"[language] {BASE_PROPS['language']}",
            f"[framework] {BASE_PROPS['framework']}",
            f"[description] {BASE_PROPS['error_description']}",
            f"[pattern] {BASE_PROPS['problematic_pattern']}",
            f"[solution] {BASE_PROPS['solution']}",
            f"[file_pattern] {BASE_PROPS['file_pattern']}",
            f"[class_pattern] {BASE_PROPS['class_pattern']}",
        ]),
    }
    # 空字符串与"键完全不存在"必须表现一致（老库、老调用方都不会传这个键）
    for candidate in (dict(BASE_PROPS, llm_semantic=""), dict(BASE_PROPS)):
        assert _layer_texts_weaviate(candidate) == expected
        assert _layer_texts_vector_sync(candidate) == expected


def test_two_implementations_agree_on_all_four_layers():
    """两份实现必须逐字节一致（历史上曾经不一致过，导致 file/class 恒为空）。"""
    for extra in ({}, {"llm_semantic": LLM_TEXT}):
        props = dict(BASE_PROPS, **extra)
        assert _layer_texts_weaviate(props) == _layer_texts_vector_sync(props)


def test_record_roundtrip_keeps_llm_semantic():
    """SQLite 侧：字段必须能读进来、并传进 Weaviate 的 payload。"""
    data = dict(BASE_PROPS)
    data["llm_semantic"] = LLM_TEXT
    rec = IssuePatternRecord.from_dict(data)
    assert rec.llm_semantic == LLM_TEXT
    assert rec.to_agent_payload()["llm_semantic"] == LLM_TEXT
    # 老数据（没有这一列的值）不能炸
    legacy = IssuePatternRecord.from_dict(BASE_PROPS)
    assert legacy.llm_semantic == ""
    assert legacy.to_agent_payload()["llm_semantic"] == ""


def test_field_is_appended_last_in_dataclass():
    """字段顺序敏感：新字段必须追加在末尾（按位置构造的调用方依赖这一点）。"""
    import dataclasses

    names = [f.name for f in dataclasses.fields(IssuePatternRecord)]
    assert names[-1] == "llm_semantic", names
    # 排在它前面的字段顺序也不能动（历史顺序即契约）
    assert names[:11] == ["id", "error_type", "severity", "status", "language",
                          "framework", "error_description", "problematic_pattern",
                          "solution", "file_pattern", "class_pattern"], names


def test_migration_adds_column_to_old_schema_db():
    """老库（没有 llm_semantic 列）必须能被**幂等**地补上该列。

    为什么必须有这条：`Base.metadata.create_all` 只建缺失的**表**，不会给已存在的表补列。
    线上那份 mas.db 就是老结构，缺了补列这一步，一读新字段就报 "no such column"。

    这里用 sqlite3 手搓一个"老结构"的表，再让 DatabaseService 去初始化它——
    比拿真库试更严格（真库可能已经被补过了，测不出问题）。
    """
    import sqlite3
    import uuid
    from pathlib import Path

    from infrastructure.database.sqlite.service import DatabaseService

    # 注意：**不要**用 tempfile / tmp_path —— 在受限环境（沙箱）下新建目录会被拒，
    # 测试会在 setup 阶段就挂，报的错还和被测逻辑无关。
    # 这里用一个已存在且可写的目录 + 唯一文件名，结束后自己删干净。
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / ("_test_migrate_%s.db" % uuid.uuid4().hex[:8])
    try:
        con = sqlite3.connect(str(db))
        con.execute("""CREATE TABLE issue_patterns (
            id INTEGER PRIMARY KEY, title VARCHAR(255), error_type VARCHAR(255),
            severity VARCHAR(50), language VARCHAR(50), framework VARCHAR(100),
            error_description TEXT, problematic_pattern TEXT, solution TEXT,
            file_pattern VARCHAR(255), class_pattern VARCHAR(255), tags TEXT,
            status VARCHAR(50), created_at DATETIME, updated_at DATETIME)""")
        con.execute(
            "INSERT INTO issue_patterns (id, title, error_type) VALUES (1, 'CVE-X', 'dos')")
        con.commit()
        before = {r[1] for r in con.execute("PRAGMA table_info(issue_patterns)")}
        con.close()
        assert "llm_semantic" not in before, "构造的测试库应当没有新列"

        url = "sqlite:///%s" % db.as_posix()
        svc1 = DatabaseService(database_url=url)
        con = sqlite3.connect(str(db))
        after = {r[1] for r in con.execute("PRAGMA table_info(issue_patterns)")}
        assert "llm_semantic" in after, "补列没有生效"
        # 原有数据必须完好（只做 ADD COLUMN，不该动数据）
        assert con.execute("select title from issue_patterns where id=1").fetchone()[0] == "CVE-X"
        con.close()

        # 幂等：再跑一次不能报错、也不能重复加列
        svc2 = DatabaseService(database_url=url)
        con = sqlite3.connect(str(db))
        cols = [r[1] for r in con.execute("PRAGMA table_info(issue_patterns)")]
        con.close()
        assert cols.count("llm_semantic") == 1
    finally:
        # 必须先 dispose 连接池，否则 Windows 上文件被占用、删不掉，测试会在 reports/ 里留垃圾
        for svc in (locals().get("svc1"), locals().get("svc2")):
            try:
                svc.engine.dispose()
            except Exception:  # noqa: BLE001
                pass
        for suffix in ("", "-journal", "-wal", "-shm"):
            try:
                Path(str(db) + suffix).unlink()
            except OSError:
                pass
