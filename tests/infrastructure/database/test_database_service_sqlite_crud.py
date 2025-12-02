import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.base import MASTestCase
from infrastructure.database.sqlite.service import DatabaseService


class TestDatabaseServiceSQLiteCRUD(MASTestCase):
    """
    针对 SQLite 后端的基础增删改查测试：
    - ReviewSession（审核会话）
    - CuratedIssue（人工确认的问题实例）
    - IssuePattern（错误模式 + 解决方案）
    """

    def setUp(self):
        super().setUp()
        db_path = Path(self.temp_dir) / "test_mas_db.sqlite3"
        self.db = DatabaseService(database_url=f"sqlite:///{db_path}")

    # ====== ReviewSession CRUD ======

    async def _review_session_flow(self):
        # 创建新会话
        session_id = await self.db.create_review_session(
            session_id="cli-session-1",
            user_message="请审核这个目录的代码",
            code_directory="/tmp/project",
            display_id="UR-TEST-0001",
            title="测试审核会话",
            status="open",
        )
        self.assertIsInstance(session_id, int)

        # 查询所有会话
        all_sessions = await self.db.get_review_sessions()
        self.assertTrue(len(all_sessions) >= 1)
        created = [s for s in all_sessions if s["id"] == session_id][0]
        self.assertEqual(created["display_id"], "UR-TEST-0001")
        self.assertEqual(created["status"], "open")

        # 更新状态为 completed，验证 state 更新
        ok = await self.db.update_review_session_status(session_id, "completed")
        self.assertTrue(ok)

        completed_sessions = await self.db.get_review_sessions(status="completed")
        self.assertTrue(
            any(s["id"] == session_id and s["status"] == "completed" for s in completed_sessions)
        )

        # 模拟“软删除”：标记为 cancelled，并确认在 open 列表中不可见
        ok = await self.db.update_review_session_status(session_id, "cancelled")
        self.assertTrue(ok)
        open_sessions = await self.db.get_review_sessions(status="open")
        self.assertFalse(any(s["id"] == session_id for s in open_sessions))

        # 硬删该会话，确认在任意状态下都不可见
        ok = await self.db.delete_review_session(session_id)
        self.assertTrue(ok)
        all_after_delete = await self.db.get_review_sessions()
        self.assertFalse(any(s["id"] == session_id for s in all_after_delete))

    def test_review_session_crud(self):
        asyncio.run(self._review_session_flow())

    # ====== CuratedIssue CRUD ======

    async def _curated_issue_flow(self):
        # 先创建一个会话以便挂载问题实例
        session_id = await self.db.create_review_session(
            session_id="cli-session-2",
            user_message="检查支付模块",
            code_directory="/tmp/payment",
            display_id="UR-TEST-0002",
        )

        # 创建问题实例
        issue_id = await self.db.create_curated_issue(
            session_id=session_id,
            file_path="app/payment.py",
            start_line=10,
            end_line=20,
            code_snippet="def pay():\n    pass",
            problem_phenomenon="在高并发下会偶发重复扣款",
            root_cause="缺少幂等控制",
            solution="增加幂等键与数据库唯一约束",
            severity="high",
            status="open",
        )
        self.assertIsInstance(issue_id, int)

        # 按会话查询
        issues = await self.db.get_curated_issues(session_db_id=session_id)
        self.assertTrue(len(issues) >= 1)
        created = [i for i in issues if i["id"] == issue_id][0]
        self.assertEqual(created["file_path"], "app/payment.py")
        self.assertEqual(created["severity"], "high")
        self.assertEqual(created["status"], "open")

        # 更新状态为 resolved
        ok = await self.db.update_curated_issue_status(issue_id, "resolved")
        self.assertTrue(ok)

        updated = await self.db.get_curated_issues(session_db_id=session_id, status="resolved")
        self.assertTrue(
            any(i["id"] == issue_id and i["status"] == "resolved" for i in updated)
        )

        # 模拟软删：在 open 状态列表中不应再出现
        open_issues = await self.db.get_curated_issues(session_db_id=session_id, status="open")
        self.assertFalse(any(i["id"] == issue_id for i in open_issues))

        # 硬删该问题实例，确认在任何状态下都不再出现
        ok = await self.db.delete_curated_issue(issue_id)
        self.assertTrue(ok)
        all_issues = await self.db.get_curated_issues(session_db_id=session_id)
        self.assertFalse(any(i["id"] == issue_id for i in all_issues))

    def test_curated_issue_crud(self):
        asyncio.run(self._curated_issue_flow())

    # ====== IssuePattern CRUD ======

    async def _issue_pattern_flow(self):
        # 创建模式
        pattern_id = await self.db.create_issue_pattern(
            error_type="SQLInjection",
            error_description="使用字符串拼接构造 SQL 语句，存在注入风险",
            problematic_pattern="cursor.execute(f\"SELECT * FROM users WHERE name = '{name}'\")",
            solution="使用参数化查询或 ORM 查询接口，避免直接拼接用户输入",
            severity="critical",
            title="字符串拼接 SQL 导致注入",
            kb_code="KB-SEC-SQLI-TEST",
            language="python",
            framework="django",
            file_pattern="*views.py",
            class_pattern="*View",
            tags='["security","sql","injection"]',
            status="active",
        )
        self.assertIsInstance(pattern_id, int)

        # 查询所有模式
        patterns = await self.db.get_issue_patterns()
        created_list = [p for p in patterns if p["id"] == pattern_id]
        self.assertEqual(len(created_list), 1)
        created = created_list[0]
        self.assertEqual(created["error_type"], "SQLInjection")
        self.assertEqual(created["severity"], "critical")
        self.assertEqual(created["status"], "active")

        # 更新部分字段
        ok = await self.db.update_issue_pattern(
            pattern_id,
            severity="high",
            status="deprecated",
        )
        self.assertTrue(ok)

        # 重新获取所有模式，确认字段已更新
        updated_patterns = await self.db.get_issue_patterns()
        updated = [p for p in updated_patterns if p["id"] == pattern_id][0]
        self.assertEqual(updated["severity"], "high")
        self.assertEqual(updated["status"], "deprecated")

        # 使用状态过滤模拟“软删”效果：
        # 在 active 列表中不应出现该模式
        active_patterns = await self.db.get_issue_patterns(status="active")
        self.assertFalse(any(p["id"] == pattern_id for p in active_patterns))

        # 在 deprecated 列表中应能找到
        deprecated_patterns = await self.db.get_issue_patterns(status="deprecated")
        self.assertTrue(any(p["id"] == pattern_id for p in deprecated_patterns))

        # 硬删该模式，确认不再出现在列表中
        ok = await self.db.delete_issue_pattern(pattern_id)
        self.assertTrue(ok)
        all_after_delete = await self.db.get_issue_patterns()
        self.assertFalse(any(p["id"] == pattern_id for p in all_after_delete))

    def test_issue_pattern_crud(self):
        asyncio.run(self._issue_pattern_flow())


