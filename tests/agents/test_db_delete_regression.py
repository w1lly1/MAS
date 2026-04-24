import unittest
from unittest.mock import patch

from core.agents.ai_driven_database_manage_agent import AIDrivenDatabaseManageAgent


class TestDeleteFlowRegression(unittest.TestCase):
    @patch.object(AIDrivenDatabaseManageAgent, "_init_weaviate_connection", lambda self: None)
    def test_normalize_task_extracts_ids_from_where(self):
        agent = AIDrivenDatabaseManageAgent()

        action, target, data = agent._normalize_task(
            action="delete",
            target="curated_issue",
            data={"where": "id IN (3, 4)"},
            session_id="s1",
        )

        self.assertEqual(target, "curated_issue")
        self.assertEqual(action, "delete_by_ids")
        self.assertEqual(data.get("ids"), [3, 4])

    @patch.object(AIDrivenDatabaseManageAgent, "_init_weaviate_connection", lambda self: None)
    def test_requires_delete_all_confirm_false_when_ids_present(self):
        agent = AIDrivenDatabaseManageAgent()

        need_confirm = agent._requires_delete_all_confirm(
            tasks=[{"target": "curated_issue", "action": "delete", "data": {"ids": [3, 4]}}],
            session_id="s1",
        )

        self.assertFalse(need_confirm)

    @patch.object(AIDrivenDatabaseManageAgent, "_init_weaviate_connection", lambda self: None)
    def test_requires_delete_all_confirm_true_for_unscoped_delete(self):
        agent = AIDrivenDatabaseManageAgent()

        need_confirm = agent._requires_delete_all_confirm(
            tasks=[{"target": "curated_issue", "action": "delete", "data": {}}],
            session_id="s1",
        )

        self.assertTrue(need_confirm)


if __name__ == "__main__":
    unittest.main()
