"""Tests for current AIDrivenUserCommunicationAgent contract.

These tests intentionally focus on the current async routing/dispatch behavior
instead of legacy synchronous helper methods.
"""

import unittest
from unittest.mock import AsyncMock, Mock

from core.agents.ai_driven_user_communication_agent import AIDrivenUserCommunicationAgent


class _FakeDataManageAgent:
	def __init__(self, result=None):
		self.agent_id = "db_manage_agent"
		self._result = result or {"status": "success"}

	async def user_requirement_interpret(self, user_requirement, session_id):
		return self._result


class _FakeIntegration:
	def __init__(self, data_manage):
		self.agents = {"data_manage": data_manage}


class TestAIUserCommunicationAgentCurrentContract(unittest.IsolatedAsyncioTestCase):
	def setUp(self):
		self.agent = AIDrivenUserCommunicationAgent()

	def test_parse_task_plan_parses_valid_intent(self):
		raw = '{"intent":"db","explanation":""}'
		user_visible, plan = self.agent._parse_task_plan_from_response(raw)

		self.assertIsInstance(user_visible, str)
		self.assertEqual(plan.get("intent"), "db")
		self.assertIn("code_analysis_tasks", plan)

	def test_parse_task_plan_fallback_from_malformed_json(self):
		malformed = (
			'{\n'
			'  "intent": "db",\n'
			'  "explanation": "delete scoped rows"\n'
			'  "code_analysis_tasks": []\n'
			'}'
		)

		user_visible, plan = self.agent._parse_task_plan_from_response(malformed)

		self.assertIsInstance(user_visible, str)
		self.assertEqual(plan.get("intent"), "db")

	def test_apply_confirm_to_tasks_preserves_delete_scope(self):
		tasks = [{"target": "curated_issue", "action": "delete", "data": {"ids": [3, 4]}}]
		confirmed = self.agent._apply_confirm_to_tasks(tasks)

		self.assertEqual(len(confirmed), 1)
		self.assertEqual(confirmed[0]["action"], "delete")
		self.assertEqual(confirmed[0]["data"].get("ids"), [3, 4])
		self.assertTrue(confirmed[0]["data"].get("confirm"))

	async def test_dispatch_db_tasks_sync_sets_pending_confirm(self):
		pending_tasks = [{"target": "curated_issue", "action": "delete_all", "data": {}}]
		data_manage = _FakeDataManageAgent(
			{
				"status": "need_confirm",
				"pending_action": "delete_all",
				"pending_tasks": pending_tasks,
			}
		)
		self.agent.agent_integration = _FakeIntegration(data_manage)

		await self.agent._dispatch_db_tasks(
			db_tasks=[],
			session_id="s1",
			wait_for_db=True,
			raw_user_message="删除全部",
		)

		pending = self.agent._get_pending_db_confirm("s1")
		self.assertIsNotNone(pending)
		self.assertEqual(pending.get("pending_action"), "delete_all")
		self.assertEqual(pending.get("tasks"), pending_tasks)

	async def test_dispatch_db_tasks_async_forward_message(self):
		data_manage = Mock()
		data_manage.agent_id = "db_manage_agent"
		self.agent.agent_integration = _FakeIntegration(data_manage)
		self.agent.dispatch_message = AsyncMock()

		await self.agent._dispatch_db_tasks(
			db_tasks=[],
			session_id="s2",
			wait_for_db=False,
			raw_user_message="查询数据库",
			forced_tasks=[{"target": "review_session", "action": "query", "data": {}}],
		)

		self.agent.dispatch_message.assert_awaited_once()
		kwargs = self.agent.dispatch_message.await_args.kwargs
		self.assertEqual(kwargs.get("receiver"), "db_manage_agent")
		self.assertEqual(kwargs.get("message_type"), "user_requirement")
		requirement = kwargs.get("content", {}).get("requirement", {})
		self.assertEqual(requirement.get("raw_text"), "查询数据库")
		self.assertEqual(
			requirement.get("forced_tasks"),
			[{"target": "review_session", "action": "query", "data": {}}],
		)


if __name__ == "__main__":
	unittest.main()
