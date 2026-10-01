# -*- coding: utf-8 -*-
"""R1 回归测试：**超时/跑不完，绝不能记成成功**。

为什么专门测这个：批处理原先无条件写 `status='done'`，
于是"整段二次分析被静默跳过"和"真的跑完了"在汇总表和 CSV 里**长得一模一样**，
A/B 对比会被直接污染（而且不报错）。这四个用例各自锁住一环：

1. 等待助手必须把"超时"如实返回，并说出**缺哪几类**结果；
2. 正常完成时必须返回 completed；
3. 批处理遇到超时必须记 `partial`，**且不能记成 done**；
4. 汇总 agent 的体检函数要能指出"卡在哪个文件、缺哪一类"。
"""
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from api import main as api_main

REPO = Path(__file__).resolve().parent.parent


class TestWaitHelperStatusContract(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_is_reported_as_timeout_and_says_what_is_missing(self):
        agent_system = AsyncMock()
        agent_system.wait_for_run_completion = AsyncMock(return_value={
            'status': 'timeout',
            'summary_report': None,
            'consolidated_reports': [],
            'incomplete': {
                'known': True, 'expected': 2, 'completed': 1, 'pending': 1,
                'pending_details': [{
                    'requirement_id': 7, 'file': 'a.c',
                    'received': ['static_analysis'],
                    'missing': ['ai_analysis', 'performance_analysis', 'security_analysis'],
                }],
            },
        })
        echoed = []
        with patch("api.main.click.echo", side_effect=echoed.append):
            outcome = await api_main._async_wait_for_reports(agent_system, "run-1", 3, timeout=5)

        self.assertEqual(outcome['status'], 'timeout')
        text = "\n".join(str(x) for x in echoed)
        self.assertIn('超时', text)
        self.assertIn('ai_analysis', text)      # 缺哪一类必须被说出来
        self.assertIn('req 7', text.replace('未完成 req 7', 'req 7'))

    async def test_completed_path_returns_completed(self):
        agent_system = AsyncMock()
        agent_system.wait_for_run_completion = AsyncMock(return_value={
            'status': 'completed',
            'summary_report': None,
            'consolidated_reports': [],
        })
        with patch("api.main.click.echo"):
            outcome = await api_main._async_wait_for_reports(agent_system, "run-2", 1, timeout=5)
        self.assertEqual(outcome['status'], 'completed')

    async def test_wait_exception_is_error_not_completed(self):
        agent_system = AsyncMock()
        agent_system.wait_for_run_completion = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("api.main.click.echo"):
            outcome = await api_main._async_wait_for_reports(agent_system, "run-3", 1, timeout=5)
        self.assertEqual(outcome['status'], 'error')


class TestBatchFlowMarksPartial(unittest.IsolatedAsyncioTestCase):
    async def _run_batch(self, wait_return):
        cfg = REPO / "reports" / "_test_batch_partial.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({
            "items": [{"target_dir": str(REPO), "output_dir": "CVE-TEST-PARTIAL"}],
        }), encoding="utf-8")
        agent_system = AsyncMock()
        dispatch = {
            'status': 'dispatched', 'run_id': 'run-x', 'total_files': 1,
            'estimated_timeout_seconds': 1,
        }
        captured = {}

        def _fake_auto_summary(config_path, items, results):
            captured['results'] = [dict(r) for r in results]
            return []

        try:
            with patch("api.main._init_system", AsyncMock(return_value=agent_system)), \
                 patch("api.main._dispatch_directory_analysis", AsyncMock(return_value=dispatch)), \
                 patch("api.main._async_wait_for_reports", AsyncMock(return_value=wait_return)), \
                 patch("utils.experiments.auto_summary.run_auto_summary", side_effect=_fake_auto_summary), \
                 patch("api.main.click.echo"):
                await api_main._run_batch_flow(str(cfg), use_cpu=True)
        finally:
            cfg.unlink(missing_ok=True)
        return captured.get('results', [])

    async def test_timed_out_item_is_partial_not_done(self):
        results = await self._run_batch({'status': 'timeout', 'incomplete': {'known': True, 'pending_details': []}})
        self.assertEqual([r['status'] for r in results], ['partial'])
        self.assertNotIn('done', [r['status'] for r in results])

    async def test_completed_item_is_done(self):
        results = await self._run_batch({'status': 'completed'})
        self.assertEqual([r['status'] for r in results], ['done'])


class TestSummaryAgentDiagnosis(unittest.TestCase):
    def test_reports_missing_types_per_requirement(self):
        from core.agents.analysis_result_summary_agent import SummaryAgent

        agent = SummaryAgent()
        agent.run_meta['r1'] = {
            'expected': {1, 2}, 'completed': {1}, 'issues': [],
            'target_directory': None, 'closed': False,
        }
        agent.analysis_results[2] = {'types': {'static_analysis'}, 'file_path': 'x.c'}

        info = agent.describe_incomplete_requirements('r1')
        self.assertTrue(info['known'])
        self.assertEqual(info['expected'], 2)
        self.assertEqual(info['completed'], 1)
        self.assertEqual(info['pending'], 1)
        self.assertEqual(info['pending_details'][0]['requirement_id'], 2)
        self.assertEqual(info['pending_details'][0]['missing'],
                         ['ai_analysis', 'performance_analysis', 'security_analysis'])
        self.assertEqual(info['pending_details'][0]['received'], ['static_analysis'])

    def test_unknown_run_is_reported_not_guessed(self):
        from core.agents.analysis_result_summary_agent import SummaryAgent

        agent = SummaryAgent()
        info = agent.describe_incomplete_requirements('nope')
        self.assertFalse(info['known'])


class TestIntegrationDelegation(unittest.TestCase):
    def test_delegates_to_summary_agent(self):
        from core.agents_integration import AgentIntegration

        system = AgentIntegration.__new__(AgentIntegration)   # 跳过重量级 __init__
        system.agents = {
            'summary': type('S', (), {
                'describe_incomplete_requirements': lambda self, rid: {'run_id': rid, 'known': True},
            })(),
        }
        self.assertEqual(system.describe_incomplete_requirements('r9')['run_id'], 'r9')

    def test_missing_summary_agent_is_safe(self):
        from core.agents_integration import AgentIntegration

        system = AgentIntegration.__new__(AgentIntegration)
        system.agents = {}
        info = system.describe_incomplete_requirements('r9')
        self.assertFalse(info['known'])


if __name__ == '__main__':
    unittest.main()
