import unittest
from unittest.mock import AsyncMock, patch

from api import main as api_main


class TestSingleAnalysisFlowWaitContract(unittest.IsolatedAsyncioTestCase):
    async def test_single_analysis_flow_passes_agent_system_to_wait_helper(self):
        agent_system = AsyncMock()
        dispatch_result = {
            "status": "dispatched",
            "run_id": "run-123",
            "total_files": 3,
            "report_path": "report.json",
        }

        with patch("api.main._init_system", AsyncMock(return_value=agent_system)), \
            patch("api.main._dispatch_directory_analysis", AsyncMock(return_value=dispatch_result)), \
            patch("api.main._async_wait_for_reports", AsyncMock(return_value=None)) as wait_mock, \
            patch("api.main.click.echo"):
            await api_main._run_single_analysis_flow("target-dir")

        wait_mock.assert_awaited_once_with(agent_system, "run-123", 3)