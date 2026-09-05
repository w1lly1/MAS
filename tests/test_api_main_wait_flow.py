import unittest
from unittest.mock import AsyncMock, patch

from api import main as api_main
from utils.run_output import normalize_output_dir


class TestSingleAnalysisFlowWaitContract(unittest.IsolatedAsyncioTestCase):
    async def test_single_analysis_flow_passes_agent_system_to_wait_helper(self):
        agent_system = AsyncMock()
        dispatch_result = {
            "status": "dispatched",
            "run_id": "run-123",
            "total_files": 3,
            "report_path": "report.json",
            "report_relpath": "CVE-2011-1078/run-123",
            "estimated_timeout_seconds": 222,
        }

        with patch("api.main._init_system", AsyncMock(return_value=agent_system)), \
            patch("api.main._dispatch_directory_analysis", AsyncMock(return_value=dispatch_result)) as dispatch_mock, \
            patch("api.main._async_wait_for_reports", AsyncMock(return_value=None)) as wait_mock, \
            patch("api.main.click.echo"):
            await api_main._run_single_analysis_flow("target-dir", output_dir="CVE-2011-1078")

        dispatch_mock.assert_awaited_once_with(agent_system, "target-dir", output_dir="CVE-2011-1078")
        wait_mock.assert_awaited_once_with(agent_system, "run-123", 3, timeout=222)


class TestNormalizeOutputDirAlias(unittest.TestCase):
    def test_accepts_simple_name(self):
        self.assertEqual(normalize_output_dir("CVE-2011-1078"), "CVE-2011-1078")
