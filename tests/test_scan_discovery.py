import tempfile
import unittest
from pathlib import Path

from utils.scan_discovery import discover_source_files, estimate_analysis_timeout


class TestScanDiscovery(unittest.TestCase):
    def test_discover_source_files_stops_at_budget_and_skips_ignored_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_dir = root / "src"
            ignored_dir = root / ".git" / "objects"
            src_dir.mkdir(parents=True)
            ignored_dir.mkdir(parents=True)

            (src_dir / "a.py").write_text("print('a')\n", encoding="utf-8")
            (src_dir / "b.py").write_text("print('b')\n", encoding="utf-8")
            (src_dir / "c.py").write_text("print('c')\n", encoding="utf-8")
            (ignored_dir / "hidden.py").write_text("print('hidden')\n", encoding="utf-8")

            result = discover_source_files(
                str(root),
                supported_extensions=(".py",),
                ignored_directories=(".git",),
                max_files=2,
            )

            self.assertTrue(result["partial_scan"])
            self.assertEqual(2, result["scanned_files"])
            self.assertGreaterEqual(result["skipped_directories"], 1)
            self.assertTrue(all(".git" not in item["path"] for item in result["files"]))

    def test_timeout_estimator_scales_with_file_size_and_clamps(self):
        small_files = [
            {"path": "small.py", "size": 100},
            {"path": "small2.py", "size": 120},
        ]
        large_files = [
            {"path": "large.py", "size": 500_000},
            {"path": "large2.py", "size": 750_000},
        ]

        small = estimate_analysis_timeout(
            small_files,
            enabled_agents=("ai_code_quality", "ai_security", "ai_performance", "static_scan"),
            timeout_config={"min_timeout_seconds": 60, "max_timeout_seconds": 1200, "safety_factor": 1.1},
        )
        large = estimate_analysis_timeout(
            large_files,
            enabled_agents=("ai_code_quality", "ai_security", "ai_performance", "static_scan"),
            timeout_config={"min_timeout_seconds": 60, "max_timeout_seconds": 1200, "safety_factor": 1.1},
        )

        self.assertGreater(large["estimated_timeout_seconds"], small["estimated_timeout_seconds"])
        self.assertGreaterEqual(small["estimated_timeout_seconds"], 60)
        self.assertLessEqual(large["estimated_timeout_seconds"], 1200)
