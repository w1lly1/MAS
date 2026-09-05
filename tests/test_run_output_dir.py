import tempfile
import unittest
from pathlib import Path

from infrastructure.reports import ReportManager
from utils.run_output import format_run_report_relpath, normalize_output_dir


class TestNormalizeOutputDir(unittest.TestCase):
    def test_accepts_simple_name(self):
        self.assertEqual(normalize_output_dir("CVE-2011-1078"), "CVE-2011-1078")

    def test_uses_basename_for_paths(self):
        self.assertEqual(
            normalize_output_dir(r"reports\analysis\CVE-2011-1078"),
            "CVE-2011-1078",
        )

    def test_rejects_invalid_characters(self):
        with self.assertRaises(ValueError):
            normalize_output_dir("bad name")

    def test_format_relpath(self):
        self.assertEqual(format_run_report_relpath("uuid-1"), "uuid-1")
        self.assertEqual(
            format_run_report_relpath("uuid-1", "CVE-2011-1078"),
            "CVE-2011-1078/uuid-1",
        )


class TestReportManagerRunScope(unittest.TestCase):
    def test_register_nested_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ReportManager(base_dir=Path(tmp))
            root = mgr.register_run_scope("run-abc", output_dir="CVE-2011-1078")
            self.assertEqual(root, Path(tmp) / "analysis" / "CVE-2011-1078" / "run-abc")
            self.assertTrue(root.exists())
            self.assertEqual(mgr.resolve_run_root("run-abc"), root)
            path = mgr.generate_run_scoped_report(
                "run-abc",
                {"ok": True},
                "dispatch.json",
            )
            self.assertTrue(str(path).endswith(str(Path("CVE-2011-1078") / "run-abc" / "dispatch.json")))
