# -*- coding: utf-8 -*-
"""回归测试：写盘瘦身（《01》待办 7 —— 把"瘦身"从运营脚本变成生产行为）。

背景：候选里**每条都内嵌一份被分析文件全文**（`_current_code`），实测单个 r2 文件 227MB。
写盘时去掉它不丢信息（那份源码就在数据集目录里，证据里留着 `_analysis_file`）。

三条必须成立：
  ① 写盘产物里没有 `_current_code`，但**工具真正读的字段一个都不能少**；
  ② **内存里的判定不受影响** —— 门控要用 `_current_code`，瘦身只在写盘那一步；
  ③ 可关闭（配置项 / 环境变量），关掉就回到历史行为。

工作目录用**仓库内固定路径**（系统临时目录在本机沙箱里清理会抛 WinError 5，《03》坑 32）。
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "local_libs"))
sys.path.insert(0, str(REPO_ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

import utils.artifact_compaction as ac  # noqa: E402
from infrastructure.reports import ReportManager  # noqa: E402

SCRATCH = REPO_ROOT / "reports" / "_test_artifact_compaction"
BIG_CODE = "x = 1\n" * 20000          # ~120KB，足以触发"单条字符串截断"

PROBE_FIELDS = ("channel", "sqlite_id", "gating_decision", "rejection_reason",
                "matched_fields", "structured_score", "semantic_score", "context_score",
                "anchor_score", "total_score", "vector_layer", "error_type", "severity")


def make_payload():
    cand = {f: ("v-%s" % f) for f in PROBE_FIELDS}
    cand["structured_score"] = 0.83
    cand["semantic_score"] = 0.0
    cand["_current_code"] = BIG_CODE          # 会被**丢弃**（体积元凶）
    cand["error_description"] = BIG_CODE      # 会被**截断**（另一个巨型字段，用于测截断路径）
    return {
        "run_id": "r-1",
        "file": "/data/before/CVE-X/a.c",
        "new_findings": [{"file": "/data/before/CVE-X/a.c", "evidence": dict(cand)}],
        "retrieval_evidence": [{"issue_file": "/data/before/CVE-X/a.c", "candidates": [dict(cand)]}],
        "gap_retrieval_evidence": [{"issue_file": "/data/before/CVE-X/a.c", "candidates": [dict(cand)],
                                    "query_semantic_used": True}],
    }


def probe(payload):
    """抽出"工具判定真正依赖的字段"，用于瘦身前后比对。"""
    out = {"new_findings": len(payload.get("new_findings") or []),
           "file": payload.get("file"), "run_id": payload.get("run_id"), "evidence": {}}
    for key in ("retrieval_evidence", "gap_retrieval_evidence"):
        out["evidence"][key] = [
            {"issue_file": ev.get("issue_file"),
             "query_semantic_used": ev.get("query_semantic_used"),
             "candidates": [{f: c.get(f) for f in PROBE_FIELDS if f in c}
                            for c in (ev.get("candidates") or [])]}
            for ev in (payload.get(key) or [])
        ]
    return out


class TestArtifactCompaction(unittest.TestCase):
    def setUp(self):
        if SCRATCH.exists():
            shutil.rmtree(SCRATCH, ignore_errors=True)
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(SCRATCH, ignore_errors=True)
        os.environ.pop("MAS_ARTIFACT_COMPACTION", None)

    # ---------- ① 瘦身本身 ----------
    def test_drops_current_code_and_truncates_long_strings(self):
        payload = make_payload()
        small, stats = ac.compact_payload(payload)
        text = json.dumps(small, ensure_ascii=False)
        self.assertNotIn("_current_code", text)
        # 截断路径要真的被执行到：用**另一个**长字段（error_description）来验证
        # （第一版测试只放了 `_current_code`，它被丢弃后就没有长字符串可截断 → 断言失败，
        #   这条测试自己抓出了这个盲点）
        self.assertIn("truncated", stats, "超长字符串应当被截断")
        desc = small["retrieval_evidence"][0]["candidates"][0]["error_description"]
        self.assertLess(len(desc), len(BIG_CODE))
        self.assertIn("已截断", desc)
        # 入参未被改动
        self.assertIn("_current_code", json.dumps(payload, ensure_ascii=False))

    def test_keeps_every_field_the_tools_read(self):
        payload = make_payload()
        small, _ = ac.compact_payload(payload)
        self.assertEqual(probe(small), probe(payload),
                         "瘦身改变了工具读得到的字段 —— 这是不能接受的")

    def test_size_reduction_is_large(self):
        payload = make_payload()
        before = len(json.dumps(payload, ensure_ascii=False))
        small, _ = ac.compact_payload(payload)
        after = len(json.dumps(small, ensure_ascii=False))
        self.assertLess(after, before / 2, "瘦身效果应当明显（本例是一半以上）")

    # ---------- ② 写盘路径真的用了它 ----------
    def test_report_manager_writes_compacted_file(self):
        mgr = ReportManager(base_dir=SCRATCH)
        mgr.register_run_scope("run-c1", output_dir="CVE-TEST")
        path = mgr.generate_run_scoped_report("run-c1", make_payload(), "x_r2.json",
                                             subdir="second_pass/consolidated")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertNotIn("_current_code", Path(path).read_text(encoding="utf-8"))
        self.assertEqual(probe(data), probe(make_payload()))

    # ---------- ③ 可关闭 ----------
    def test_can_be_disabled_by_env(self):
        os.environ["MAS_ARTIFACT_COMPACTION"] = "off"
        self.assertFalse(ac.compaction_enabled())
        mgr = ReportManager(base_dir=SCRATCH)
        mgr.register_run_scope("run-c2", output_dir="CVE-TEST")
        path = mgr.generate_run_scoped_report("run-c2", make_payload(), "y_r2.json",
                                             subdir="second_pass/consolidated")
        self.assertIn("_current_code", Path(path).read_text(encoding="utf-8"),
                      "关掉开关后应当回到历史行为（原样写入）")

    # ---------- 变异对照 ----------
    def test_mutation_empty_drop_set_keeps_current_code(self):
        """变异对照：把 DROP_KEYS 清空（等价于没做这一步），文件里就会出现 `_current_code`。

        这条保证"写盘产物里没有 _current_code"这个断言真的在管行为。
        """
        original = ac.DROP_KEYS
        try:
            ac.DROP_KEYS = set()
            small, stats = ac.compact_payload(make_payload())
            self.assertIn("_current_code", json.dumps(small, ensure_ascii=False))
            self.assertNotIn("_current_code", stats)
        finally:
            ac.DROP_KEYS = original


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
