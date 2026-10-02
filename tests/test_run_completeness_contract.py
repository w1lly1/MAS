"""完整性审计的契约与判据测试（T3）。

两件事必须被钉住，否则审计工具会**悄悄失准**：

1. **落盘契约**：审计工具靠"文件名 + 子目录"识别四类结果。这套约定写死在四个 agent 里，
   一旦有人改了写入路径或文件名，审计会开始漏判（把跑完的说成没跑完，或反过来）。
   所以这里**去源码里核对**模板原文，而不是核对审计工具自己抄的一份常量。
2. **判据同源**：完成判据必须是运行时用的那四个分析类型集合，不能在工具里另写一套。

另外用合成 run 目录验判定逻辑本身，并做一次**变异对照**（删一个类，判定必须翻转）。
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "local_libs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "local_libs"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agents.analysis_result_summary_agent import SummaryAgent  # noqa: E402
from utils.experiments.audit_run_completeness import (  # noqa: E402
    REPORT_LAYOUT,
    REQUIRED_ANALYSIS_TYPES,
    audit,
    report_filename,
    scan_run,
)


@pytest.fixture()
def workdir():
    """自管的临时目录（**不用 pytest 的 tmp_path**）。

    原因：本机沙箱下 pytest 的 basetemp 走 `\\\\?\\` 长路径删除，会被拒（WinError 5），
    整个 session 在 teardown 时炸掉。这里把夹具放在仓库内，创建/删除都由自己负责。
    """
    path = REPO_ROOT / "reports" / "_test_run_completeness" / uuid.uuid4().hex[:8]
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_required_types_match_production():
    """判据同源：工具里的集合必须就是运行时那个集合。"""
    assert REQUIRED_ANALYSIS_TYPES == set(SummaryAgent.REQUIRED_ANALYSIS_TYPES)
    assert len(REQUIRED_ANALYSIS_TYPES) == 4


def test_report_layout_matches_agent_sources():
    """落盘契约：四个 agent 源码里必须真的存在我们认的模板与子目录。

    这是**结构性守卫**：它不跑 agent，只核对"写文件那一行"还在不在原处。
    """
    assert set(REPORT_LAYOUT) == REQUIRED_ANALYSIS_TYPES
    for type_name, layout in REPORT_LAYOUT.items():
        src = REPO_ROOT / layout["agent_file"]
        assert src.is_file(), "agent 源码不见了：%s" % src
        text = src.read_text(encoding="utf-8")
        assert 'subdir="%s"' % layout["subdir"] in text, (
            "%s 里找不到 subdir=%r —— 落盘目录变了，审计工具要同步"
            % (layout["agent_file"], layout["subdir"]))
        assert layout["filename_template"] in text, (
            "%s 里找不到文件名模板 %r —— 文件名变了，审计工具要同步"
            % (layout["agent_file"], layout["filename_template"]))


def test_report_filename_builds_real_names():
    assert report_filename("static_analysis", 1002) == "static_req_1002.json"
    assert report_filename("ai_analysis", 7) == "quality_req_7.json"
    assert report_filename("security_analysis", 1) == "security_req_1.json"
    assert report_filename("performance_analysis", 42) == "performance_req_42.json"


def _make_run(run_dir: Path, reqs, with_summary=True) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    for req, types in reqs.items():
        for type_name in types:
            sub = run_dir / REPORT_LAYOUT[type_name]["subdir"]
            sub.mkdir(parents=True, exist_ok=True)
            (sub / report_filename(type_name, req)).write_text("{}", encoding="utf-8")
    if with_summary:
        (run_dir / "run_summary.json").write_text(
            json.dumps({"status": "run_completed"}), encoding="utf-8")
    return run_dir


ALL4 = sorted(REQUIRED_ANALYSIS_TYPES)


def test_scan_run_complete(workdir):
    run = _make_run(workdir / "CVE-X" / "run1", {1: ALL4})
    row = scan_run(run)
    assert row["verdict"] == "complete"
    assert row["confidence"] == "high"
    assert row["missing_types_total"] == 0


def test_scan_run_detects_missing_type_and_locates_requirement(workdir):
    run = _make_run(workdir / "CVE-X" / "run2", {1: ALL4, 2: ALL4[:3]}, with_summary=False)
    row = scan_run(run)
    assert row["verdict"] == "incomplete"
    # 没有 run_summary → 只有 agent 文件这一路证据，必须提示去原始目录复核
    assert row["confidence"] == "needs_original_dir"
    assert list(row["incomplete_requirements"]) == ["2"]
    assert row["missing_types_total"] == 1


def test_mutation_removing_one_type_flips_verdict(workdir):
    """变异对照：完整 → 删掉一类 → 必须变成 incomplete。"""
    run = _make_run(workdir / "CVE-X" / "run3", {1: ALL4})
    assert scan_run(run)["verdict"] == "complete"
    victim = run / REPORT_LAYOUT["security_analysis"]["subdir"] / report_filename(
        "security_analysis", 1)
    victim.unlink()
    row = scan_run(run)
    assert row["verdict"] == "incomplete"
    assert row["incomplete_requirements"] == {"1": ["security_analysis"]}


def test_false_success_requires_recorded_done(workdir):
    """"记为成功但未完成"必须同时满足两条：批处理记 done + run 目录不完整。"""
    root = workdir / "analysis"
    _make_run(root / "CVE-Y" / "run_partial", {1: ALL4[:2]}, with_summary=False)
    _make_run(root / "CVE-Y" / "run_ok", {1: ALL4})
    csv_path = workdir / "batch_summary.csv"
    csv_path.write_text(
        "cve,role,status,run_id\n"
        "CVE-Y,kb,done,run_partial\n"
        "CVE-Y,kb,done,run_ok\n",
        encoding="utf-8")
    result = audit(root, csv_path)
    assert result["runs_scanned"] == 2
    assert result["false_success_runs"] == ["run_partial"]
    # 本地挑选副本造成的"缺类"不该计入头号数字
    assert result["false_success_high_confidence_count"] == 0
    assert result["needs_confirmation_runs"] == ["run_partial"]


def test_fragment_dir_is_skipped_not_silently_dropped(workdir):
    root = workdir / "analysis"
    frag = root / "CVE-Z" / "run_fragment"
    frag.mkdir(parents=True)
    (frag / "notes.txt").write_text("归档片段", encoding="utf-8")
    result = audit(root)
    assert result["runs_scanned"] == 0
    assert any(s.endswith("run_fragment") for s in result["skipped_dirs"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
