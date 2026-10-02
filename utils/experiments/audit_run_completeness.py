"""历史批次完整性审计：找出「批处理记为成功、但 run 其实没跑完」的样本。

**为什么要它（T3）**
修复 R1 之前，批处理侧的等待函数拿不到"没跑完"这个信息，于是**干等超时后照样把
状态写成 `done`**。实测口径：等待上限 180s 时 Arm1 有 **12/30** 个样本是这种"假成功"，
上限改成 685s 后降到 **1/30**。历史批次（400 样本、8 样本冒烟、门控 A/B）都是在上限
提高之前跑的，所以**它们的召回数字里可能混着没跑完的样本**，必须抽查而不是断言。

**判据（与运行时同源，不自己发明）**
一个 requirement（一个待分析文件）只有在四类结果全到齐时才算完成：
    static_analysis / ai_analysis / security_analysis / performance_analysis
四类各自落盘为（见四个 agent 的 `generate_run_scoped_report` 调用）：

    agents/static/static_req_<id>.json
    agents/code_quality/quality_req_<id>.json
    agents/security/security_req_<id>.json
    agents/performance/performance_req_<id>.json

而 run 级完成产物是 run 根目录下的 `run_summary.json`（只有在**所有** requirement
都四类齐全后才会写）。因此：

    记为 done 但 run_summary.json 缺失   → **疑似未完成**
    记为 done 且存在"缺类的 requirement" → **确实未完成**（缺哪类、卡在哪个文件都能列出）

`REQUIRED_ANALYSIS_TYPES` 直接从生产模块 import；文件名模板另有契约测试
（`tests/test_run_completeness_contract.py`）盯住，防止两边漂移。

**用法**

    # 本地自验（合成样本，正/负对照 + 变异对照），不需要真机
    python -X utf8 utils/experiments/audit_run_completeness.py --selftest

    # 真机上审历史批次（reports/analysis 下的 run 目录 + 批处理汇总 CSV）
    python -X utf8 utils/experiments/audit_run_completeness.py \
        --reports-root reports/analysis \
        --batch-summary reports/batch_summary_seed2025.csv \
        --runs-file reports/arm1_runs.txt \
        --json-out reports/run_completeness_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
# 先补 local_libs（提供 weaviate 客户端/stub），再补仓库根：core 包的 __init__ 会连带导入
# agents_integration → weaviate，少了这一步 import 直接失败。
if str(REPO_ROOT / "local_libs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "local_libs"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agents.analysis_result_summary_agent import SummaryAgent  # noqa: E402

REQUIRED_ANALYSIS_TYPES = set(SummaryAgent.REQUIRED_ANALYSIS_TYPES)

# 分析类型 → (子目录, 文件名模板)。改这里必须同步改 agent 里的写入调用，
# `tests/test_run_completeness_contract.py` 会去四个 agent 源码里按模板原文核对。
REPORT_LAYOUT: Dict[str, Dict[str, str]] = {
    "static_analysis": {
        "agent_file": "core/agents/static_scan_agent.py",
        "subdir": "agents/static",
        "filename_template": "static_req_{requirement_id}.json",
    },
    "ai_analysis": {
        "agent_file": "core/agents/ai_driven_code_quality_agent.py",
        "subdir": "agents/code_quality",
        "filename_template": "quality_req_{requirement_id}.json",
    },
    "security_analysis": {
        "agent_file": "core/agents/ai_driven_security_agent.py",
        "subdir": "agents/security",
        "filename_template": "security_req_{requirement_id}.json",
    },
    "performance_analysis": {
        "agent_file": "core/agents/ai_driven_performance_agent.py",
        "subdir": "agents/performance",
        "filename_template": "performance_req_{requirement_id}.json",
    },
}

REQ_ID_PLACEHOLDER = "{requirement_id}"
KIND_TO_TYPE = {
    "static": "static_analysis",
    "quality": "ai_analysis",
    "security": "security_analysis",
    "performance": "performance_analysis",
}
# 由模板生成匹配正则（模板里的花括号是字面量，不做正则转义歧义）
_REQ_FILE_RE = re.compile("^(?P<kind>%s)_req_(?P<req>\\d+)\\.json$" % "|".join(KIND_TO_TYPE))


def report_filename(type_name: str, requirement_id: int) -> str:
    """按生产约定拼出某类分析结果的文件名。"""
    return REPORT_LAYOUT[type_name]["filename_template"].replace(
        REQ_ID_PLACEHOLDER, str(requirement_id))


def _agent_req_files(run_dir: Path) -> Dict[int, Set[str]]:
    """扫 run 目录，返回 {requirement_id: {分析类型, ...}}。"""
    found: Dict[int, Set[str]] = {}
    for type_name, layout in REPORT_LAYOUT.items():
        sub = run_dir / layout["subdir"]
        if not sub.is_dir():
            continue
        for path in sub.glob("*_req_*.json"):
            m = _REQ_FILE_RE.match(path.name)
            if not m:
                continue
            if KIND_TO_TYPE[m.group("kind")] != type_name:
                continue
            found.setdefault(int(m.group("req")), set()).add(type_name)
    return found


def scan_run(run_dir: Path) -> Dict[str, object]:
    """体检单个 run 目录。纯函数式（只读，不写盘），方便测试与变异对照。"""
    agent_files = _agent_req_files(run_dir)
    summary_path = run_dir / "run_summary.json"
    has_summary = summary_path.is_file()
    report_status = None
    if has_summary:
        try:
            report_status = json.loads(summary_path.read_text(encoding="utf-8")).get("status")
        except Exception:  # noqa: BLE001 - 坏 JSON 不该让审计崩掉
            report_status = "unreadable"

    expected: Set[int] = set(agent_files)
    incomplete = {req: sorted(REQUIRED_ANALYSIS_TYPES - types)
                  for req, types in agent_files.items()
                  if REQUIRED_ANALYSIS_TYPES - types}

    if not agent_files and not has_summary:
        verdict = "no_evidence"          # 归档片段/空目录，无法判定
    elif not has_summary:
        verdict = "incomplete"           # 没有 run 级完成产物
    elif incomplete:
        verdict = "incomplete"           # 有完成产物但仍有缺类（异常，值得单列）
    else:
        verdict = "complete"

    # 证据强度：run_summary.json 是**运行时**在"所有 requirement 四类齐全"后才写的产物，
    # 有它就能直接下"跑完了"的结论；只有 agent 报告文件时，"缺类"也可能是**复制/归档时
    # 挑掉了大文件**造成的假象 → 必须在原始 run 目录上复核，不能就地断言。
    if has_summary:
        confidence = "high"
    elif agent_files:
        confidence = "needs_original_dir"
    else:
        confidence = "unknown"

    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "verdict": verdict,
        "confidence": confidence,
        # 结论是靠什么下的：run_summary.json 是"跑完了"的强证据，agent 报告文件用于定位缺哪类
        "evidence_basis": ("run_summary+agent_files"
                           if has_summary and agent_files
                           else ("run_summary" if has_summary else "agent_files")),
        "has_run_summary": has_summary,
        "run_summary_status": report_status,
        "expected_requirements": len(expected),
        "complete_requirements": len(expected) - len(incomplete),
        "incomplete_requirements": {str(k): v for k, v in sorted(incomplete.items())},
        "missing_types_total": sum(len(v) for v in incomplete.values()),
    }


def iter_run_dirs(reports_root: Path):
    """reports/analysis 下的结构是 <输出目录>/<run_id>/。

    只有"像 run 的目录"才纳入审计：它要么有 run_summary.json（跑完了），要么有 agents/
    子目录（至少写出一类结果，即 partial）。其余的（归档片段、空目录）**不静默丢弃**，
    而是单独返回，让审计报告里能看到"跳过了谁、为什么跳"。
    """
    run_dirs: List[Path] = []
    skipped: List[str] = []
    if not reports_root.is_dir():
        return run_dirs, skipped
    for child in sorted(reports_root.iterdir()):
        if not child.is_dir():
            continue
        for run_dir in sorted(child.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "agents").is_dir() or (run_dir / "run_summary.json").is_file():
                run_dirs.append(run_dir)
            else:
                skipped.append(str(run_dir))
    return run_dirs, skipped


def load_batch_summary(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """run_id → 批处理汇总行（含旧代码无条件写的 status）。"""
    if not path or not path.is_file():
        return {}
    rows = {}
    with path.open(encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        for row in csv.DictReader(fh, dialect=dialect):
            rid = (row.get("run_id") or "").strip()
            if rid:
                rows[rid] = row
    return rows


def load_runs_file(path: Optional[Path]) -> List[str]:
    if not path or not path.is_file():
        return []
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.split("/")[-1])
    return ids


def audit(reports_root: Path, batch_summary: Optional[Path] = None,
          runs_file: Optional[Path] = None) -> Dict[str, object]:
    recorded = load_batch_summary(batch_summary)
    run_dirs, skipped_dirs = iter_run_dirs(reports_root)
    if runs_file:
        wanted = {rid: None for rid in load_runs_file(runs_file)}
    else:
        wanted = None

    rows = []
    for run_dir in run_dirs:
        row = scan_run(run_dir)
        if wanted is not None and row["run_id"] not in wanted:
            continue
        rec = recorded.get(str(row["run_id"]))
        row["recorded_status"] = (rec or {}).get("status")
        row["recorded_cve"] = (rec or {}).get("cve")
        row["recorded_role"] = (rec or {}).get("role")
        # 关键口径：批处理记成功，而 run 目录说明它没跑完
        row["false_success"] = bool(
            row["recorded_status"] == "done" and row["verdict"] == "incomplete")
        rows.append(row)
        if wanted is not None:
            wanted.pop(str(row["run_id"]), None)

    missing_dirs = [rid for rid, _ in (wanted or {}).items()]

    by_verdict: Dict[str, int] = {}
    for row in rows:
        by_verdict[row["verdict"]] = by_verdict.get(row["verdict"], 0) + 1
    false_success = [row for row in rows if row["false_success"]]
    # 头号数字只认高置信（有 run_summary 判定/或原始目录可复核）的；免得被"挑选副本"污染
    hard_false = [row for row in false_success if row["confidence"] != "needs_original_dir"]

    return {
        "reports_root": str(reports_root),
        "batch_summary": str(batch_summary) if batch_summary else None,
        "runs_file": str(runs_file) if runs_file else None,
        "runs_scanned": len(rows),
        "by_verdict": by_verdict,
        "false_success_count": len(false_success),
        "false_success_high_confidence_count": len(hard_false),
        "false_success_runs": [row["run_id"] for row in false_success],
        "needs_confirmation_runs": [row["run_id"] for row in false_success
                                    if row["confidence"] == "needs_original_dir"],
        "missing_run_dirs": missing_dirs,
        "skipped_dirs": skipped_dirs,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 本地自验：合成 run 目录 + 正/负对照 + 变异对照
# ---------------------------------------------------------------------------

def _make_run(root: Path, rel: str, reqs: Dict[int, List[str]],
              with_summary: bool = True) -> Path:
    run_dir = root / rel
    run_dir.mkdir(parents=True, exist_ok=True)
    for req, types in reqs.items():
        for type_name in types:
            sub = run_dir / REPORT_LAYOUT[type_name]["subdir"]
            sub.mkdir(parents=True, exist_ok=True)
            fname = report_filename(type_name, req)
            (sub / fname).write_text(json.dumps({"requirement_id": req}), encoding="utf-8")
    if with_summary:
        (run_dir / "run_summary.json").write_text(
            json.dumps({"status": "run_completed", "run_id": run_dir.name}), encoding="utf-8")
    return run_dir


def selftest() -> int:
    """合成四个 run：完整 / 缺类 / 记成功但无 run_summary / 归档片段。
    然后做一次**变异**（把完整的那份删掉一个类），要求判定从 complete 翻成 incomplete。
    """
    sandbox = REPO_ROOT / "reports" / "_selftest_run_completeness"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    all4 = sorted(REQUIRED_ANALYSIS_TYPES)

    _make_run(sandbox, "CVE-A/run_complete", {1: all4, 2: all4})
    _make_run(sandbox, "CVE-B/run_missing_type", {1: all4, 2: all4[:3]}, with_summary=False)
    _make_run(sandbox, "CVE-C/run_no_summary", {1: all4}, with_summary=False)
    frag = sandbox / "CVE-D/run_fragment"
    frag.mkdir(parents=True, exist_ok=True)
    (frag / "notes.txt").write_text("归档片段，无任何 agent 产物", encoding="utf-8")

    case_csv = sandbox / "batch_summary.csv"
    case_csv.write_text(
        "cve,role,status,run_id,security_strategy,llm_active,vulns_detected,raw_text_length\n"
        "CVE-A,kb,done,run_complete,hybrid_fusion,1,5,1460\n"
        "CVE-B,kb,done,run_missing_type,hybrid_fusion,1,4,1440\n"
        "CVE-C,kb,done,run_no_summary,hybrid_fusion,1,3,1400\n"
        "CVE-D,kb,done,run_fragment,hybrid_fusion,1,0,10\n",
        encoding="utf-8")

    result = audit(sandbox, case_csv)
    got = {row["run_id"]: row["verdict"] for row in result["rows"]}
    expect = {
        "run_complete": "complete",
        "run_missing_type": "incomplete",
        "run_no_summary": "incomplete",
    }
    checks = [("判定 %s" % rid, got.get(rid) == want) for rid, want in expect.items()]
    checks.append(("记成功但未完成 = 2", result["false_success_count"] == 2))
    # 归档片段不该被当成 run，但也不能被静默丢掉
    checks.append(("归档片段被跳过且已记录",
                   "run_fragment" not in got
                   and any(s.endswith("run_fragment") for s in result["skipped_dirs"])))
    # 缺类定位：run_missing_type 的第 2 个 requirement 应恰好缺 1 类
    row_missing = next(r for r in result["rows"] if r["run_id"] == "run_missing_type")
    checks.append(("缺类能定位到 requirement",
                   row_missing["missing_types_total"] == 1
                   and list(row_missing["incomplete_requirements"]) == ["2"]))

    # 变异对照：把"完整"那份删掉一个类，判定必须翻转
    victim = sandbox / "CVE-A/run_complete" / REPORT_LAYOUT["security_analysis"]["subdir"]
    removed = None
    for path in sorted(victim.glob("*.json")):
        path.unlink()
        removed = path.name
        break
    mutated = {r["run_id"]: r["verdict"] for r in audit(sandbox, case_csv)["rows"]}
    checks.append(("变异对照：删掉 %s 后 complete→incomplete" % removed,
                   mutated.get("run_complete") == "incomplete"))

    print("=" * 78)
    print("本地自验（合成 run 目录；判据取自生产 REQUIRED_ANALYSIS_TYPES）")
    print("=" * 78)
    print("  要求四类：%s" % ", ".join(sorted(REQUIRED_ANALYSIS_TYPES)))
    print("  判定结果：%s" % json.dumps(got, ensure_ascii=False))
    print("  跳过（非 run 目录）：%s" % json.dumps(result["skipped_dirs"], ensure_ascii=False))
    print("  记成功但未完成：%d 条 → %s" % (result["false_success_count"],
                                            result["false_success_runs"]))
    ok = True
    for name, passed in checks:
        ok = ok and passed
        print("  [%s] %s" % ("OK" if passed else "NG", name))
    print("\n  结论：%s" % ("自验通过：判据、缺类定位、变异对照都对得上"
                            if ok else "自验失败，判据有问题，别拿去审真机数据"))
    print("  （自验沙箱留在 %s，可人工翻看；确认后可删）" % sandbox)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="历史批次完整性审计（T3）")
    ap.add_argument("--reports-root", default="reports/analysis",
                    help="run 目录所在根（结构 <输出目录>/<run_id>/）")
    ap.add_argument("--batch-summary", default=None, help="批处理汇总 CSV（含 status 列）")
    ap.add_argument("--runs-file", default=None, help="只审这些 run（每行 CVE/run_id）")
    ap.add_argument("--json-out", default=None, help="把完整结果写成 JSON")
    ap.add_argument("--selftest", action="store_true", help="本地自验（合成数据）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    root = Path(args.reports_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    summary = Path(args.batch_summary) if args.batch_summary else None
    runs = Path(args.runs_file) if args.runs_file else None

    result = audit(root, summary, runs)
    print("=" * 78)
    print("完整性审计：%s" % result["reports_root"])
    print("=" * 78)
    print("  扫到 run 目录：%d" % result["runs_scanned"])
    print("  判定分布：%s" % json.dumps(result["by_verdict"], ensure_ascii=False))
    print("  **记为成功但实际未完成：%d 条**（其中高置信 %d 条）"
          % (result["false_success_count"], result["false_success_high_confidence_count"]))
    for rid in result["false_success_runs"][:40]:
        print("     - %s" % rid)
    if result["needs_confirmation_runs"]:
        print("  ⚠ 需在**原始 run 目录**复核：%d 条（本地只有挑选副本时，"
              "「缺类」可能是复制造成的假象）→ %s"
              % (len(result["needs_confirmation_runs"]),
                 ", ".join(result["needs_confirmation_runs"][:10])))
    if len(result["false_success_runs"]) > 40:
        print("     … 其余 %d 条见 JSON" % (len(result["false_success_runs"]) - 40))
    if result["missing_run_dirs"]:
        print("  列表里有、但目录不存在：%d 条（旧批次目录可能已被压缩/清理）"
              % len(result["missing_run_dirs"]))
    no_ev = [r["run_id"] for r in result["rows"] if r["verdict"] == "no_evidence"]
    if no_ev:
        print("  无任何产物、无法判定：%d 条 → %s%s"
              % (len(no_ev), ", ".join(no_ev[:10]), " …" if len(no_ev) > 10 else ""))
    if result["skipped_dirs"]:
        print("  跳过的非 run 目录：%d 个 → %s%s"
              % (len(result["skipped_dirs"]), ", ".join(
                  Path(d).name for d in result["skipped_dirs"][:10]),
                 " …" if len(result["skipped_dirs"]) > 10 else ""))

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print("  明细已写入 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
