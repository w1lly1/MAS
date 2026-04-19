from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .rules import (
        derive_error_type,
        derive_problematic_pattern,
        derive_solution_template,
        normalize_text,
        score_to_severity,
    )
except ImportError:
    from rules import (
        derive_error_type,
        derive_problematic_pattern,
        derive_solution_template,
        normalize_text,
        score_to_severity,
    )


@dataclass
class BuildConfig:
    metadata_root: Path
    before_root: Path
    after_root: Path
    output_dir: Path
    start: int
    count: int
    max_snippet_chars: int
    session_id: str


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _find_top_cve_dirs(metadata_root: Path, start: int, count: int) -> List[Path]:
    cve_dirs = [p for p in metadata_root.iterdir() if p.is_dir() and p.name.startswith("CVE-")]
    cve_dirs.sort(key=lambda x: x.name)
    return cve_dirs[start : start + count]


def _find_commit_meta_paths(cve_dir: Path) -> List[Path]:
    result: List[Path] = []
    for child in sorted(cve_dir.iterdir(), key=lambda x: x.name):
        if child.is_dir():
            commit_meta = child / "commit_metadata.json"
            if commit_meta.exists():
                result.append(commit_meta)
    return result


def _changed_range(before_text: str, after_text: str) -> Tuple[int, int]:
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            start = i1 + 1
            end = i2 if i2 > i1 else start
            return start, end
    return 0, 0


def _build_issue_pattern_task(cve_meta: Dict[str, Any]) -> Dict[str, Any]:
    summary = normalize_text(cve_meta.get("summary", ""))
    cwe_id = normalize_text(cve_meta.get("cwe_id", ""))
    classification = normalize_text(cve_meta.get("vulnerability_classification", ""))
    score = str(cve_meta.get("score", ""))
    severity = score_to_severity(score)
    error_type = derive_error_type(cwe_id, classification, summary)

    data = {
        "title": normalize_text(cve_meta.get("cve_id", "")),
        "error_type": error_type,
        "severity": severity,
        "language": normalize_text(cve_meta.get("lang", "")),
        "framework": normalize_text(cve_meta.get("project", "")),
        "error_description": summary,
        "problematic_pattern": derive_problematic_pattern(error_type, summary),
        "solution": derive_solution_template(error_type),
        "file_pattern": "",
        "class_pattern": "",
        "tags": normalize_text(classification or cwe_id),
        "status": "active",
    }
    return {"target": "issue_pattern", "action": "upsert", "data": data}


def _split_summary(summary: str) -> Tuple[str, str]:
    summary = normalize_text(summary)
    if not summary:
        return "", ""
    # Split only on sentence boundaries (dot followed by whitespace),
    # which avoids breaking versions like "2.6.19" into fragments.
    parts = [s.strip() for s in re.split(r"(?<=\.)\s+", summary) if s.strip()]
    if not parts:
        return summary, ""
    phenomenon = parts[0]
    root_cause = parts[1] if len(parts) > 1 else summary
    return phenomenon, root_cause


def _build_curated_issue_tasks(
    cve_meta: Dict[str, Any],
    commit_meta_paths: List[Path],
    before_root: Path,
    after_root: Path,
    max_snippet_chars: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    summary = normalize_text(cve_meta.get("summary", ""))
    phenomenon, root_cause = _split_summary(summary)
    classification = normalize_text(cve_meta.get("vulnerability_classification", ""))
    cwe_id = normalize_text(cve_meta.get("cwe_id", ""))
    error_type = derive_error_type(cwe_id, classification, summary)
    solution = derive_solution_template(error_type)
    severity = score_to_severity(str(cve_meta.get("score", "")))
    cve_id = cve_meta.get("cve_id", "")

    for commit_meta_path in commit_meta_paths:
        commit_meta = _load_json(commit_meta_path)
        commit_dir = commit_meta_path.parent.name
        files = commit_meta.get("files", []) if isinstance(commit_meta.get("files"), list) else []

        for file_info in files:
            local_name = file_info.get("local_name", "")
            before_path = before_root / cve_id / commit_dir / local_name
            after_path = after_root / cve_id / commit_dir / local_name
            before_text = _safe_read_text(before_path)
            after_text = _safe_read_text(after_path)
            start_line, end_line = _changed_range(before_text, after_text)
            snippet = before_text[:max_snippet_chars] if before_text else ""

            data = {
                "project_path": str(before_root / cve_id),
                "file_path": normalize_text(file_info.get("original_path", "")),
                "start_line": start_line,
                "end_line": end_line,
                "code_snippet": snippet,
                "problem_phenomenon": phenomenon,
                "root_cause": root_cause,
                "solution": solution,
                "severity": severity,
                "status": "open",
            }
            tasks.append({"target": "curated_issue", "action": "upsert", "data": data})

    if not tasks:
        data = {
            "project_path": str(before_root / cve_id),
            "file_path": "",
            "start_line": 0,
            "end_line": 0,
            "code_snippet": "",
            "problem_phenomenon": phenomenon,
            "root_cause": root_cause,
            "solution": solution,
            "severity": severity,
            "status": "open",
        }
        tasks.append({"target": "curated_issue", "action": "upsert", "data": data})

    return tasks


def _validate_tasks(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    allowed_targets = {"review_session", "issue_pattern", "curated_issue"}
    allowed_actions = {"upsert", "create", "update", "query", "delete", "delete_all", "sync"}

    for idx, task in enumerate(tasks):
        target = task.get("target")
        action = task.get("action")
        data = task.get("data")
        if target not in allowed_targets:
            errors.append(f"task[{idx}] invalid target: {target}")
        if action not in allowed_actions:
            errors.append(f"task[{idx}] invalid action: {action}")
        if not isinstance(data, dict):
            errors.append(f"task[{idx}] data must be dict")
            continue

        if target == "issue_pattern":
            for key in ["error_type", "error_description", "problematic_pattern", "solution"]:
                if not normalize_text(data.get(key, "")):
                    errors.append(f"task[{idx}] issue_pattern missing {key}")
        if target == "curated_issue":
            for key in ["problem_phenomenon", "root_cause", "solution"]:
                if not normalize_text(data.get(key, "")):
                    warnings.append(f"task[{idx}] curated_issue weak field {key}")

    return {"errors": errors, "warnings": warnings, "valid": len(errors) == 0}


def _build_summary(tasks: List[Dict[str, Any]], cve_ids: List[str]) -> Dict[str, Any]:
    by_target: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}

    for task in tasks:
        target = task.get("target", "unknown")
        by_target[target] = by_target.get(target, 0) + 1
        sev = normalize_text(task.get("data", {}).get("severity", ""))
        if sev:
            by_severity[sev] = by_severity.get(sev, 0) + 1

    return {
        "cve_count": len(cve_ids),
        "cve_ids": cve_ids,
        "task_count": len(tasks),
        "by_target": by_target,
        "by_severity": by_severity,
    }


def build_payload(cfg: BuildConfig) -> Dict[str, Any]:
    cve_dirs = _find_top_cve_dirs(cfg.metadata_root, cfg.start, cfg.count)
    cve_ids: List[str] = []
    tasks: List[Dict[str, Any]] = []

    # Session task first: db_manage_agent will ensure system-owned fields as needed.
    session_task = {
        "target": "review_session",
        "action": "upsert",
        "data": {
            "session_id": cfg.session_id,
            "user_message": "BigVul top-20 metadata batch import",
            "code_directory": str(cfg.before_root),
            "status": "open",
            "code_patch": "",
            "git_commit": "",
        },
    }
    tasks.append(session_task)

    for cve_dir in cve_dirs:
        cve_meta_path = cve_dir / "cve_metadata.json"
        if not cve_meta_path.exists():
            continue
        cve_meta = _load_json(cve_meta_path)
        cve_id = normalize_text(cve_meta.get("cve_id", cve_dir.name))
        cve_ids.append(cve_id)

        tasks.append(_build_issue_pattern_task(cve_meta))
        commit_meta_paths = _find_commit_meta_paths(cve_dir)
        tasks.extend(
            _build_curated_issue_tasks(
                cve_meta,
                commit_meta_paths,
                cfg.before_root,
                cfg.after_root,
                cfg.max_snippet_chars,
            )
        )

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_metadata_root": str(cfg.metadata_root),
            "source_before_root": str(cfg.before_root),
            "source_after_root": str(cfg.after_root),
            "start": cfg.start,
            "count": cfg.count,
            "session_id": cfg.session_id,
        },
        "tasks": tasks,
        "summary": _build_summary(tasks, cve_ids),
    }


def write_outputs(payload: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / "bigvul_top20_db_tasks.json"
    summary_path = output_dir / "bigvul_top20_summary.json"
    validation_path = output_dir / "bigvul_top20_validation.json"

    validation = _validate_tasks(payload.get("tasks", []))

    tasks_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(payload.get("summary", {}), ensure_ascii=False, indent=2), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {tasks_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {validation_path}")
    print(f"Validation valid={validation['valid']} errors={len(validation['errors'])} warnings={len(validation['warnings'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MAS DB task JSON from BigVul metadata")
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/metadata"),
    )
    parser.add_argument(
        "--before-root",
        type=Path,
        default=Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/before"),
    )
    parser.add_argument(
        "--after-root",
        type=Path,
        default=Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/after"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("utils/bigvul_ingest/output"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-snippet-chars", type=int, default=2000)
    parser.add_argument("--session-id", type=str, default=f"bigvul-top20-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BuildConfig(
        metadata_root=args.metadata_root,
        before_root=args.before_root,
        after_root=args.after_root,
        output_dir=args.output_dir,
        start=args.start,
        count=args.count,
        max_snippet_chars=args.max_snippet_chars,
        session_id=args.session_id,
    )

    payload = build_payload(cfg)
    write_outputs(payload, cfg.output_dir)


if __name__ == "__main__":
    main()
