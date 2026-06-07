from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    output_name: str
    start: int
    count: int
    max_snippet_chars: int
    session_id: str
    ingest_mode: str


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clean_text(value: Any) -> str:
    text = normalize_text(value)
    if text.lower() == "nan":
        return ""
    return text


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


def _split_summary(summary: str) -> Tuple[str, str]:
    summary = normalize_text(summary)
    if not summary:
        return "", ""
    parts = [s.strip() for s in re.split(r"(?<=\.)\s+", summary) if s.strip()]
    if not parts:
        return summary, ""
    phenomenon = parts[0]
    root_cause = parts[1] if len(parts) > 1 else summary
    return phenomenon, root_cause


def _build_pattern(cve_meta: Dict[str, Any]) -> Dict[str, Any]:
    summary = _clean_text(cve_meta.get("summary", ""))
    cwe_id = _clean_text(cve_meta.get("cwe_id", ""))
    classification = _clean_text(cve_meta.get("vulnerability_classification", ""))
    score = str(cve_meta.get("score", ""))
    severity = score_to_severity(score)
    error_type = derive_error_type(cwe_id, classification, summary)
    tags = _clean_text(classification or cwe_id)

    return {
        "title": _clean_text(cve_meta.get("cve_id", "")),
        "error_type": error_type,
        "severity": severity,
        "language": _clean_text(cve_meta.get("lang", "")),
        "framework": _clean_text(cve_meta.get("project", "")),
        "error_description": summary,
        "problematic_pattern": derive_problematic_pattern(error_type, summary),
        "solution": derive_solution_template(error_type),
        "file_pattern": "",
        "class_pattern": "",
        "tags": tags,
        "status": "active",
    }


def _build_instances(
    cve_meta: Dict[str, Any],
    commit_meta_paths: List[Path],
    before_root: Path,
    after_root: Path,
    max_snippet_chars: int,
    session_id: str,
    session_message: str,
) -> List[Dict[str, Any]]:
    summary = _clean_text(cve_meta.get("summary", ""))
    phenomenon, root_cause = _split_summary(summary)
    classification = _clean_text(cve_meta.get("vulnerability_classification", ""))
    cwe_id = _clean_text(cve_meta.get("cwe_id", ""))
    error_type = derive_error_type(cwe_id, classification, summary)
    solution = derive_solution_template(error_type)
    severity = score_to_severity(str(cve_meta.get("score", "")))
    cve_id = _clean_text(cve_meta.get("cve_id", ""))
    project_path = _clean_text(cve_meta.get("project", ""))
    code_directory = (before_root / cve_id).as_posix()

    instances: List[Dict[str, Any]] = []

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

            instances.append(
                {
                    "session_meta": {
                        "session_id": session_id,
                        "user_message": session_message,
                        "code_directory": code_directory,
                    },
                    "issue": {
                        "project_path": project_path,
                        "file_path": _clean_text(file_info.get("original_path", "")),
                        "start_line": start_line,
                        "end_line": end_line,
                        "code_snippet": snippet,
                        "problem_phenomenon": phenomenon,
                        "root_cause": root_cause,
                        "solution": solution,
                        "severity": severity,
                        "status": "open",
                    },
                }
            )

    if not instances:
        instances.append(
            {
                "session_meta": {
                    "session_id": session_id,
                    "user_message": session_message,
                    "code_directory": code_directory,
                },
                "issue": {
                    "project_path": project_path,
                    "file_path": "",
                    "start_line": 0,
                    "end_line": 0,
                    "code_snippet": "",
                    "problem_phenomenon": phenomenon,
                    "root_cause": root_cause,
                    "solution": solution,
                    "severity": severity,
                    "status": "open",
                },
            }
        )

    return instances


def build_payload(cfg: BuildConfig) -> Dict[str, Any]:
    cve_dirs = _find_top_cve_dirs(cfg.metadata_root, cfg.start, cfg.count)
    data: List[Dict[str, Any]] = []

    session_message = f"Ingesting BigVul data range {cfg.start}-{cfg.start + cfg.count}"

    for cve_dir in cve_dirs:
        cve_meta_path = cve_dir / "cve_metadata.json"
        if not cve_meta_path.exists():
            continue
        cve_meta = _load_json(cve_meta_path)
        commit_meta_paths = _find_commit_meta_paths(cve_dir)

        data.append(
            {
                "pattern": _build_pattern(cve_meta),
                "instances": _build_instances(
                    cve_meta,
                    commit_meta_paths,
                    cfg.before_root,
                    cfg.after_root,
                    cfg.max_snippet_chars,
                    cfg.session_id,
                    session_message,
                ),
            }
        )

    return {
        "version": "1.0",
        "ingest_mode": cfg.ingest_mode,
        "data": data,
    }


def write_output(payload: Dict[str, Any], output_dir: Path, output_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured ingest JSON from BigVul metadata")
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
    parser.add_argument("--output-name", type=str, default="structured_ingest_sample.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-snippet-chars", type=int, default=2000)
    parser.add_argument(
        "--session-id",
        type=str,
        default=f"bigvul-structured-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
    )
    parser.add_argument("--ingest-mode", type=str, default="strict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = args.start
    count = args.count
    if args.end is not None:
        if args.end <= start:
            raise ValueError("--end must be greater than --start")
        count = args.end - start

    cfg = BuildConfig(
        metadata_root=args.metadata_root,
        before_root=args.before_root,
        after_root=args.after_root,
        output_dir=args.output_dir,
        output_name=args.output_name,
        start=start,
        count=count,
        max_snippet_chars=args.max_snippet_chars,
        session_id=args.session_id,
        ingest_mode=args.ingest_mode,
    )

    payload = build_payload(cfg)
    output_path = write_output(payload, cfg.output_dir, cfg.output_name)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
