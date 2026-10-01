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
        VALID_ERROR_TYPES,
        derive_error_type,
        derive_file_pattern,
        derive_problematic_pattern,
        derive_solution_from_diff,
        derive_solution_template,
        extract_function_name_from_summary,
        extract_snippet_around_lines,
        normalize_text,
        score_to_severity,
    )
except ImportError:
    from rules import (
        VALID_ERROR_TYPES,
        derive_error_type,
        derive_file_pattern,
        derive_problematic_pattern,
        derive_solution_from_diff,
        derive_solution_template,
        extract_function_name_from_summary,
        extract_snippet_around_lines,
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
    # 旁挂的"大模型语义理解"文件（{CVE: 文本}），可选。为空 → llm_semantic 为空 →
    # semantic/full 两层层文本与加该字段之前逐字节相同（既有向量不受影响）。
    llm_semantic_path: Optional[Path] = None


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


def _build_pattern(
    cve_meta: Dict[str, Any],
    *,
    file_pattern: str = "",
    class_pattern: str = "",
    solution: str = "",
    llm_semantic: str = "",
    llm_family: str = "",
) -> Dict[str, Any]:
    summary = _clean_text(cve_meta.get("summary", ""))
    cwe_id = _clean_text(cve_meta.get("cwe_id", ""))
    classification = _clean_text(cve_meta.get("vulnerability_classification", ""))
    score = str(cve_meta.get("score", ""))
    severity = score_to_severity(score)
    rule_family = derive_error_type(cwe_id, classification, summary)
    # 分类以**大模型读了代码之后的判断**为准（合法值才采纳），规则作为兜底。
    #
    # 为什么让模型赢：实测 8 个样本里，模型**没被告知答案**时选的家族与库分类一致率
    # 只有 4/8，而分歧几乎全是"摘要说的是影响（DoS），代码看起来是内存安全"这类情形 ——
    # 摘要描述影响、代码体现机制，而这一列应该描述**机制**。
    # 另外 `problematic_pattern`（模式描述句）**是按家族选的**，家族错了那句话也就错了。
    family = llm_family if str(llm_family or "").strip().lower() in VALID_ERROR_TYPES \
        else rule_family
    source = "llm" if family == str(llm_family or "").strip().lower() and family != rule_family \
        else ("llm" if str(llm_family or "").strip().lower() in VALID_ERROR_TYPES else "rules")
    if not class_pattern:
        class_pattern = extract_function_name_from_summary(summary)
    if not solution:
        solution = derive_solution_template(family)

    return {
        "title": _clean_text(cve_meta.get("cve_id", "")),
        "error_type": family,
        "severity": severity,
        "language": _clean_text(cve_meta.get("lang", "")),
        "framework": _clean_text(cve_meta.get("project", "")),
        "error_description": summary,
        "problematic_pattern": derive_problematic_pattern(family, summary),
        "solution": solution,
        "file_pattern": file_pattern,
        "class_pattern": class_pattern,
        # tags 里记下分类来源与规则原判，便于审计"哪些条目是被模型改过分类的"
        "tags": "|".join([t for t in (_clean_text(classification or cwe_id),
                                      "error_type_source=%s" % source,
                                      "rule_family=%s" % rule_family) if t]),
        "status": "active",
        # 大模型对该代码的语义理解（英文；功能 + 风险，语域对齐本库的其它散文）。
        # **只进 semantic / full 两层索引文本**，见 weaviate/service.py 的层构造器。
        # 由调用方通过旁挂文件提供（--llm-semantic）；不提供时为空 → 层文本与以前逐字节相同。
        "llm_semantic": _clean_text(llm_semantic),
    }


def load_llm_semantic(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """读"大模型语义理解"的旁挂文件，两种写法都支持：

        {"CVE-X": "两句英文描述"}                      ← 只有文本
        {"CVE-X": {"text": "...", "family": "dos"}}    ← 文本 + 模型判定的弱点家族

    返回值统一成 `{CVE: {"text": ..., "family": ...}}`（family 可能为空串）。

    做成旁挂文件、而不是塞进数据集 metadata，原因有三：
      · LLM 的产出是**后加的**、可重跑、可换模型，不该污染原始数据集；
      · 数据集是公共输入，写进去之后无法区分"原样"与"我们加工的"；
      · 旁挂文件缺失/为空 → 字段为空 → 层文本与以前逐字节相同，**不会**意外改变既有向量。
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("llm_semantic 旁挂文件应为 {CVE: 文本 或 {text,family}} 的对象")
    out: Dict[str, Dict[str, str]] = {}
    for k, v in data.items():
        key = _clean_text(k).upper()
        if not key:
            continue
        if isinstance(v, dict):
            text = _clean_text(v.get("text", ""))
            fam = _clean_text(v.get("family", "")).lower()
        else:
            text, fam = _clean_text(v), ""
        if text or fam:
            out[key] = {"text": text, "family": fam}
    return out


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
            # 跳过非代码文件（NEWS/README/ChangeLog/TESTLIST 等）：
            # 它们会被 derive_file_pattern 变成泛化的 file_pattern，导致跨项目误报。
            _ext = Path(local_name).suffix.lower()
            if _ext not in {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".s", ".S"}:
                continue
            before_path = before_root / cve_id / commit_dir / local_name
            after_path = after_root / cve_id / commit_dir / local_name
            before_text = _safe_read_text(before_path)
            after_text = _safe_read_text(after_path)
            start_line, end_line = _changed_range(before_text, after_text)
            snippet = extract_snippet_around_lines(
                before_text,
                start_line,
                end_line,
                max_chars=max_snippet_chars,
            )
            solution = derive_solution_from_diff(before_text, after_text, error_type)

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
                        "status": "resolved",
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
                    "solution": derive_solution_template(error_type),
                    "severity": severity,
                    "status": "resolved",
                },
            }
        )

    return instances


def build_payload(cfg: BuildConfig) -> Dict[str, Any]:
    cve_dirs = _find_top_cve_dirs(cfg.metadata_root, cfg.start, cfg.count)
    data: List[Dict[str, Any]] = []

    session_message = f"Ingesting BigVul data range {cfg.start}-{cfg.start + cfg.count}"
    # 大模型语义理解旁挂文件（可选）；缺失/为空都不影响其它字段
    llm_semantic = load_llm_semantic(getattr(cfg, "llm_semantic_path", None))

    for cve_dir in cve_dirs:
        cve_meta_path = cve_dir / "cve_metadata.json"
        if not cve_meta_path.exists():
            continue
        cve_meta = _load_json(cve_meta_path)
        commit_meta_paths = _find_commit_meta_paths(cve_dir)
        instances = _build_instances(
            cve_meta,
            commit_meta_paths,
            cfg.before_root,
            cfg.after_root,
            cfg.max_snippet_chars,
            cfg.session_id,
            session_message,
        )
        _key = _clean_text(cve_meta.get("cve_id", "")).upper()
        first_issue = (instances[0].get("issue") or {}) if instances else {}
        file_pattern = derive_file_pattern(str(first_issue.get("file_path") or ""))
        class_pattern = extract_function_name_from_summary(_clean_text(cve_meta.get("summary", "")))
        solution = str(first_issue.get("solution") or "")

        data.append(
            {
                "pattern": _build_pattern(
                    cve_meta,
                    file_pattern=file_pattern,
                    class_pattern=class_pattern,
                    solution=solution,
                    llm_semantic=(llm_semantic.get(_key, {}) or {}).get("text", ""),
                    llm_family=(llm_semantic.get(_key, {}) or {}).get("family", ""),
                ),
                "instances": instances,
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
    parser.add_argument(
        "--llm-semantic",
        type=Path,
        default=None,
        help="旁挂的『大模型语义理解』文件（{CVE: 文本}）。它只进 semantic/full 两层索引文本；"
             "不提供则该字段为空，层文本与以前逐字节相同。",
    )
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
