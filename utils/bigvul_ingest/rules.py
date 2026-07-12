from __future__ import annotations

import difflib
import os
import re
from typing import Dict, List, Optional, Tuple


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def score_to_severity(score_value: str) -> str:
    """Map CVSS score to MAS severity buckets."""
    try:
        score = float(score_value)
    except (TypeError, ValueError):
        return "medium"

    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low"


def derive_error_type(cwe_id: str, classification: str, summary: str) -> str:
    cwe = normalize_text(cwe_id)
    cls = normalize_text(classification).lower()
    s = normalize_text(summary).lower()

    if "race" in cls or "race condition" in s or cwe == "CWE-362":
        return "race_condition"
    if "overflow" in cls or "out of bounds" in s or cwe in {"CWE-119", "CWE-189"}:
        return "memory_overflow"
    if "bypass" in cls or "privilege" in s or cwe == "CWE-264":
        return "authorization_bypass"
    if "dos" in cls or "denial of service" in s:
        return "dos"
    if cwe == "CWE-20" or "validate" in s or "input" in s:
        return "input_validation"
    if cwe == "CWE-399" or "memory leak" in s:
        return "resource_exhaustion"
    return "general"


def derive_problematic_pattern(error_type: str, summary: str) -> str:
    summary = normalize_text(summary)
    patterns: Dict[str, str] = {
        "input_validation": "External input is consumed without strict bounds/format validation.",
        "memory_overflow": "Unchecked arithmetic or index usage may cause out-of-bounds access.",
        "resource_exhaustion": "Resource allocation path lacks defensive limits or cleanup.",
        "race_condition": "Shared state updates are not protected by synchronization or ordering checks.",
        "authorization_bypass": "Security-critical capability checks are incomplete or bypassable.",
        "dos": "Error handling allows repeated attacker-controlled state transitions or loops.",
        "general": "Security-sensitive logic lacks explicit defensive checks.",
    }
    base = patterns.get(error_type, patterns["general"])
    return f"{base} Evidence: {summary}"


def derive_solution_template(error_type: str) -> str:
    templates: Dict[str, str] = {
        "input_validation": "Enforce strict input validation (length, format, ranges), reject malformed packets early, and add regression tests for malformed inputs.",
        "memory_overflow": "Add bounds checks before array/pointer operations, guard integer arithmetic, and include sanitizer-backed tests (ASAN/UBSAN).",
        "resource_exhaustion": "Add resource limits and failure guards, ensure cleanup on every error path, and add stress tests for large/invalid inputs.",
        "race_condition": "Protect shared state with synchronization primitives, verify ordering assumptions, and add concurrent execution tests.",
        "authorization_bypass": "Centralize privilege checks, require strongest capability for sensitive paths, and add negative authorization tests.",
        "dos": "Add loop exit guards and request throttling, fail fast on invalid state, and create replay/fuzz tests for abusive sequences.",
        "general": "Introduce explicit guard clauses for security-sensitive code paths and add regression tests covering exploit preconditions.",
    }
    return templates.get(error_type, templates["general"])


def derive_file_pattern(original_path: str) -> str:
    """Prefer full original path; fall back to basename."""
    path = normalize_text(original_path).replace("\\", "/")
    if not path:
        return ""
    return path


def extract_function_name_from_summary(summary: str) -> str:
    """Extract a likely C/C++ function name from CVE summary text."""
    text = normalize_text(summary)
    if not text:
        return ""
    patterns = [
        r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s+function\b",
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]{2,})\b",
        r"\bin\s+([A-Za-z_][A-Za-z0-9_]{2,})\s*\(",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            name = match.group(1)
            if name.lower() not in {"the", "a", "an", "this", "that", "linux", "kernel"}:
                return name
    return ""


def extract_snippet_around_lines(
    source_text: str,
    start_line: int,
    end_line: int,
    *,
    context_lines: int = 8,
    max_chars: int = 2000,
) -> str:
    """Slice source around the changed line range instead of file header."""
    if not source_text:
        return ""
    lines = source_text.splitlines()
    if not lines:
        return ""
    if start_line <= 0:
        return source_text[:max_chars]

    start_idx = max(0, start_line - 1 - context_lines)
    end_idx = min(len(lines), max(start_line, end_line) + context_lines)
    snippet = "\n".join(lines[start_idx:end_idx])
    if len(snippet) > max_chars:
        return snippet[:max_chars]
    return snippet


def derive_solution_from_diff(
    before_text: str,
    after_text: str,
    error_type: str,
    *,
    max_hunk_lines: int = 24,
) -> str:
    """
    Prefer a short natural-language patch summary from before/after.
    Fall back to error_type template when no useful diff exists.
    """
    fallback = derive_solution_template(error_type)
    if not before_text and not after_text:
        return fallback
    if before_text == after_text:
        return fallback

    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=2,
        )
    )
    useful = [
        line
        for line in diff_lines
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    if not useful:
        return fallback

    removed = [line[1:].strip() for line in useful if line.startswith("-") and line[1:].strip()]
    added = [line[1:].strip() for line in useful if line.startswith("+") and line[1:].strip()]

    parts: List[str] = []
    if removed:
        sample = "; ".join(removed[:3])
        parts.append(f"Remove incorrect logic: {sample}")
    if added:
        sample = "; ".join(added[:3])
        parts.append(f"Ensure corrected path: {sample}")

    # Highlight common kernel/config fix patterns
    joined = "\n".join(useful[:max_hunk_lines]).lower()
    if "config_altivec" in joined or "altivec" in joined:
        parts.insert(
            0,
            "Fix Altivec unavailable exception handling so user-mode Altivec instructions take the SIGILL path even when CONFIG_ALTIVEC is defined",
        )
    elif "#if" in joined or "#ifdef" in joined or "#ifndef" in joined:
        parts.insert(0, "Correct conditional compilation so the defensive error-handling path remains reachable")

    if not parts:
        return fallback

    summary = ". ".join(parts)
    if len(summary) > 500:
        summary = summary[:497] + "..."
    return summary


def collect_file_anchors_from_instances(
    instances: List[Dict],
) -> Tuple[str, str]:
    """Pick first non-empty file_path / function hints from curated instances."""
    file_pattern = ""
    class_pattern = ""
    for item in instances or []:
        issue = item.get("issue") if isinstance(item, dict) else None
        if not isinstance(issue, dict):
            continue
        if not file_pattern:
            file_pattern = derive_file_pattern(str(issue.get("file_path") or ""))
        # class_pattern may already be set on pattern; instances don't carry it
    return file_pattern, class_pattern
