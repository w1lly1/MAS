from __future__ import annotations

from typing import Dict


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
