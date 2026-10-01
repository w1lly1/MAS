from __future__ import annotations

import difflib
import re
from typing import Dict, List, Tuple

# 知识库的**错误分类体系**：全库只有这 7 类。索引侧的层文本会写成 `[error_type] <值>`，
# 生成 `problematic_pattern` 也按这个值选句子，所以两侧（以及大模型那边）必须共用同一套。
VALID_ERROR_TYPES = (
    "race_condition",
    "resource_exhaustion",
    "memory_overflow",
    "authorization_bypass",
    "input_validation",
    "dos",
    "general",
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


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


def normalize_for_match(value: str) -> str:
    """把文本归一成"**小写 + 所有非字母数字都变成空格**"的形式，供关键词匹配。

    ## 为什么必须归一（这是实测踩到的 bug）

    原实现直接在原文里找 `"out of bounds"`（空格写法），而 CVE 摘要里写的是
    `"out-of-bounds"`（**带连字符**）—— 匹配失败，于是一条明确的"越界读"被归到 `general`。
    `CVE-2018-20854` 就是这么被判错的。

    连字符只是其中一种：下划线、斜杠、逗号、多空格、大小写，都会造成**同类漏匹配**。
    所以先统一归一，再用**词边界正则**去匹配 —— 这样 `out-of-bounds`、`out_of_bounds`、
    `out of  bounds`、`Out Of Bounds` 全都会命中同一个模式。
    """
    return " ".join(_NON_ALNUM.split(str(value or "").lower())).strip()


def _has(haystack: str, *patterns: str) -> bool:
    """在**已归一的**文本上按词边界匹配（patterns 也必须是归一写法）。"""
    for p in patterns:
        if re.search(r"\b%s\b" % re.escape(p), haystack):
            return True
    return False


# CWE → 家族。CWE 是权威信息，能对上就直接用，不再看关键词。
CWE_TO_TYPE: Dict[str, str] = {}
for _type, _cwes in {
    "race_condition": ("CWE-362", "CWE-364", "CWE-366", "CWE-367", "CWE-368", "CWE-370"),
    "resource_exhaustion": ("CWE-399", "CWE-400", "CWE-401", "CWE-404", "CWE-770", "CWE-771"),
    "memory_overflow": ("CWE-119", "CWE-120", "CWE-121", "CWE-122", "CWE-124", "CWE-125",
                        "CWE-126", "CWE-127", "CWE-129", "CWE-131", "CWE-189", "CWE-190",
                        "CWE-191", "CWE-680", "CWE-787", "CWE-788", "CWE-805", "CWE-823",
                        "CWE-824"),
    "authorization_bypass": ("CWE-264", "CWE-269", "CWE-285", "CWE-287", "CWE-732", "CWE-862",
                             "CWE-863", "CWE-639"),
    "input_validation": ("CWE-20", "CWE-22", "CWE-74", "CWE-78", "CWE-79", "CWE-89", "CWE-90",
                         "CWE-94", "CWE-134", "CWE-434", "CWE-502", "CWE-611", "CWE-918"),
    "dos": ("CWE-617", "CWE-674", "CWE-770", "CWE-834", "CWE-835", "CWE-1333"),
}.items():
    for _c in _cwes:
        CWE_TO_TYPE.setdefault(_c, _type)

# 关键词规则。**顺序 = 优先级**，并且刻意把"机制"排在"影响"前面：
# 很多 CVE 的 CVSS 影响是 DoS，但根因是内存越界；分类应该描述**机制**，
# 否则 `problematic_pattern` 会写成"错误处理允许重复状态转换"这种与代码不符的话。
#   （旧实现把 dos 排在较后是对的，但它同时把 `input` 这种过宽的词也当规则，会误吞很多条目。）
KEYWORD_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    # 竞态：说"race"基本就确定
    ("race_condition", ("race condition", "race conditions", "data race", "race",
                        "tocttou", "toctou", "double fetch")),
    # 资源耗尽：必须有"泄漏/耗尽/上限"这类词，不能只靠 "resource"
    ("resource_exhaustion", ("memory leak", "resource leak", "resource exhaustion",
                             "exhausts memory", "exhausts resources", "fails to release",
                             "not released", "unbounded memory", "out of memory")),
    # 内存越界：机制类，优先级高于 dos
    ("memory_overflow", ("out of bounds", "out of bound", "outofbounds", "bounds check",
                         "buffer overflow", "heap overflow", "stack overflow", "heap based buffer",
                         "integer overflow", "integer underflow", "off by one", "offbyone",
                         "overread", "over read", "out of range access", "memory corruption",
                         "use after free", "useafterfree", "buffer over read", "wild write",
                         "oob", "underflow", "overflow", "overwrite the buffer")),
    # 权限/授权
    ("authorization_bypass", ("bypass", "bypasses", "privilege escalation", "escalate privileges",
                              "improper authorization", "missing authorization",
                              "insufficient permission", "permission check", "access control",
                              "capability check", "sandbox escape", "authentication bypass")),
    # 输入校验：**不要**用裸 "input"（"input file"/"input buffer" 会误吞大量条目）
    ("input_validation", ("input validation", "validation", "validate", "validates",
                          "not validated", "improper input", "malformed", "sanitiz",
                          "injection", "sql injection", "command injection", "path traversal",
                          "directory traversal", "format string", "untrusted input")),
    # DoS：放在机制类之后
    ("dos", ("denial of service", "denialofservice", "infinite loop", "endless loop",
             "infinite recursion", "hang", "hangs", "crash", "crashes", "dos")),
)


def derive_error_type(cwe_id: str, classification: str, summary: str) -> str:
    """把一条知识归到 7 类之一。

    判定顺序：**CWE（权威）→ 关键词（按 KEYWORD_RULES 的优先级）→ `general`**。
    输入文本先经 `normalize_for_match` 归一，因此连字符/下划线/大小写差异不会再造成漏匹配。
    """
    cwe = normalize_text(cwe_id).upper()
    if cwe in CWE_TO_TYPE:
        return CWE_TO_TYPE[cwe]

    haystack = normalize_for_match("%s %s" % (classification or "", summary or ""))
    if not haystack:
        return "general"
    for family, patterns in KEYWORD_RULES:
        if _has(haystack, *patterns):
            return family
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
