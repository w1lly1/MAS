from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple


_PROMPT_BUDGET_DEBUG = False


def _flatten_input_ids(input_ids: Any) -> List[int]:
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return list(input_ids[0])
        return list(input_ids)
    if hasattr(input_ids, "tolist"):
        try:
            flattened = input_ids.tolist()
            if isinstance(flattened, list):
                if flattened and isinstance(flattened[0], list):
                    return list(flattened[0])
                return list(flattened)
        except Exception:
            pass
    return []


def _extract_input_ids(encoded: Any) -> Any:
    if encoded is None:
        return None

    if isinstance(encoded, dict):
        return encoded.get("input_ids")

    try:
        getter = getattr(encoded, "get", None)
        if callable(getter):
            input_ids = getter("input_ids")
            if input_ids is not None:
                return input_ids
    except Exception:
        pass

    try:
        return encoded["input_ids"]
    except Exception:
        pass

    return getattr(encoded, "input_ids", None)


def exact_token_count(tokenizer: Any, text: str, add_special_tokens: bool = True) -> int:
    if not tokenizer or not text:
        return 0
    try:
        encoded = tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)
        input_ids = _extract_input_ids(encoded)
        if input_ids is None:
            return estimate_token_count(tokenizer, text)
        return len(_flatten_input_ids(input_ids))
    except Exception:
        return estimate_token_count(tokenizer, text)


def truncate_text_to_token_budget(tokenizer: Any, text: str, max_tokens: int, add_special_tokens: bool = True) -> str:
    normalized = _normalize_text(text)
    if not normalized or max_tokens <= 0:
        return ""

    if exact_token_count(tokenizer, normalized, add_special_tokens=add_special_tokens) <= max_tokens:
        return normalized

    try:
        encoded = tokenizer(normalized, add_special_tokens=False, truncation=False)
        input_ids = _extract_input_ids(encoded)
        ids = _flatten_input_ids(input_ids)
        if not ids:
            return normalized[: max_tokens * 4]

        if _PROMPT_BUDGET_DEBUG:
            print(f"[PROMPT BUDGET DEBUG] encoded_type={type(encoded).__name__} input_ids_type={type(input_ids).__name__} raw_ids_len={len(ids)} max_tokens={max_tokens}")

        low = 0
        high = len(ids)
        best = ""

        while low <= high:
            mid = (low + high) // 2
            candidate = tokenizer.decode(ids[:mid], skip_special_tokens=True)
            candidate_len = exact_token_count(tokenizer, candidate, add_special_tokens=add_special_tokens)
            if candidate_len <= max_tokens:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1

        if best:
            return best
    except Exception:
        pass

    return normalized[: max_tokens * 4]


def resolve_model_max_tokens(tokenizer: Any, fallback: int = 1024) -> int:
    if tokenizer is None:
        return fallback

    for attr in ("model_max_length", "max_position_embeddings", "n_positions", "seq_length"):
        try:
            value = int(getattr(tokenizer, attr, 0) or 0)
        except Exception:
            value = 0
        if 64 <= value <= 200000:
            return value
    return fallback


def estimate_token_count(tokenizer: Any, text: str) -> int:
    if not tokenizer or not text:
        return 0
    try:
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return len(input_ids[0])
            return len(input_ids)
    except Exception:
        pass
    return max(0, len(text) // 4)


def _normalize_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").strip()


def _looks_like_code(text: str) -> bool:
    if not text:
        return False
    code_markers = (
        "def ",
        "class ",
        "async def ",
        "return ",
        "import ",
        "from ",
        "if ",
        "for ",
        "while ",
        "try:",
        "except ",
        "public ",
        "private ",
        "protected ",
        "function ",
        "const ",
        "let ",
        "var ",
    )
    lines = text.splitlines()
    marker_hits = sum(
        1 for line in lines if line.lstrip().startswith(code_markers) or line.rstrip().endswith((":", "{", "}", ";"))
    )
    return marker_hits >= 2 or (len(lines) >= 4 and marker_hits >= 1)


def _split_sentences(text: str) -> List[str]:
    pieces = re.split(r"(?<=[。！？!?\.])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def _split_code_blocks(text: str) -> List[str]:
    lines = text.splitlines()
    if not lines:
        return []

    blocks: List[str] = []
    current: List[str] = []
    boundary_patterns = (
        re.compile(r"^\s*(async\s+def|def|class)\s+"),
        re.compile(r"^\s*@"),
    )

    def flush() -> None:
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue

        if current and any(pattern.match(line) for pattern in boundary_patterns):
            flush()

        current.append(line)

    flush()
    return [block for block in blocks if block]


def semantic_split_text(text: str) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    if _looks_like_code(normalized):
        blocks = _split_code_blocks(normalized)
        return blocks if blocks else [normalized]

    if "\n" not in normalized:
        return _split_sentences(normalized) or [normalized]

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", normalized) if paragraph.strip()]
    units: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= 200 or not _split_sentences(paragraph):
            units.append(paragraph)
        else:
            units.extend(_split_sentences(paragraph))
    return units if units else [normalized]


def _truncate_with_tokenizer(tokenizer: Any, text: str, max_tokens: int) -> str:
    if not tokenizer or not text or max_tokens <= 0:
        return ""
    try:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
            return_tensors=None,
        )
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        if isinstance(input_ids, list) and input_ids:
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return tokenizer.decode(input_ids[:max_tokens], skip_special_tokens=True)
    except Exception:
        pass
    return text[: max_tokens * 4]


def semantic_truncate_text(tokenizer: Any, text: str, max_tokens: int) -> str:
    normalized = _normalize_text(text)
    if not normalized or max_tokens <= 0:
        return ""

    if estimate_token_count(tokenizer, normalized) <= max_tokens:
        return normalized

    units = semantic_split_text(normalized)
    if not units:
        return _truncate_with_tokenizer(tokenizer, normalized, max_tokens)

    kept: List[str] = []
    used_tokens = 0

    for unit in units:
        unit_tokens = estimate_token_count(tokenizer, unit)
        if unit_tokens <= 0:
            continue

        if used_tokens + unit_tokens <= max_tokens:
            kept.append(unit)
            used_tokens += unit_tokens
            continue

        if not kept:
            return _truncate_with_tokenizer(tokenizer, unit, max_tokens)
        break

    result = "\n\n".join(kept).strip()
    if result:
        return result
    return _truncate_with_tokenizer(tokenizer, normalized, max_tokens)


def prepare_generation_prompt(
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    fallback_model_max: int = 1024,
    safety_margin: int = 32,
) -> Tuple[str, int, int]:
    model_max = resolve_model_max_tokens(tokenizer, fallback=fallback_model_max)
    requested_new = int(max_new_tokens or 0)
    max_generation_budget = max(0, model_max - safety_margin)
    requested_new = min(max(16, requested_new), max_generation_budget)
    input_budget = max(0, model_max - requested_new - safety_margin)
    safe_prompt = semantic_truncate_text(tokenizer, prompt, input_budget)
    safe_prompt = truncate_text_to_token_budget(tokenizer, safe_prompt, input_budget, add_special_tokens=True)
    return safe_prompt, input_budget, requested_new


def budget_text_segments(
    tokenizer: Any,
    segments: Sequence[str],
    max_tokens: int,
) -> List[str]:
    kept: List[str] = []
    used_tokens = 0
    for segment in segments:
        if not segment:
            continue
        remaining = max_tokens - used_tokens
        if remaining <= 0:
            break
        trimmed = semantic_truncate_text(tokenizer, segment, remaining)
        if not trimmed:
            continue
        trimmed_tokens = estimate_token_count(tokenizer, trimmed)
        if trimmed_tokens <= 0:
            continue
        kept.append(trimmed)
        used_tokens += trimmed_tokens
    return kept