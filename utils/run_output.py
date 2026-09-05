"""Helpers for custom analysis output directory names."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def normalize_output_dir(output_dir: Optional[str]) -> Optional[str]:
    """Normalize --output-dir into a single parent folder under reports/analysis/.

    Final layout: reports/analysis/<output_dir>/<run_id>/
    Accepts names like CVE-2011-1078. If a path is provided, only the final
    segment is used.
    """
    if output_dir is None:
        return None
    raw = str(output_dir).strip()
    if not raw:
        return None
    name = Path(raw.replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError(f"无效的 output-dir: {output_dir!r}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError(f"output-dir 仅允许字母/数字/._-，收到: {name!r}")
    return name


# Backward-compatible alias used by earlier wiring/tests.
normalize_output_run_id = normalize_output_dir


def format_run_report_relpath(run_id: str, output_dir: Optional[str] = None) -> str:
    """Return relative path under reports/analysis for display."""
    if output_dir:
        return f"{output_dir}/{run_id}"
    return str(run_id)
