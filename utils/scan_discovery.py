"""Shared helpers for bounded source-file discovery and timeout estimation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_IGNORED_DIRECTORIES: Tuple[str, ...] = (
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    "venv",
    "env",
    ".venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".tox",
    ".pytest_cache",
)


DEFAULT_SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".py",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".cxx",
    ".js",
    ".ts",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
)


ANALYSIS_AGENT_DEFAULTS: Dict[str, Dict[str, float]] = {
    "ai_code_quality": {"base_seconds": 20.0, "per_file_seconds": 1.5, "per_mb_seconds": 3.5},
    "ai_security": {"base_seconds": 18.0, "per_file_seconds": 1.4, "per_mb_seconds": 3.0},
    "ai_performance": {"base_seconds": 18.0, "per_file_seconds": 1.2, "per_mb_seconds": 2.5},
    "static_scan": {"base_seconds": 14.0, "per_file_seconds": 0.9, "per_mb_seconds": 1.8},
}


def _normalize_extensions(extensions: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not extensions:
        return DEFAULT_SUPPORTED_EXTENSIONS
    normalized = []
    for ext in extensions:
        if not ext:
            continue
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return tuple(dict.fromkeys(normalized)) or DEFAULT_SUPPORTED_EXTENSIONS


def _normalize_ignored_directories(ignored_directories: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not ignored_directories:
        return DEFAULT_IGNORED_DIRECTORIES
    normalized = []
    for directory in ignored_directories:
        if not directory:
            continue
        directory = directory.strip()
        if directory:
            normalized.append(directory)
    return tuple(dict.fromkeys(normalized)) or DEFAULT_IGNORED_DIRECTORIES


def discover_source_files(
    target_directory: str,
    *,
    supported_extensions: Optional[Sequence[str]] = None,
    ignored_directories: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """Discover source files with bounded traversal.

    Returns a dict with:
      - files: list[dict(path, relative_path, size)]
      - scanned_files: number of matching files selected
      - visited_directories: number of directories entered
      - skipped_directories: ignored directories pruned during traversal
      - partial_scan: whether traversal stopped early because of a budget
    """

    root_path = Path(target_directory)
    if not root_path.exists():
        return {
            "files": [],
            "scanned_files": 0,
            "visited_directories": 0,
            "skipped_directories": 0,
            "partial_scan": False,
            "max_files": max_files,
            "max_depth": max_depth,
            "supported_extensions": list(_normalize_extensions(supported_extensions)),
            "ignored_directories": list(_normalize_ignored_directories(ignored_directories)),
        }

    extensions = _normalize_extensions(supported_extensions)
    ignored = set(_normalize_ignored_directories(ignored_directories))
    discovered: List[Dict[str, Any]] = []
    visited_directories = 0
    skipped_directories = 0
    partial_scan = False

    for root, dirs, files in os.walk(root_path):
        visited_directories += 1
        current_root = Path(root)
        try:
            relative_depth = len(current_root.relative_to(root_path).parts)
        except Exception:
            relative_depth = 0

        if max_depth is not None and relative_depth >= max_depth:
            dirs[:] = []

        pruned_dirs = []
        for directory in dirs:
            if directory in ignored:
                skipped_directories += 1
                continue
            pruned_dirs.append(directory)
        dirs[:] = pruned_dirs

        for file_name in files:
            if max_files is not None and len(discovered) >= max_files:
                partial_scan = True
                return {
                    "files": discovered,
                    "scanned_files": len(discovered),
                    "visited_directories": visited_directories,
                    "skipped_directories": skipped_directories,
                    "partial_scan": partial_scan,
                    "max_files": max_files,
                    "max_depth": max_depth,
                    "supported_extensions": list(extensions),
                    "ignored_directories": list(ignored),
                }

            file_ext = Path(file_name).suffix.lower()
            if file_ext not in extensions:
                continue

            file_path = current_root / file_name
            try:
                size = file_path.stat().st_size
            except Exception:
                size = 0

            try:
                relative_path = str(file_path.relative_to(root_path))
            except Exception:
                relative_path = str(file_path)

            discovered.append(
                {
                    "path": str(file_path),
                    "relative_path": relative_path,
                    "size": int(size),
                }
            )

    return {
        "files": discovered,
        "scanned_files": len(discovered),
        "visited_directories": visited_directories,
        "skipped_directories": skipped_directories,
        "partial_scan": partial_scan,
        "max_files": max_files,
        "max_depth": max_depth,
        "supported_extensions": list(extensions),
        "ignored_directories": list(ignored),
    }


def estimate_analysis_timeout(
    file_infos: Sequence[Dict[str, Any]],
    *,
    enabled_agents: Sequence[str],
    timeout_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Estimate an analysis timeout using file size and file count.

    The estimator is intentionally conservative: it sums an agent-specific cost
    model, then applies a safety factor and clamps the final value between
    configurable minimum and maximum bounds.
    """

    timeout_config = timeout_config or {}
    min_timeout = int(timeout_config.get("min_timeout_seconds", 600))
    max_timeout = int(timeout_config.get("max_timeout_seconds", 3600))
    safety_factor = float(timeout_config.get("safety_factor", 1.25))
    device_multiplier = float(timeout_config.get("device_multiplier", 1.0))
    size_divisor = float(timeout_config.get("size_divisor", 1024 * 1024))
    agent_weights = timeout_config.get("agent_weights", {}) or {}

    total_size_bytes = 0
    for item in file_infos:
        try:
            total_size_bytes += int(item.get("size", 0) or 0)
        except Exception:
            continue

    file_count = len(file_infos)
    size_mb = total_size_bytes / size_divisor if size_divisor else 0.0

    breakdown: Dict[str, Dict[str, float]] = {}
    raw_timeout = 0.0

    for agent_name in enabled_agents:
        agent_profile = dict(ANALYSIS_AGENT_DEFAULTS.get(agent_name, {}))
        agent_profile.update(agent_weights.get(agent_name, {}))
        base_seconds = float(agent_profile.get("base_seconds", 10.0))
        per_file_seconds = float(agent_profile.get("per_file_seconds", 1.0))
        per_mb_seconds = float(agent_profile.get("per_mb_seconds", 1.0))
        agent_cost = base_seconds + (per_file_seconds * file_count) + (per_mb_seconds * size_mb)
        breakdown[agent_name] = {
            "base_seconds": base_seconds,
            "per_file_seconds": per_file_seconds,
            "per_mb_seconds": per_mb_seconds,
            "estimated_seconds": round(agent_cost, 2),
        }
        raw_timeout += agent_cost

    adjusted_timeout = raw_timeout * device_multiplier
    estimated = int(math.ceil(adjusted_timeout * safety_factor)) if adjusted_timeout > 0 else min_timeout
    estimated = max(min_timeout, min(max_timeout, estimated))

    return {
        "estimated_timeout_seconds": estimated,
        "raw_timeout_seconds": round(raw_timeout, 2),
        "device_multiplier": device_multiplier,
        "safety_factor": safety_factor,
        "min_timeout_seconds": min_timeout,
        "max_timeout_seconds": max_timeout,
        "total_file_count": file_count,
        "total_size_bytes": total_size_bytes,
        "total_size_mb": round(size_mb, 2),
        "agent_breakdown": breakdown,
    }