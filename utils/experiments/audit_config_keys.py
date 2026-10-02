# -*- coding: utf-8 -*-
"""配置死键审计（《01》小事 13 的工具化版本）。

做两件事：
  ① 扫描 `infrastructure/config/ai_agent_config.json` 的每个叶子键，在代码里搜它的名字，
     报出"没有任何代码引用"的键（默认输出，`--all` 看全部键的引用位置）。
  ② `--deleted` 模式：核对一批**已被删除**的键确实没有读取路径
     （`.get("键")` / `["键"]`），并顺带列出"动态拼键"的可疑写法 —— 防止
     `config.get(f"{name}_timeout")` 这种写法把"字面量搜不到"误判成"没人读"。

用法：
    python -X utf8 utils/experiments/audit_config_keys.py            # 只列疑似死键
    python -X utf8 utils/experiments/audit_config_keys.py --all      # 每个键都列
    python -X utf8 utils/experiments/audit_config_keys.py --deleted  # 核对删除清单

判据（写死在脚本里，避免事后自我说服）：
  · "被引用" = 该键名作为**字符串字面量**出现在某个 .py 里，且不在注释行、不在 `_comment*` 里；
  · `_` 开头的键（`_comment`、`_gate_comment` 等）一律**不算死键**：它们是给人看的说明，
    其中多条是门控/克隆参数的理由记录，删了会丢掉论证链；
  · 结论只覆盖本仓库的 .py/.json（不含 venv、reports、论文/专利/软著等目录）。
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG = os.path.join(REPO, "infrastructure", "config", "ai_agent_config.json")

SKIP_DIRS = {
    "venv", ".git", "__pycache__", ".pytest_cache", ".tmp_test", ".pytest_tmp",
    ".pytest_bt2", "reports", "reports_backup_westb", "model_cache", "local_libs",
    "论文", "专利", "软著", "docs", "refers", "demo", "node_modules", "BigVul",
}

# 小事 13 里已删除的键（--deleted 模式核对它们没有读取路径）
DELETED_KEYS = [
    "enable_readability_enhancement_ai",
    "batch_size",
    "max_analysis_length",
    "vulnerability_confidence_threshold",
    "max_threat_analysis_length",
    "complexity_analysis_depth",
    "max_optimization_length",
    "top_p",
    "conversation_history_length",
    "response_max_length",
    "fallback_models",
    "cache_ttl_seconds",
    "max_cache_size",
    "enable_pylint", "enable_bandit", "enable_flake8", "enable_mypy", "enable_safety",
    "primary_chat_model", "fallback_chat_models", "model_selection_strategy",
    "transformers_version", "deployment_notes",
]

DYNAMIC_PATTERNS = [
    re.compile(r"""f["'][^"']*_(?:models|length|timeout|threshold)["']"""),
    re.compile(r"""["'][^"']*_(?:models|length|timeout|threshold)["']\s*\+"""),
    re.compile(r"""\+=?\s*["'][^"']*_(?:models|length|timeout|threshold)["']"""),
]


def _read_sources():
    out = []
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if not (name.endswith(".py") or name.endswith(".json")):
                continue
            path = os.path.join(root, name)
            if os.path.normcase(path) == os.path.normcase(CONFIG):
                continue
            out.append((os.path.relpath(path, REPO), io.open(path, encoding="utf-8", errors="replace").read()))
    return out


SOURCES = _read_sources()


def references(key):
    """返回 (真实引用 [(文件, 行数)], 仅注释行数)。"""
    lit = re.compile(r"""["']%s["']""" % re.escape(key))
    real, comment_only = [], 0
    for rel, text in SOURCES:
        lines = [ln for ln in text.splitlines() if lit.search(ln)]
        if not lines:
            continue
        non_comment = [
            ln for ln in lines
            if not ln.strip().startswith("#") and "_comment" not in ln
        ]
        if non_comment:
            real.append((rel, len(non_comment)))
        else:
            comment_only += len(lines)
    return real, comment_only


def flatten(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    else:
        yield prefix, node


def scan(show_all: bool) -> int:
    config = json.loads(io.open(CONFIG, encoding="utf-8").read())
    leaves = list(flatten(config))
    print("=" * 100)
    print(f"配置: {os.path.relpath(CONFIG, REPO)}   叶子键: {len(leaves)}")
    print("=" * 100)
    dead_ops, dead_docs = [], []
    for path, value in sorted(leaves):
        key = path.split(".")[-1]
        real, comment_only = references(key)
        total = sum(n for _, n in real)
        is_doc = path.split(".")[-1].startswith("_")
        if total == 0:
            (dead_docs if is_doc else dead_ops).append((path, value, comment_only))
            flag = "❌ 无真引用"
        else:
            flag = f"✅ {total} 处"
        if show_all or total == 0:
            where = ", ".join(f"{f}({n})" for f, n in real[:4]) or f"仅注释 {comment_only} 处"
            print(f"{flag:<12} {path:<58} = {json.dumps(value, ensure_ascii=False)[:26]:<28} {where}")
    print("-" * 100)
    print(f"【操作性死键】{len(dead_ops)} 个（这些才是「没人读的配置键」）:")
    for path, value, _ in dead_ops:
        print(f"   · {path} = {json.dumps(value, ensure_ascii=False)}")
    print(f"【说明性 `_` 键】{len(dead_docs)} 个（不算死键：是给人看的注释/理由记录）")
    print("=" * 100)
    return 0


def check_deleted() -> int:
    print("=" * 100)
    print("核对「已删除的键」是否有读取路径（.get(\"键\") / [\"键\"]）")
    print("=" * 100)
    bad = []
    for key in DELETED_KEYS:
        read_pat = re.compile(r"""(?:\.get\(\s*|\[\s*)["']%s["']""" % re.escape(key))
        hits = [(rel, ln.strip()) for rel, t in SOURCES for ln in t.splitlines() if read_pat.search(ln)]
        loose = sum(1 for _, t in SOURCES for ln in t.splitlines()
                    if re.search(r"""["']%s["']""" % re.escape(key), ln))
        print(f"{'✅' if not hits else '❌'} {key:<36} 读取形态={len(hits):<3} 同名字符串={loose}")
        if hits:
            bad.append((key, hits))
    print("-" * 100)
    suspicious = [
        (rel, ln.strip()) for rel, t in SOURCES for ln in t.splitlines()
        if any(p.search(ln) for p in DYNAMIC_PATTERNS)
        and os.path.basename(rel) != os.path.basename(__file__)
    ]
    print("动态拼键的可疑写法（人工过一遍，确认没有 config.get(f\"{name}_xxx\")）:")
    for rel, ln in suspicious:
        print(f"   {rel}: {ln[:110]}")
    if not suspicious:
        print("   （无）")
    print("-" * 100)
    print("结论：" + ("所有被删的键都没有读取形态 ✅" if not bad else f"⚠️ 需复核 {len(bad)} 个"))
    print("=" * 100)
    return 0


if __name__ == "__main__":
    if "--deleted" in sys.argv:
        sys.exit(check_deleted())
    sys.exit(scan(show_all="--all" in sys.argv))
