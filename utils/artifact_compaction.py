# -*- coding: utf-8 -*-
"""写盘前给 run 级报告"瘦身"（生产代码，不再是运营脚本）。

## 为什么需要它
实测：库外样本的 `second_pass/consolidated/*_r2.json` **单个可到 227MB** ——
原因是 **每条候选都内嵌了一份完整的被分析文件源码**（`_current_code`），
而一次分析会产生几百到几千条候选。一个样本就能吃掉 0.4~1.3G：
几十个样本把磁盘打满 → Weaviate 到 90% 转只读、正在跑的批次当场失败。

## 关键前提：**代码本来就在磁盘上**
`_current_code` 就是"被分析文件的全文"，那份文件**本来就在数据集目录里**（证据里还留着
`_analysis_file` / `issue_file` 指向它）。所以写盘时不重复存它**不丢任何信息**：
需要复现时按路径读原文件即可（真实操作过的例子：`predict_t2_gain.py` 就是这么回退的）。

## 边界（三条）
1. **只影响写盘**：内存里的候选字典**原样保留** —— 门控判据 `_candidate_code_fixed()`
   要用 `_current_code`，动它会改行为（这是"可观测性改动"，不是"判定改动"）；
2. **工具读得到的字段一个不能少**（通道/条目/决策/拒因/`matched_fields`/各项分数/
   `new_findings`/证据级字段…），超长字符串只**截断**不删除，并留下标记；
3. **可关**：配置 `artifact_settings.compact_on_write=false` 或环境变量
   `MAS_ARTIFACT_COMPACTION=off` 即恢复历史行为（排查"是不是瘦身导致的"时用）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: 每条候选里重复内嵌的整份源码 —— 体积元凶，写盘时丢弃
DROP_KEYS = {"_current_code"}

#: 单条字符串超过这个长度就截断（防止其它未预料的巨型字段）
DEFAULT_MAX_STR = 20000

_TRUNCATION_MARK = "\n…[artifact_compaction 已截断：完整内容见被分析文件本身]"

_config_cache: Optional[Dict[str, Any]] = None


def _load_settings() -> Dict[str, Any]:
    """读 `artifact_settings`（失败就用默认值，绝不因为配置读不到而让写盘失败）。"""
    global _config_cache
    if _config_cache is None:
        cfg: Dict[str, Any] = {}
        try:
            root = Path(__file__).resolve().parents[1]
            path = root / "infrastructure/config/ai_agent_config.json"
            if path.is_file():
                cfg = json.loads(path.read_text(encoding="utf-8")).get("artifact_settings") or {}
        except Exception:  # noqa: BLE001
            cfg = {}
        _config_cache = cfg
    return _config_cache


def compaction_enabled() -> bool:
    """是否启用写盘瘦身（环境变量优先，便于临时复现历史行为）。"""
    env = str(os.environ.get("MAS_ARTIFACT_COMPACTION", "")).strip().lower()
    if env in {"off", "0", "false", "no"}:
        return False
    if env in {"on", "1", "true", "yes"}:
        return True
    return bool(_load_settings().get("compact_on_write", True))


def max_str_chars() -> int:
    try:
        return int(_load_settings().get("max_str_chars", DEFAULT_MAX_STR))
    except Exception:  # noqa: BLE001
        return DEFAULT_MAX_STR


def compact_payload(payload: Any, max_str: Optional[int] = None) -> Tuple[Any, Dict[str, int]]:
    """返回 (瘦身后的副本, 统计)。**不改动入参**（深拷贝语义靠重建容器实现）。"""
    limit = max_str_chars() if max_str is None else int(max_str)
    stats: Dict[str, int] = {}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k in DROP_KEYS:
                    stats[k] = stats.get(k, 0) + 1
                    continue
                out[k] = walk(v)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str) and limit > 0 and len(node) > limit:
            stats["truncated"] = stats.get("truncated", 0) + 1
            return node[:limit] + _TRUNCATION_MARK
        return node

    return walk(payload), stats
