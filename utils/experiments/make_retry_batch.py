# -*- coding: utf-8 -*-
"""从一次批次的日志里挑出被判定为 `partial`（未确认完成）的样本，生成"只补跑这些"的批次配置。

## 为什么需要它

R1 修复后，超时/没跑完的条目会被如实标成 `partial`（不再假装成功）。但那些样本**数据不完整**，
不能直接进对比。整臂重跑要 80 分钟；而**只补跑失败的几个**通常只要十几分钟。

**口径说明（必须写进结论）**：补跑用的是**新的等待上限**，所以报告里要说明
"完成样本 = 首次通过 + 用更长上限补跑通过"，并给出补跑前的 partial 数 —— 不能只报一个"全部完成"。

## 用法

    python utils/experiments/make_retry_batch.py --log arm1_new_new.log \
        --config utils/experiments/smoke_kb30.json --out utils/experiments/smoke_kb30_retry.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ITEM_RE = re.compile(r"^\s*\[(\d+)/(\d+)\]\s+\S+\s+(.+?)\s*$")
PARTIAL_MARK = "未确认完成"


def parse_partial_targets(log_path: Path) -> list:
    """按顺序读日志：记住最近一条 `[n/N] 📂 <dir>`，遇到 `未确认完成` 就把那个 dir 记下来。"""
    current = None
    out = []
    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ITEM_RE.match(line)
            if m:
                current = m.group(3).strip()
                continue
            if PARTIAL_MARK in line and current:
                if current not in out:
                    out.append(current)
                current = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    targets = parse_partial_targets(args.log)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    items = cfg.get("items", [])
    by_dir = {str(it.get("target_dir")): it for it in items}

    picked, missing = [], []
    for t in targets:
        it = by_dir.get(t)
        if it is None:
            # 日志里的路径可能是绝对化过的，做一次后缀匹配兜底
            cand = [v for k, v in by_dir.items() if k.endswith(t.split("/")[-1])]
            if cand:
                it = cand[0]
            else:
                missing.append(t)
                continue
        picked.append(it)

    cfg["items"] = picked
    cfg["description"] = ("补跑批次：仅包含上一次被判定为 partial 的 %d 个样本"
                          "（等待上限已提高，见 analysis_timeout_min_seconds）" % len(picked))
    cfg["why_kb"] = cfg.get("why_kb", "")
    cfg["retry_of"] = str(args.config)
    cfg["retry_log"] = str(args.log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("日志: %s" % args.log.name)
    print("判定为 partial 的样本: %d 个" % len(targets))
    for t in targets:
        print("   - %s" % t.split("/")[-1])
    print("已写入补跑批次: %s（items=%d）" % (args.out, len(picked)))
    if missing:
        print("⚠️ 有 %d 个在配置里找不到，未纳入：%s" % (len(missing), missing))


if __name__ == "__main__":
    main()
