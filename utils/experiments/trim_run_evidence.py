# -*- coding: utf-8 -*-
"""给 run 的证据文件"瘦身"：**原文无损留存（.gz）+ 精简版就地替换**。

## 为什么要它
真机上 held 样本的 `second_pass/consolidated/*_r2.json` 单个能到 **227MB** ——
原因是**每条候选都内嵌了整份被分析文件源码**（`_current_code`），而 held 样本会捞到
大量外来候选。一个样本就能吃掉 0.4~1.3G，几十个样本直接把磁盘打满
（Weaviate 到 90% 会转只读，正在跑的批次当场失败）。

## 做法（三条硬约束）
1. **零丢失**：原文先 gzip 存成 `同名 + .gz`，校验解压后 sha256 与原文件一致，才允许替换；
2. **工具零改动**：精简版仍写成**同名普通 json**，保留工具真正读的字段；
   只删/截断"体积大但不参与判定"的东西：
   * `_current_code`（每条候选内嵌的整份源码）→ 直接删（判定不用它；要复现时从 `.gz` 取）
   * 任何超过 `--max-str` 的字符串 → 截断并标注（避免其它未预料的巨型字段）
   * `original_issues` / `corrected_issues` 里超长的代码块同样按上面的规则处理
3. **可验证**：瘦身前后，用生产口径读一遍"判定相关字段"（候选数、每条的 channel/
   sqlite_id/决策/拒因/matched_fields/分数、new_findings 数），**必须完全一致**。

用法：
    python -X utf8 utils/experiments/trim_run_evidence.py --runs reports/arm1_runs.txt --apply
    python -X utf8 utils/experiments/trim_run_evidence.py --run-dir reports/analysis/CVE-X/<run> --apply
不加 `--apply` 只报告能省多少。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DROP_KEYS = {"_current_code"}          # 体积元凶：每条候选内嵌的整份源码
TRUNCATE_KEYS: set[str] = set()        # 想单独控制的键（现在统一按长度处理）
PATTERNS = ("second_pass/consolidated/*_r2.json", "fullLayer/consolidated/*_r1.json",
            "pureLLM/consolidated/*.json")

PROBE_FIELDS = ("channel", "sqlite_id", "gating_decision", "rejection_reason",
                "matched_fields", "structured_score", "semantic_score", "context_score",
                "anchor_score", "total_score", "vector_layer", "error_type", "severity")


def trim(obj, max_str: int, stats: dict):
    """递归删/截断巨型字段，其余原样保留。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in DROP_KEYS:
                stats["dropped_" + k] = stats.get("dropped_" + k, 0) + 1
                continue
            out[k] = trim(v, max_str, stats)
        return out
    if isinstance(obj, list):
        return [trim(v, max_str, stats) for v in obj]
    if isinstance(obj, str) and len(obj) > max_str:
        stats["truncated"] = stats.get("truncated", 0) + 1
        return obj[:max_str] + "\n…[trim_run_evidence 已截断，原文见同名 .gz]"
    return obj


def probe(payload: dict) -> dict:
    """抽出"工具判定真正依赖的字段"，用于瘦身前后一致性比对。"""
    out = {"new_findings": len(payload.get("new_findings") or []),
           "file": payload.get("file"), "run_id": payload.get("run_id"),
           "evidence": {}}
    for key in ("retrieval_evidence", "gap_retrieval_evidence"):
        rows = []
        for ev in (payload.get(key) or []):
            rows.append({
                "issue_file": ev.get("issue_file"),
                "query_semantic_used": ev.get("query_semantic_used"),
                "n_candidates": len(ev.get("candidates") or []),
                "candidates": [
                    {f: c.get(f) for f in PROBE_FIELDS if f in c}
                    for c in (ev.get("candidates") or []) if isinstance(c, dict)
                ],
            })
        out["evidence"][key] = rows
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process(path: Path, max_str: int, apply: bool) -> tuple[int, int, bool]:
    """返回 (原大小, 新大小, 一致性是否通过)。"""
    before = path.stat().st_size
    payload = json.loads(path.read_text(encoding="utf-8"))
    ref = probe(payload)
    stats: dict = {}
    small = trim(payload, max_str, stats)
    if not apply:
        text = json.dumps(small, ensure_ascii=False)
        return before, len(text.encode("utf-8")), True

    # 1) 原文无损留档
    gz = path.with_suffix(path.suffix + ".gz")
    if not gz.exists():
        raw = path.read_bytes()
        # compresslevel=1：证据里大量是重复的源码文本，低级别就能压得很好，
        # 而 227MB 的文件用默认级别会明显拖慢（低优先级跑也还是占 IO）。
        with gzip.open(gz, "wb", compresslevel=1) as fh:
            fh.write(raw)
        if gzip.open(gz, "rb").read() != raw:
            raise SystemExit("*** %s 压缩后与原文件不一致，abort" % path)
    # 2) 写精简版
    path.write_text(json.dumps(small, ensure_ascii=False), encoding="utf-8")
    # 3) 一致性比对（生产口径读到的字段必须一模一样）
    again = json.loads(path.read_text(encoding="utf-8"))
    ok = probe(again) == ref
    return before, path.stat().st_size, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=None, help="每行 CVE/run_id 的清单")
    ap.add_argument("--run-dir", type=Path, action="append", default=[])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-str", type=int, default=20000,
                    help="单条字符串超过这个长度就截断（默认 20000 字符）")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.run_dir]
    if args.runs:
        for line in args.runs.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                dirs.append(ROOT / "reports/analysis" / line)
    if not dirs:
        raise SystemExit("需要 --runs 或 --run-dir")

    tot_before = tot_after = 0
    all_ok = True
    for d in dirs:
        if not d.is_dir():
            print("  跳过（不存在）: %s" % d)
            continue
        for pattern in PATTERNS:
            for f in sorted(d.glob(pattern)):
                if f.suffix == ".gz":
                    continue
                if f.stat().st_size < 1 << 20:      # 小于 1MB 的不折腾
                    continue
                b, a, ok = process(f, args.max_str, args.apply)
                tot_before += b
                tot_after += a
                all_ok = all_ok and ok
                print("  %-72s %8.1fMB -> %7.1fMB  一致=%s"
                      % (str(f.relative_to(d))[:72], b / 1e6, a / 1e6, ok))
    print("\n合计 %.1f MB -> %.1f MB（省 %.1f MB）%s"
          % (tot_before / 1e6, tot_after / 1e6, (tot_before - tot_after) / 1e6,
             "" if args.apply else "（未加 --apply，只是预演）"))
    print("判定字段一致性: %s" % ("全部通过" if all_ok else "**有不一致，别信这份精简版**"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
