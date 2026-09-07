# -*- coding: utf-8 -*-
"""TODO5: 构建「缺失型」缺陷测试集（评审意见 #5：测试集仅覆盖错误型，缺失型未验证）。

口径（复用 prepare_400_error_only_test.py 的 classify_cve）：
  - 缺失型（missing）：任一代码文件 diff 存在新增行（+ 行），且**无删除行**（- 行）。
    —— 即缺陷是「漏写检查」，修复只增不改，没有可提取的"错误代码片段"。
  - 完整性过滤同 prepare 脚本（before/after 有代码文件 + cve_metadata 有 cve_id/summary）。

产出：
  1. reports/negative_exp_manifest_400_missing.json
  2. utils/experiments/test_400_missing_batch.json

运行（MAS 根目录，需 GPU/BigVul 数据）：
    python utils/experiments/prepare_missing_type_test.py --total 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.bigvul_ingest.build_structured_ingest import (  # noqa: E402
    _find_commit_meta_paths,
    _load_json,
)
from utils.experiments.prepare_400_error_only_test import (  # noqa: E402
    BEFORE, METADATA, _has_code_files, _safe_read, diff_lines,
)

CODE_EXTS = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".s", ".S"}


def classify_missing(cve_dir: Path):
    """缺失型判定：有新增行、无删除行。返回 (is_missing, has_added, has_removed)。"""
    has_added = False
    has_removed = False
    for commit_meta_path in _find_commit_meta_paths(cve_dir):
        try:
            commit_meta = _load_json(commit_meta_path)
        except Exception:
            continue
        commit_dir = commit_meta_path.parent.name
        files = commit_meta.get("files", []) if isinstance(commit_meta.get("files"), list) else []
        for file_info in files:
            local_name = file_info.get("local_name", "")
            if Path(local_name).suffix.lower() not in CODE_EXTS:
                continue
            before_path = BEFORE / cve_dir.name / commit_dir / local_name
            after_path = BEFORE.parent / "after" / cve_dir.name / commit_dir / local_name
            removed, added = diff_lines(_safe_read(before_path), _safe_read(after_path))
            if removed:
                has_removed = True
            if added:
                has_added = True
    return (has_added and not has_removed), has_added, has_removed


def is_complete(cve_dir: Path) -> bool:
    try:
        meta = json.loads((cve_dir / "cve_metadata.json").read_text(encoding="utf-8"))
        cve_id = str(meta.get("cve_id", "")).strip()
        summary = str(meta.get("summary", "")).strip()
    except Exception:
        return False
    if not cve_id or not summary:
        return False
    if not _has_code_files(BEFORE / cve_id):
        return False
    if not _has_code_files(BEFORE.parent / "after" / cve_id):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    candidates = []
    for d in sorted(METADATA.glob("CVE-*")):
        if not (d / "cve_metadata.json").exists():
            continue
        is_missing, has_added, has_removed = classify_missing(d)
        if not is_missing:
            continue
        if not is_complete(d):
            continue
        try:
            meta = json.loads((d / "cve_metadata.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates.append({
            "cve": str(meta.get("cve_id", "")).strip(),
            "project": meta.get("project", "unknown"),
        })

    print(f"缺失型候选池: {len(candidates)}")
    if len(candidates) < args.total:
        raise SystemExit(f"❌ 缺失型候选不足 {args.total}，只有 {len(candidates)}")

    rng = random.Random(args.seed)
    picked = rng.sample(candidates, args.total)

    # manifest
    rows = [{"role": "missing", "cve": c["cve"], "project": c["project"],
             "before": (BEFORE / c["cve"]).as_posix()} for c in picked]
    manifest = {
        "seed": args.seed, "total": len(picked),
        "pool": "missing-type (added lines, no removed lines) - no error code fragment",
        "candidate_pool_size": len(candidates),
        "missing_cves": [c["cve"] for c in picked],
        "rows": rows,
    }
    mpath = ROOT / "reports" / "negative_exp_manifest_400_missing.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[1] manifest: {mpath}")

    # batch
    items = [{"role": "missing", "cve": c["cve"], "project": c["project"],
              "target_dir": (BEFORE / c["cve"]).as_posix(), "output_dir": c["cve"]}
             for c in picked]
    batch = {
        "description": f"缺失型测试集（评审#5）：{len(items)} 个缺失型 CVE。seed={args.seed}。",
        "total": len(items), "items": items,
    }
    bpath = ROOT / "utils" / "experiments" / "test_400_missing_batch.json"
    bpath.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[2] batch: {bpath}  total={len(items)}")

    print("\n完成。下一步：跑 batch(missing) → 检查这些缺失型 CVE 的二次校验是否误报(错配)。")


if __name__ == "__main__":
    main()
