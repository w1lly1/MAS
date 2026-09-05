# -*- coding: utf-8 -*-
"""TODO1: 只从「错误型 且 能提取有效错误代码」的 BigVul CVE 里重抽 400 测试集。

口径（与 stat_error_type_coverage.py 完全一致）：
  - 错误型（error）：任一代码文件 diff 存在删除行（- 行，非 --- 头）
  - 有效错误代码：删除行拼接后 token 化 >=4 且非全通用 token
完整性（与 prepare_400_test.py 一致）：
  - before/after 目录有代码文件 + cve_metadata.json 有 cve_id 和 summary

产出（新文件名，不覆盖旧 400 全量实验产物）：
  1. reports/negative_exp_manifest_400_error.json
  2. utils/bigvul_ingest/output/negative_exp/structured_ingest_kb200_error.json
  3. utils/experiments/test_400_error_batch.json

运行（MAS 根目录）：
    venv/Scripts/python.exe utils/experiments/prepare_400_error_only_test.py
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.bigvul_ingest.build_structured_ingest import (  # noqa: E402
    BuildConfig,
    _find_commit_meta_paths,
    _load_json,
)
from utils.bigvul_ingest.build_kb40_tasks import (  # noqa: E402
    _build_payload_for_cves,
    _validate,
)

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
METADATA = DS / "metadata"
BEFORE = DS / "before"
AFTER = DS / "after"
OUT_DIR = ROOT / "utils/bigvul_ingest/output/negative_exp"

CODE_EXTS = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".s", ".S"}
TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|[+\-*/%=<>!&|^~]")
GENERIC = {"if", "else", "for", "while", "return", "int", "char", "void", "size", "len",
           "sizeof", "null", "true", "false", "0", "1", "break", "continue", "goto",
           "case", "switch", "do", "struct", "unsigned", "long", "short", "static", "const",
           "err", "ret", "i", "j", "k", "buf", "data", "ptr", "tmp", "res", "len_", "count"}


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _tokenize(text):
    return TOKEN_RE.findall(text or "")


def diff_lines(before_text, after_text):
    removed, added = [], []
    for line in difflib.unified_diff(
        before_text.splitlines(), after_text.splitlines(), lineterm="", n=0
    ):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].strip())
    return [r for r in removed if r], [a for a in added if a]


def classify_cve(cve_dir: Path):
    """返回 (has_removed, has_added, removed_all) 聚合该 CVE 所有代码文件。"""
    has_removed = False
    has_added = False
    removed_all = []
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
            after_path = AFTER / cve_dir.name / commit_dir / local_name
            removed, added = diff_lines(_safe_read(before_path), _safe_read(after_path))
            if removed:
                has_removed = True
                removed_all.extend(removed)
            if added:
                has_added = True
    return has_removed, has_added, removed_all


def has_valid_error_code(removed_lines):
    toks = _tokenize(" ".join(removed_lines))
    if len(toks) < 4:
        return False
    if all(t.lower() in GENERIC for t in toks):
        return False
    return True


def _has_code_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(f.is_file() for f in path.rglob("*"))


def is_complete(cve_dir: Path) -> bool:
    """完整性过滤（与 prepare_400_test.load_cves 一致）。"""
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
    if not _has_code_files(AFTER / cve_id):
        return False
    return True


def load_error_candidate_cves() -> list:
    """返回「错误型 且 能提取有效错误代码 且 完整性」三条件齐全的 CVE 列表。"""
    candidates = []
    skipped = defaultdict(int)
    for d in sorted(METADATA.glob("CVE-*")):
        if not (d / "cve_metadata.json").exists():
            continue
        has_removed, has_added, removed_all = classify_cve(d)
        if not has_removed:
            skipped["非错误型(缺失型/无变化)"] += 1
            continue
        if not has_valid_error_code(removed_all):
            skipped["错误型但无有效错误代码"] += 1
            continue
        if not is_complete(d):
            skipped["不满足完整性(before/after/summary)"] += 1
            continue
        try:
            meta = json.loads((d / "cve_metadata.json").read_text(encoding="utf-8"))
        except Exception:
            skipped["metadata读取失败"] += 1
            continue
        candidates.append({
            "cve": str(meta.get("cve_id", "")).strip(),
            "project": meta.get("project", "unknown"),
            "lang": meta.get("lang", ""),
        })
    return candidates, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=400)
    ap.add_argument("--kb", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    candidates, skipped = load_error_candidate_cves()
    print(f"候选池（错误型+有效错误代码+完整性）: {len(candidates)}")
    for k, v in skipped.items():
        print(f"  跳过[{k}]: {v}")
    if len(candidates) < args.total:
        raise SystemExit(f"❌ 候选池不足 {args.total}，只有 {len(candidates)}")

    rng = random.Random(args.seed)
    picked = rng.sample(candidates, args.total)
    kb = picked[: args.kb]
    held = picked[args.kb:]
    print(f"抽样: kb={len(kb)}  held={len(held)}  seed={args.seed}")

    kb_proj = defaultdict(int)
    for c in kb:
        kb_proj[c["project"]] += 1
    print("建库池项目分布 Top8:", dict(sorted(kb_proj.items(), key=lambda x: -x[1])[:8]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) manifest
    rows = []
    for c in kb:
        rows.append({"role": "kb", "cve": c["cve"], "project": c["project"],
                     "before": (BEFORE / c["cve"]).as_posix()})
    for c in held:
        rows.append({"role": "held", "cve": c["cve"], "project": c["project"],
                     "before": (BEFORE / c["cve"]).as_posix()})
    manifest = {
        "seed": args.seed, "total": args.total, "kb": len(kb), "held": len(held),
        "pool": "error-type with valid error code (removed lines token>=4, non-all-generic)",
        "candidate_pool_size": len(candidates),
        "kb_cves": [c["cve"] for c in kb],
        "held_cves": [c["cve"] for c in held],
        "rows": rows,
    }
    mpath = ROOT / "reports" / "negative_exp_manifest_400_error.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[1] manifest: {mpath}")

    # 2) 建库 JSON（200 条）
    cfg = BuildConfig(
        metadata_root=METADATA, before_root=BEFORE, after_root=AFTER,
        output_dir=OUT_DIR, output_name="structured_ingest_kb200_error.json",
        start=0, count=len(kb), max_snippet_chars=2000,
        session_id=f"neg-exp-kb{len(kb)}-error", ingest_mode="strict",
    )
    payload = _build_payload_for_cves([c["cve"] for c in kb], cfg)
    kb_path = OUT_DIR / "structured_ingest_kb200_error.json"
    kb_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    val = _validate(payload, len(kb))
    print(f"[2] 建库 JSON: {kb_path}  valid={val['valid']} titles={len(val['titles'])}")
    if not val["valid"]:
        for p in val.get("problems", [])[:10]:
            print("   -", p)

    # 3) batch 配置（400 项：200 kb + 200 held）
    items = []
    for c in kb:
        items.append({"role": "kb", "cve": c["cve"], "project": c["project"],
                      "target_dir": (BEFORE / c["cve"]).as_posix(),
                      "output_dir": c["cve"]})
    for c in held:
        items.append({"role": "held", "cve": c["cve"], "project": c["project"],
                      "target_dir": (BEFORE / c["cve"]).as_posix(),
                      "output_dir": c["cve"]})
    batch = {
        "description": (
            f"TODO1 重抽 400 样本（仅错误型+有效错误代码）：{len(kb)} kb(召回,答案在库) "
            f"+ {len(held)} held(误报,答案不在库)。seed={args.seed}。"
            "跑完用 evaluate_400.py --manifest reports/negative_exp_manifest_400_error.json 统计。"
        ),
        "kb_count": len(kb), "held_count": len(held), "total": len(items),
        "items": items,
    }
    bpath = ROOT / "utils" / "experiments" / "test_400_error_batch.json"
    bpath.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3] batch 配置: {bpath}  total={len(items)}")

    print("\n完成。下一步：清库 → 灌库(kb200_error) → 跑 batch(400_error) → 评测")


if __name__ == "__main__":
    main()
