from __future__ import annotations

"""Build the MAS-ingestable structured KB JSON for the 40-CVE KB split.

与 build_structured_ingest.py 产出格式**完全一致**：
    { "version": "1.0", "ingest_mode": "strict", "data": [ {pattern:{...}, instances:[...]}, ... ] }
（即与 utils/bigvul_ingest/output/structured_ingest_21_50.json 同构）

只针对 reports/held_out_manifest.json 中 split=="KB40" 的 40 个 CVE 生成。
字段推导复用 build_structured_ingest 的内部逻辑（_build_pattern / _build_instances），
因此产出可被 MAS 以与原始 50 样本完全相同的方式识别与入库。

运行（仓库根目录）：
    python -m utils.bigvul_ingest.build_kb40_tasks
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from .build_structured_ingest import (
        BuildConfig,
        _build_instances,
        _build_pattern,
        _clean_text,
        _find_commit_meta_paths,
        _load_json,
    )
    from .rules import (
        derive_file_pattern,
        extract_function_name_from_summary,
    )
except ImportError:  # 以脚本方式直接运行时
    from build_structured_ingest import (
        BuildConfig,
        _build_instances,
        _build_pattern,
        _clean_text,
        _find_commit_meta_paths,
        _load_json,
    )
    from rules import (
        derive_file_pattern,
        extract_function_name_from_summary,
    )

DEFAULT_DS = Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured")
DEFAULT_MANIFEST = Path("reports/held_out_manifest.json")
DEFAULT_OUTPUT = Path("utils/bigvul_ingest/output")
DEFAULT_NAME = "structured_ingest_kb40.json"


def _build_payload_for_cves(cve_ids: List[str], cfg: BuildConfig) -> Dict[str, Any]:
    """与 build_structured_ingest.build_payload 同构，但按给定 CVE 列表过滤。"""
    cve_dirs: List[Path] = []
    for cid in cve_ids:
        d = cfg.metadata_root / cid
        if d.is_dir():
            cve_dirs.append(d)
        else:
            print(f"WARN: metadata dir not found, skip: {d}")

    data: List[Dict[str, Any]] = []
    session_message = f"Ingesting KB40 split ({len(cve_ids)} CVEs)"

    for cve_dir in cve_dirs:
        cve_meta_path = cve_dir / "cve_metadata.json"
        if not cve_meta_path.exists():
            print(f"WARN: {cve_dir.name} missing cve_metadata.json, skip")
            continue
        cve_meta = _load_json(cve_meta_path)
        commit_meta_paths = _find_commit_meta_paths(cve_dir)
        instances = _build_instances(
            cve_meta,
            commit_meta_paths,
            cfg.before_root,
            cfg.after_root,
            cfg.max_snippet_chars,
            cfg.session_id,
            session_message,
        )
        first_issue = (instances[0].get("issue") or {}) if instances else {}
        file_pattern = derive_file_pattern(str(first_issue.get("file_path") or ""))
        class_pattern = extract_function_name_from_summary(
            _clean_text(cve_meta.get("summary", ""))
        )
        solution = str(first_issue.get("solution") or "")
        data.append(
            {
                "pattern": _build_pattern(
                    cve_meta,
                    file_pattern=file_pattern,
                    class_pattern=class_pattern,
                    solution=solution,
                ),
                "instances": instances,
            }
        )

    return {
        "version": "1.0",
        "ingest_mode": cfg.ingest_mode,
        "data": data,
    }


def _load_kb40_cves(manifest: Path) -> List[str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return [d["cve"] for d in data if d.get("split") == "KB40"]


def _validate(payload: Dict[str, Any], expected: int) -> Dict[str, Any]:
    problems: List[str] = []
    data = payload.get("data", [])
    if payload.get("version") != "1.0":
        problems.append(f"version != 1.0: {payload.get('version')}")
    if payload.get("ingest_mode") != "strict":
        problems.append(f"ingest_mode != strict: {payload.get('ingest_mode')}")
    if len(data) != expected:
        problems.append(f"data len {len(data)} != expected {expected}")
    titles = []
    for i, entry in enumerate(data):
        pat = entry.get("pattern") or {}
        t = pat.get("title", "")
        titles.append(t)
        for key in ("error_type", "error_description", "problematic_pattern", "solution"):
            if not _clean_text(pat.get(key, "")):
                problems.append(f"data[{i}] pattern missing {key}")
        if "instances" not in entry:
            problems.append(f"data[{i}] missing instances")
    return {"problems": problems, "valid": len(problems) == 0, "titles": sorted(titles)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build structured KB40 ingest JSON from held-out manifest")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--metadata-root", type=Path, default=DEFAULT_DS / "metadata")
    p.add_argument("--before-root", type=Path, default=DEFAULT_DS / "before")
    p.add_argument("--after-root", type=Path, default=DEFAULT_DS / "after")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--output-name", type=str, default=DEFAULT_NAME)
    p.add_argument("--ingest-mode", type=str, default="strict")
    p.add_argument("--max-snippet-chars", type=int, default=2000)
    p.add_argument(
        "--session-id",
        type=str,
        default=f"bigvul-kb40-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cve_ids = _load_kb40_cves(args.manifest)
    print(f"Manifest: {args.manifest} -> KB40 CVE count = {len(cve_ids)}")
    assert len(cve_ids) == 40, f"expected 40 KB CVEs, got {len(cve_ids)}"

    cfg = BuildConfig(
        metadata_root=args.metadata_root,
        before_root=args.before_root,
        after_root=args.after_root,
        output_dir=args.output_dir,
        output_name=args.output_name,
        start=0,
        count=len(cve_ids),
        max_snippet_chars=args.max_snippet_chars,
        session_id=args.session_id,
        ingest_mode=args.ingest_mode,
    )

    payload = _build_payload_for_cves(cve_ids, cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")

    validation = _validate(payload, len(cve_ids))
    print(f"Validation valid={validation['valid']} problems={len(validation['problems'])}")
    for prob in validation["problems"][:20]:
        print("  -", prob)

    # cross-check titles vs manifest KB40
    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    kb = set(d["cve"] for d in man if d["split"] == "KB40")
    print("KB40 titles match manifest:", set(validation["titles"]) == kb)


if __name__ == "__main__":
    main()
