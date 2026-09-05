# -*- coding: utf-8 -*-
"""Build five-fold leave-out structured KB JSONs + checklists.

Uses the same 50 CVEs as reports/held_out_manifest.json.
Fold k held-out: idx % 5 == k % 5 (idx 1→fold1, …, idx 5/10/…→fold5).
Each fold: 10 held-out, 40 in-KB; JSON format matches structured_ingest_kb40.json.

Run from repo root:
    python -m utils.bigvul_ingest.build_five_fold_kb
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .build_kb40_tasks import _build_payload_for_cves, _validate
    from .build_structured_ingest import BuildConfig
except ImportError:
    from build_kb40_tasks import _build_payload_for_cves, _validate
    from build_structured_ingest import BuildConfig

DEFAULT_DS = Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured")
DEFAULT_MANIFEST = Path("reports/held_out_manifest.json")
DEFAULT_FOLD_MANIFEST = Path("reports/five_fold_manifest.json")
DEFAULT_OUTPUT = Path("utils/bigvul_ingest/output/five_fold")


def _fold_of_idx(idx: int) -> int:
    """Map 1-based idx to fold 1..5 (idx%5==1→1, …, idx%5==0→5)."""
    r = idx % 5
    return 5 if r == 0 else r


def _split_folds(manifest: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    folds: Dict[int, Dict[str, List[Dict[str, Any]]]] = {
        k: {"held": [], "kb": []} for k in range(1, 6)
    }
    for row in manifest:
        idx = int(row["idx"])
        held_fold = _fold_of_idx(idx)
        for k in range(1, 6):
            role = "held" if k == held_fold else "kb"
            folds[k][role].append(row)
    return folds


def _write_checklist(
    path: Path,
    *,
    fold: int,
    session_id: str,
    json_name: str,
    held: List[Dict[str, Any]],
    kb: List[Dict[str, Any]],
) -> None:
    lines: List[str] = [
        f"# Fold {fold} 校验表",
        "",
        f"- session_id: `{session_id}`",
        f"- 知识库 JSON: `{json_name}`",
        f"- 入库 CVE 数: {len(kb)}",
        f"- 留出 CVE 数: {len(held)}（本折评测对象，**不入库**）",
        "",
        "## 1. 留出待评测（须校验，不入库）",
        "",
        "| CVE | idx | project | gt_type | before |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in sorted(held, key=lambda x: int(x["idx"])):
        lines.append(
            f"| {r['cve']} | {r['idx']} | {r.get('project', '')} | "
            f"{r.get('gt_type', '')} | `{r.get('before', '')}` |"
        )

    lines.extend(
        [
            "",
            "## 2. 本折知识库（入库，可抽检）",
            "",
            "| CVE | idx | project | gt_type |",
            "| --- | --- | --- | --- |",
        ]
    )
    for r in sorted(kb, key=lambda x: int(x["idx"])):
        lines.append(
            f"| {r['cve']} | {r['idx']} | {r.get('project', '')} | {r.get('gt_type', '')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_fold_manifest_rows(
    folds: Dict[int, Dict[str, List[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        for role in ("held", "kb"):
            for r in folds[fold][role]:
                rows.append(
                    {
                        "fold": fold,
                        "role": role,
                        "cve": r["cve"],
                        "idx": int(r["idx"]),
                        "project": r.get("project", ""),
                        "gt_type": r.get("gt_type", ""),
                        "before": r.get("before", ""),
                        "metadata": r.get("metadata", ""),
                        "after": r.get("after", ""),
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 5-fold leave-out structured KB JSONs")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--fold-manifest-out", type=Path, default=DEFAULT_FOLD_MANIFEST)
    p.add_argument("--metadata-root", type=Path, default=DEFAULT_DS / "metadata")
    p.add_argument("--before-root", type=Path, default=DEFAULT_DS / "before")
    p.add_argument("--after-root", type=Path, default=DEFAULT_DS / "after")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--ingest-mode", type=str, default="strict")
    p.add_argument("--max-snippet-chars", type=int, default=2000)
    p.add_argument(
        "--session-prefix",
        type=str,
        default=f"bigvul-5fold-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest: List[Dict[str, Any]] = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert len(manifest) == 50, f"expected 50 CVEs, got {len(manifest)}"

    folds = _split_folds(manifest)
    for k in range(1, 6):
        assert len(folds[k]["held"]) == 10, f"fold{k} held={len(folds[k]['held'])}"
        assert len(folds[k]["kb"]) == 40, f"fold{k} kb={len(folds[k]['kb'])}"

    # Path existence check
    path_errors: List[str] = []
    for r in manifest:
        for key in ("before", "metadata"):
            rel = r.get(key, "")
            p = Path(rel)
            if not p.exists():
                path_errors.append(f"missing {key}: {rel}")
    if path_errors:
        raise SystemExit("Path check failed:\n  " + "\n  ".join(path_errors[:20]))

    fold_rows = _build_fold_manifest_rows(folds)
    args.fold_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.fold_manifest_out.write_text(
        json.dumps(
            {
                "version": "1.0",
                "source_manifest": str(args.manifest).replace("\\", "/"),
                "rule": "held_fold = idx%5; 0 maps to fold 5; each fold KB = other 40",
                "n_cves": 50,
                "n_folds": 5,
                "rows": fold_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote fold manifest: {args.fold_manifest_out}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for fold in range(1, 6):
        kb_rows = folds[fold]["kb"]
        held_rows = folds[fold]["held"]
        kb_cves = [r["cve"] for r in sorted(kb_rows, key=lambda x: int(x["idx"]))]
        held_cves = [r["cve"] for r in sorted(held_rows, key=lambda x: int(x["idx"]))]

        json_name = f"structured_ingest_fold{fold}_kb40.json"
        session_id = f"{args.session_prefix}-fold{fold}"
        cfg = BuildConfig(
            metadata_root=args.metadata_root,
            before_root=args.before_root,
            after_root=args.after_root,
            output_dir=args.output_dir,
            output_name=json_name,
            start=0,
            count=40,
            max_snippet_chars=args.max_snippet_chars,
            session_id=session_id,
            ingest_mode=args.ingest_mode,
        )

        # Temporary override session message via rebuilding with fold-aware message:
        # _build_payload_for_cves hardcodes "Ingesting KB40 split"; patch by post-edit.
        payload = _build_payload_for_cves(kb_cves, cfg)
        msg = f"Ingesting 5-fold KB (fold {fold}: {len(kb_cves)} in-KB, held-out {len(held_cves)})"
        for entry in payload.get("data", []):
            for inst in entry.get("instances") or []:
                sm = inst.get("session_meta") or {}
                sm["session_id"] = session_id
                sm["user_message"] = msg
                inst["session_meta"] = sm

        out_path = args.output_dir / json_name
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        validation = _validate(payload, 40)
        titles = set(validation["titles"])
        expected = set(kb_cves)
        match = titles == expected
        print(
            f"Fold {fold}: wrote {out_path.name}; "
            f"valid={validation['valid']} titles_match={match}; "
            f"held={held_cves}"
        )
        if not validation["valid"] or not match:
            all_ok = False
            for prob in validation["problems"][:10]:
                print("  -", prob)
            if not match:
                print("  missing", sorted(expected - titles))
                print("  extra", sorted(titles - expected))

        checklist = args.output_dir / f"fold{fold}_checklist.md"
        _write_checklist(
            checklist,
            fold=fold,
            session_id=session_id,
            json_name=json_name,
            held=held_rows,
            kb=kb_rows,
        )
        print(f"  checklist: {checklist}")

    # Spot-check: fold1 held contains CVE-2002-2443
    fold1_held = {r["cve"] for r in folds[1]["held"]}
    if "CVE-2002-2443" not in fold1_held:
        all_ok = False
        print("ERROR: fold1 held should contain CVE-2002-2443")
    else:
        print("Spot-check OK: fold1 held contains CVE-2002-2443")

    if not all_ok:
        raise SystemExit(1)
    print("All folds OK.")


if __name__ == "__main__":
    main()
