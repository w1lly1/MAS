# -*- coding: utf-8 -*-
"""Write mas.py login command lists into five_fold output dir."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../MAS
BEFORE = (
    ROOT
    / "tests"
    / "BigVul"
    / "MSR_20_Code_vulnerability_CSV_Dataset"
    / "source_code_restructured"
    / "before"
)
OUT = ROOT / "utils" / "bigvul_ingest" / "output" / "five_fold"
MANIFEST = ROOT / "reports" / "held_out_manifest.json"
FOLD_MANIFEST = ROOT / "reports" / "five_fold_manifest.json"


def login_cmd(cve: str) -> str:
    target = BEFORE / cve
    return (
        f'python mas.py login '
        f'--target-dir="{target}" '
        f'--output-dir="{cve}"'
    )


def main() -> None:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fm = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    all_lines = [
        "# All 50 CVEs — mas.py login commands",
        f"# Run from repo root: {ROOT}",
        "",
    ]
    for r in sorted(man, key=lambda x: int(x["idx"])):
        all_lines.append(
            f"# idx={r['idx']} project={r.get('project', '')} gt_type={r.get('gt_type', '')}"
        )
        all_lines.append(login_cmd(r["cve"]))
        all_lines.append("")
    all_path = OUT / "login_commands_all50.txt"
    all_path.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"Wrote {all_path}")

    for fold in range(1, 6):
        held = [
            r
            for r in fm["rows"]
            if r["fold"] == fold and r["role"] == "held"
        ]
        held = sorted(held, key=lambda x: int(x["idx"]))
        lines = [
            f"# Fold {fold} held-out (10) — login commands for evaluation targets",
            f"# KB JSON: structured_ingest_fold{fold}_kb40.json",
            "# These CVEs must NOT be in the knowledge base for this fold.",
            "",
        ]
        for r in held:
            lines.append(
                f"# idx={r['idx']} project={r.get('project', '')} gt_type={r.get('gt_type', '')}"
            )
            lines.append(login_cmd(r["cve"]))
            lines.append("")
        path = OUT / f"fold{fold}_login_commands.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {path} ({len(held)} cmds)")


if __name__ == "__main__":
    main()
