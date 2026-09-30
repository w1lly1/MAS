# -*- coding: utf-8 -*-
"""离线视图消融：复用二次校验 agent 的检索+门控，对 seed=2025 的 kb CVE 分别用
单层 semantic/code_pattern/solution/full 及全层重算召回，得到各向量层贡献。

运行（GPU，需 Weaviate + HF_HOME）：
    python utils/experiments/ablate_views_offline.py --limit 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent  # noqa: E402
from utils.scan_discovery import discover_source_files  # noqa: E402

BATCH = ROOT / "utils" / "experiments" / "test_400_error_batch.json"
DB = ROOT / "infrastructure" / "database" / "mas.db"

# layer_mode -> 实际查询的 weaviate 层（None = 全部四层）
MODES = {
    "all(4层)": None,
    "full": "full_only",
    "semantic": "semantic_only",
    "code_pattern": "code_pattern_only",
    "solution": "solution_only",
}


def _own_in_new_findings(new_findings, own_id) -> bool:
    for nf in new_findings or []:
        ev = nf.get("evidence") if isinstance(nf, dict) else {}
        if ev.get("sqlite_id") == own_id:
            return True
    return False


async def _run_one(agent, target_dir: str, own_id, layer_mode):
    disc = discover_source_files(target_dir)
    files = [f["path"] for f in disc.get("files", [])]
    if not files:
        return False
    for fp in files:
        report_data = {
            "file": fp,
            "issues": [],
            "run_id": "ablate",
            "requirement_id": 1,
        }
        try:
            r = await agent._run_second_pass(report_data, original_analysis=None, layer_mode=layer_mode)
        except Exception as e:
            print(f"    [err] {fp}: {e}")
            continue
        if _own_in_new_findings(r.get("new_findings"), own_id):
            return True
    return False


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    agent = AIDrivenSecondPassAnalysisAgent()
    await agent.initialize()

    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    kb_items = [it for it in batch["items"] if it.get("role") == "kb"][: args.limit]

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id, title FROM issue_patterns")
    id_by_title = {t: i for i, t in cur.fetchall()}
    con.close()

    print(f"kb 样本数: {len(kb_items)}")
    for mode_name, layer_mode in MODES.items():
        hits = 0
        miss_no_own = 0
        for idx, it in enumerate(kb_items, 1):
            cve = it.get("cve")
            own = id_by_title.get(cve)
            if own is None:
                miss_no_own += 1
                continue
            ok = await _run_one(agent, it["target_dir"], own, layer_mode)
            if ok:
                hits += 1
            if idx % 20 == 0:
                print(f"  [{mode_name}] {idx}/{len(kb_items)} done")
        print(f"{mode_name}: 召回 {hits}/{len(kb_items)} = {hits/len(kb_items):.1%}  (own缺失 {miss_no_own})")

    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
