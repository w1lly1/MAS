# -*- coding: utf-8 -*-
"""Patched + Benign 硬负样本（用 after 修复版，纯 CPU 离线）。

- Patched：200 kb CVE 的 after 版（答案在库、但代码已修复）→ 二次校验是否误报。
- Benign ：200 held CVE 的 after 版（答案不在库、修复后为无关普通代码）→ 是否误报。

口径：对 after 代码跑 _run_second_pass(issues=[])，new_findings 条数 > 0 即误报。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

from core.agents.ai_driven_second_pass_analysis_agent import (  # noqa: E402
    AIDrivenSecondPassAnalysisAgent,
)
from infrastructure.database.sqlite.service import DatabaseService  # noqa: E402
from utils.experiments.local_numpy_weaviate import NumpyVectorService  # noqa: E402
from utils.scan_discovery import discover_source_files  # noqa: E402

BATCH = ROOT / "论文" / "test_400_error_batch.json"
DB = ROOT / "infrastructure" / "database" / "mas.db"
KB_DUMP = ROOT / "reports" / "weaviate_kb_seed2024.jsonl"
AFTER = ROOT / "tests" / "BigVul" / "MSR_20_Code_vulnerability_CSV_Dataset" / "source_code_restructured" / "after"


def _build_agent():
    agent = AIDrivenSecondPassAnalysisAgent()
    agent.vector_service = NumpyVectorService(KB_DUMP)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    return agent


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--role", type=str, default="both", choices=["kb", "held", "both"])
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "hard_negative_after.jsonl"))
    args = ap.parse_args()

    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    agent = _build_agent()
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    items = [it for it in batch["items"] if args.role == "both" or it.get("role") == args.role]
    if args.limit > 0:
        items = items[: args.limit]
    if args.shard_count > 1:
        items = [it for i, it in enumerate(items) if i % args.shard_count == args.shard_id]

    from utils.experiments.ablate_views_singlepass import _shard_path
    out_path = _shard_path(args.out, args.shard_id, args.shard_count)
    out_f = open(out_path, "w", encoding="utf-8")

    fp_cnt = {"kb": 0, "held": 0}
    n_cnt = {"kb": 0, "held": 0}
    print(f"待跑样本: {len(items)}（shard {args.shard_id}/{args.shard_count}）", flush=True)
    for idx, it in enumerate(items, 1):
        cve = it.get("cve")
        role = it.get("role")
        ad = AFTER / cve
        files = [f["path"] for f in discover_source_files(str(ad)).get("files", [])] if ad.exists() else []
        if not files:
            print(f"[{idx}/{len(items)}] {cve} ({role}): 无 after 文件，跳过", flush=True)
            continue
        n_cnt[role] += 1
        nf_count = 0
        t0 = time.time()
        for fp in files:
            report_data = {"file": fp, "issues": [], "run_id": "hn", "requirement_id": 1}
            try:
                r = await agent._run_second_pass(report_data, original_analysis=None, layer_mode=None)
            except Exception as e:
                print(f"[{idx}/{len(items)}] {cve} {fp}: 异常 {e}", flush=True)
                continue
            nf_count += len(r.get("new_findings") or [])
        is_fp = nf_count > 0
        if is_fp:
            fp_cnt[role] += 1
        out_f.write(json.dumps({"cve": cve, "role": role, "new_findings": nf_count, "fp": is_fp}, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"[{idx}/{len(items)}] {cve} ({role}) fp={is_fp} nf={nf_count} {time.time()-t0:.1f}s", flush=True)

    out_f.close()
    for r in ("kb", "held"):
        if n_cnt[r]:
            print(f"{r}(after) 误报: {fp_cnt[r]}/{n_cnt[r]} = {fp_cnt[r]/n_cnt[r]:.1%}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
