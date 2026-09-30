# -*- coding: utf-8 -*-
"""seed=2025 held 组错配侧重跑（v5 门控口径，本地 CPU numpy 复刻）。

目的：把 seed=2025 的 held-before（跨 CVE 漏洞代码）错配样本率，从旧协议跑批
的 2.0% 复验到 v5 门控口径（θ_s=0.65/τ=0.65 两通道 DNF）。

口径：对 held CVE 的 before 代码跑 _run_second_pass(issues=[])，
new_findings 条数 > 0 即误报。与 hard_negative_patched_benign.py 的 held 分支一致，
但改用 seed=2025 batch + mas_seed2025.db + weaviate_kb_dump.jsonl（seed=2025 KB）。
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

BATCH = ROOT / "utils" / "experiments" / "test_400_error_batch_seed2025.json"
DB = ROOT / "infrastructure" / "database" / "mas_seed2025.db"
KB_DUMP = ROOT / "utils" / "experiments" / "weaviate_kb_dump.jsonl"


def _build_agent():
    agent = AIDrivenSecondPassAnalysisAgent()
    agent.vector_service = NumpyVectorService(KB_DUMP)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    return agent


def _local_remap(target_dir: str) -> str:
    """batch 里的 Linux 路径 /root/autodl-tmp/MAS/... 重映射为本地 MAS 根目录。"""
    if not target_dir:
        return target_dir
    if target_dir.startswith("/root/autodl-tmp/MAS/"):
        return str(ROOT / target_dir[len("/root/autodl-tmp/MAS/"):])
    return target_dir


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0=全部），用于冒烟")
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "held_seed2025_v5.jsonl"))
    args = ap.parse_args()

    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    agent = _build_agent()
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    items = [it for it in batch["items"] if it.get("role") == "held"]
    if args.limit > 0:
        items = items[: args.limit]
    if args.shard_count > 1:
        items = [it for i, it in enumerate(items) if i % args.shard_count == args.shard_id]

    from utils.experiments.ablate_views_singlepass import _shard_path
    out_path = _shard_path(args.out, args.shard_id, args.shard_count)
    out_f = open(out_path, "w", encoding="utf-8")

    fp_cnt = 0
    n_cnt = 0
    print(f"待跑 held 样本: {len(items)}（shard {args.shard_id}/{args.shard_count}）", flush=True)
    for idx, it in enumerate(items, 1):
        cve = it.get("cve")
        td = _local_remap(it.get("target_dir"))
        files = [f["path"] for f in discover_source_files(td).get("files", [])] if td and Path(td).is_dir() else []
        if not files:
            print(f"[{idx}/{len(items)}] {cve}: 无 before 文件，跳过", flush=True)
            continue
        n_cnt += 1
        nf_count = 0
        t0 = time.time()
        for fp in files:
            report_data = {"file": fp, "issues": [], "run_id": "held2025", "requirement_id": 1}
            try:
                r = await agent._run_second_pass(report_data, original_analysis=None, layer_mode=None)
            except Exception as e:
                print(f"[{idx}/{len(items)}] {cve} {fp}: 异常 {e}", flush=True)
                continue
            nf_count += len(r.get("new_findings") or [])
        is_fp = nf_count > 0
        if is_fp:
            fp_cnt += 1
        out_f.write(json.dumps({"cve": cve, "new_findings": nf_count, "fp": is_fp}, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"[{idx}/{len(items)}] {cve} fp={is_fp} nf={nf_count} {time.time()-t0:.1f}s", flush=True)

    out_f.close()
    print(f"\nheld(seed=2025, v5) 误报: {fp_cnt}/{n_cnt} = {fp_cnt/n_cnt:.1%}" if n_cnt else "无有效样本", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
