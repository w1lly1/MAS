# -*- coding: utf-8 -*-
"""本地（Windows CPU）离线视图消融：numpy 复刻向量检索 + seed2025 数据。

与 ablate_views_singlepass.py 同一策略（单遍检索 + 内存分层重算），但：
- vector_service 换成 NumpyVectorService（读 weaviate_kb_dump.jsonl，cosine 距离复刻）；
- db_service 指向 mas_seed2025.db；
- 用 local_libs 里的 weaviate stub 满足 agent 的 import，无需安装 weaviate-client。

运行（MAS 根目录）：
    python -u utils/experiments/ablate_views_local.py --limit 200
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "local_libs"))  # weaviate stub 优先
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
from utils.experiments.ablate_views_singlepass import (  # noqa: E402
    MODES,
    _load_ci_to_pattern,
    _load_id_by_title,
    _own_in_findings,
    _recall_for_subset,
    _remap_target_dir,
    _shard_path,
    _summary_from_results,
)
from utils.scan_discovery import discover_source_files  # noqa: E402

BATCH = ROOT / "utils" / "experiments" / "test_400_error_batch_seed2025.json"
DB = ROOT / "infrastructure" / "database" / "mas_seed2025.db"
KB_DUMP = ROOT / "utils" / "experiments" / "weaviate_kb_dump.jsonl"


def _build_agent() -> AIDrivenSecondPassAnalysisAgent:
    agent = AIDrivenSecondPassAnalysisAgent()
    agent.vector_service = NumpyVectorService(KB_DUMP)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    return agent


def _local_remap(target_dir: str) -> str:
    """batch 里的 Linux 路径 /root/autodl-tmp/MAS/... 重映射为本地 MAS 根目录。"""
    if not target_dir:
        return target_dir
    if target_dir.startswith("/root/autodl-tmp/MAS/"):
        rel = target_dir[len("/root/autodl-tmp/MAS/"):]
        return str(ROOT / rel)
    return target_dir


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "ablate_local_results.jsonl"))
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=16, help="torch 线程数（0=不覆盖，默认16，本地多线程更快）")
    args = ap.parse_args()

    if args.threads > 0:
        import os
        import torch
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        torch.set_num_threads(args.threads)

    agent = _build_agent()
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    kb_items = [it for it in batch["items"] if it.get("role") == "kb"][: args.limit]
    if args.shard_count > 1:
        kb_items = [it for i, it in enumerate(kb_items) if i % args.shard_count == args.shard_id]
    id_by_title = _load_id_by_title(str(DB))
    ci_to_pattern = _load_ci_to_pattern(str(DB))

    sqlite_patterns = await agent.db_service.get_issue_patterns(status="active")
    sqlite_patterns = sqlite_patterns[: getattr(agent, "max_sqlite_patterns", 2000)]

    out_path = _shard_path(args.out, args.shard_id, args.shard_count)
    done_cves = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done_cves.add(json.loads(line)["cve"])
            except Exception:
                pass

    out_f = open(out_path, "a" if args.resume else "w", encoding="utf-8")
    totals = {m: 0 for m in MODES}
    print(f"kb 样本数: {len(kb_items)}（shard {args.shard_id}/{args.shard_count}）", flush=True)
    for idx, it in enumerate(kb_items, 1):
        cve = it.get("cve")
        if cve in done_cves:
            continue
        own = id_by_title.get(cve)
        if own is None:
            print(f"[{idx}/{len(kb_items)}] {cve}: own 缺失，跳过", flush=True)
            continue

        t0 = time.time()
        target_dir = _local_remap(it["target_dir"])
        files = [f["path"] for f in discover_source_files(target_dir).get("files", [])]
        if not files:
            print(f"[{idx}/{len(kb_items)}] {cve}: 无源文件，跳过", flush=True)
            continue

        admitted = {m: False for m in MODES}
        for fp in files:
            report_data = {"file": fp, "issues": [], "run_id": "ablate", "requirement_id": 1}
            try:
                r = await agent._run_second_pass(report_data, original_analysis=None, layer_mode=None)
            except Exception as e:
                print(f"[{idx}/{len(kb_items)}] {cve} {fp}: 异常 {e}", flush=True)
                continue
            gap = r.get("gap_retrieval_evidence") or []
            # 一致性校验：重算的 all 与生产 new_findings 自命中一致
            prod_all = _own_in_findings(r.get("new_findings") or [], own, ci_to_pattern)
            regate_all = _recall_for_subset(agent, gap, sqlite_patterns, None, own, fp, ci_to_pattern)
            if prod_all != regate_all:
                print(f"    [warn] all 重算与生产不一致 prod={prod_all} regate={regate_all} ({cve})", flush=True)
            for m, subset in MODES.items():
                if not admitted[m] and _recall_for_subset(agent, gap, sqlite_patterns, subset, own, fp, ci_to_pattern):
                    admitted[m] = True

        rec = {"cve": cve, "own_id": own, **{m: admitted[m] for m in MODES}}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        for m in MODES:
            if admitted[m]:
                totals[m] += 1
        dt = time.time() - t0
        print(
            f"[{idx}/{len(kb_items)}] {cve} done {dt:.1f}s "
            + " ".join(f"{m}={totals[m]}" for m in MODES),
            flush=True,
        )

    out_f.close()
    _summary_from_results(out_path)


if __name__ == "__main__":
    asyncio.run(main())
