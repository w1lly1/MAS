# -*- coding: utf-8 -*-
"""结构化通道 vs 向量通道 分通道召回拆解（本地 CPU，numpy 复刻检索）。

对每个 kb CVE 分别计算三种「通道配置」下的自召回：
  - all        : 结构化(sqlite+curated) + 向量(weaviate 4 层)   → 与 #4 的 138 一致
  - structured : 只保留结构化候选（去掉所有 weaviate 候选）
  - vector     : 只保留向量候选（去掉 sqlite/curated 候选）

据此把召回拆成 structured-only / vector-only / both / neither 四类。
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
from utils.experiments.ablate_views_singlepass import (  # noqa: E402
    _gated_weaviate_candidates,
    _load_ci_to_pattern,
    _load_id_by_title,
    _own_in_findings,
    _shard_path,
)
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
    if target_dir and target_dir.startswith("/root/autodl-tmp/MAS/"):
        return str(ROOT / target_dir[len("/root/autodl-tmp/MAS/"):])
    return target_dir


def _recall_channel(agent, gap, sqlite_patterns, own_id, fp, ci_to_pattern, mode):
    re_gated = []
    for e in gap or []:
        if not isinstance(e, dict):
            continue
        ev = dict(e)
        cc = e.get("code_chunk")
        if isinstance(cc, dict):
            issue_like = agent._code_chunk_as_issue(cc)
        else:
            issue_like = {"description": e.get("issue_description"), "line": None, "file": e.get("issue_file")}
        issue_desc = str(issue_like.get("description") or "")
        issue_file = issue_like.get("file")
        constant = [c for c in (e.get("candidates") or []) if isinstance(c, dict) and c.get("channel") != "weaviate"]
        hits = [h for h in (e.get("weaviate_hits") or []) if isinstance(h, dict)]
        if mode == "structured":
            hits = []
        elif mode == "vector":
            constant = []
        wc = _gated_weaviate_candidates(agent, hits, issue_desc, issue_file, issue_like, sqlite_patterns)
        ev["candidates"] = constant + wc
        ev["weaviate_hits"] = hits
        re_gated.append(ev)
    findings = agent._derive_new_findings_from_gap_evidence(re_gated, [], [], "ablate", 1, fp)
    return _own_in_findings(findings, own_id, ci_to_pattern)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0=全部），用于冒烟")
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "channel_breakdown.jsonl"))
    ap.add_argument("--results", type=str, default="")
    args = ap.parse_args()

    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    # 读已完成的视图消融结果，得到「已召回」的 CVE 集合（只对这些重跑，省 31% 时间）
    recalled = None
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = ROOT / "reports"
    recalled_files = sorted(Path(results_path).glob("ablate_local_results_shard*.jsonl")) if results_path.is_dir() else [results_path]
    if recalled_files:
        recalled = set()
        for f in recalled_files:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("all(4层)"):
                    recalled.add(r["cve"])

    agent = _build_agent()
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    kb_items = [it for it in batch["items"] if it.get("role") == "kb"]
    if recalled is not None:
        kb_items = [it for it in kb_items if it.get("cve") in recalled]
    if args.shard_count > 1:
        kb_items = [it for i, it in enumerate(kb_items) if i % args.shard_count == args.shard_id]
    if args.limit > 0:
        kb_items = kb_items[: args.limit]

    id_by_title = _load_id_by_title(str(DB))
    ci_to_pattern = _load_ci_to_pattern(str(DB))
    sqlite_patterns = await agent.db_service.get_issue_patterns(status="active")
    sqlite_patterns = sqlite_patterns[: getattr(agent, "max_sqlite_patterns", 2000)]

    out_path = _shard_path(args.out, args.shard_id, args.shard_count)
    out_f = open(out_path, "w", encoding="utf-8")
    print(f"待跑 CVE 数: {len(kb_items)}（shard {args.shard_id}/{args.shard_count}）", flush=True)
    for idx, it in enumerate(kb_items, 1):
        cve = it.get("cve")
        own = id_by_title.get(cve)
        if own is None:
            print(f"[{idx}/{len(kb_items)}] {cve}: own 缺失", flush=True)
            continue
        t0 = time.time()
        files = [f["path"] for f in discover_source_files(_local_remap(it["target_dir"])).get("files", [])]
        if not files:
            print(f"[{idx}/{len(kb_items)}] {cve}: 无源文件", flush=True)
            continue
        res = {"all": False, "structured": False, "vector": False}
        for fp in files:
            report_data = {"file": fp, "issues": [], "run_id": "ablate", "requirement_id": 1}
            try:
                r = await agent._run_second_pass(report_data, original_analysis=None, layer_mode=None)
            except Exception as e:
                print(f"[{idx}/{len(kb_items)}] {cve} {fp}: 异常 {e}", flush=True)
                continue
            gap = r.get("gap_retrieval_evidence") or []
            for mode in ("all", "structured", "vector"):
                if not res[mode] and _recall_channel(agent, gap, sqlite_patterns, own, fp, ci_to_pattern, mode):
                    res[mode] = True
        out_f.write(json.dumps({"cve": cve, **res}, ensure_ascii=False) + "\n")
        out_f.flush()
        dt = time.time() - t0
        print(f"[{idx}/{len(kb_items)}] {cve} done {dt:.1f}s all={res['all']} structured={res['structured']} vector={res['vector']}", flush=True)

    out_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
