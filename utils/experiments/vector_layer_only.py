# -*- coding: utf-8 -*-
"""纯向量单层消融（剔除词法-结构通道贡献，seed=2025 本地 CPU numpy 复刻）。

与 ablate_local 的区别：本脚本把结构化候选（channel != "weaviate"）全部剔除
（constant=[]），只保留向量候选，考察各向量层【独立】能召回多少自身条目。

配置（每 CVE 跑一次 _run_second_pass 拿 gap，内存里重算 6 种）：
  - vector_all        : 4 层向量全保留（校验：应与 channel_breakdown 的 vector=71 一致）
  - vector_full       : 仅 full 层
  - vector_semantic   : 仅 semantic 层
  - vector_code_pattern: 仅 code_pattern 层
  - vector_solution   : 仅 solution 层
  - structured_only   : 仅结构化候选（校验：应与 channel_breakdown 的 structured=134 一致）
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

MODES = {
    "vector_all": None,
    "vector_full": {"full"},
    "vector_semantic": {"semantic"},
    "vector_code_pattern": {"code_pattern"},
    "vector_solution": {"solution"},
}


def _build_agent():
    agent = AIDrivenSecondPassAnalysisAgent()
    agent.vector_service = NumpyVectorService(KB_DUMP)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    return agent


def _local_remap(target_dir: str) -> str:
    if not target_dir:
        return target_dir
    if target_dir.startswith("/root/autodl-tmp/MAS/"):
        return str(ROOT / target_dir[len("/root/autodl-tmp/MAS/"):])
    return target_dir


def _recall_mode(agent, gap, sqlite_patterns, layer_subset, keep_structured, own, fp, ci_to_pattern):
    """重算某配置下的 self 召回。keep_structured=False 时剔除结构化候选。"""
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
        hits = [h for h in (e.get("weaviate_hits") or []) if isinstance(h, dict)]
        if layer_subset is not None:
            hits = [h for h in hits if str(h.get("vector_layer") or "").strip().lower() in layer_subset]
        wc = _gated_weaviate_candidates(agent, hits, issue_desc, issue_file, issue_like, sqlite_patterns)
        constant = []
        if keep_structured:
            constant = [c for c in (e.get("candidates") or []) if isinstance(c, dict) and c.get("channel") != "weaviate"]
        ev["candidates"] = constant + wc
        ev["weaviate_hits"] = hits
        re_gated.append(ev)
    findings = agent._derive_new_findings_from_gap_evidence(re_gated, [], [], "vablate", 1, fp)
    return _own_in_findings(findings, own, ci_to_pattern)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "vector_layer_only.jsonl"))
    args = ap.parse_args()

    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    agent = _build_agent()
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    kb_items = [it for it in batch["items"] if it.get("role") == "kb"]
    if args.limit > 0:
        kb_items = kb_items[: args.limit]
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
    totals = {m: 0 for m in list(MODES) + ["structured_only"]}
    # 从已有文件恢复 totals
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for m in totals:
                if rec.get(m):
                    totals[m] += 1
    print(f"待跑 kb 样本: {len(kb_items)}（shard {args.shard_id}/{args.shard_count}）"
          + (f"，已跳过 {len(done_cves)} 个" if args.resume else ""), flush=True)
    for idx, it in enumerate(kb_items, 1):
        cve = it.get("cve")
        if cve in done_cves:
            continue
        own = id_by_title.get(cve)
        if own is None:
            print(f"[{idx}/{len(kb_items)}] {cve}: own 缺失，跳过", flush=True)
            continue
        files = [f["path"] for f in discover_source_files(_local_remap(it["target_dir"])).get("files", [])]
        if not files:
            print(f"[{idx}/{len(kb_items)}] {cve}: 无源文件，跳过", flush=True)
            continue
        admitted = {m: False for m in list(MODES) + ["structured_only"]}
        t0 = time.time()
        for fp in files:
            report_data = {"file": fp, "issues": [], "run_id": "vablate", "requirement_id": 1}
            try:
                r = await agent._run_second_pass(report_data, original_analysis=None, layer_mode=None)
            except Exception as e:
                print(f"[{idx}/{len(kb_items)}] {cve} {fp}: 异常 {e}", flush=True)
                continue
            gap = r.get("gap_retrieval_evidence") or []
            if not admitted["structured_only"] and _recall_mode(agent, gap, sqlite_patterns, None, True, own, fp, ci_to_pattern):
                admitted["structured_only"] = True
            for m, subset in MODES.items():
                if not admitted[m] and _recall_mode(agent, gap, sqlite_patterns, subset, False, own, fp, ci_to_pattern):
                    admitted[m] = True
        rec = {"cve": cve, "own_id": own, **admitted}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        for m in rec:
            if m not in ("cve", "own_id") and rec[m]:
                totals[m] += 1
        print(f"[{idx}/{len(kb_items)}] {cve} done {time.time()-t0:.1f}s "
              + " ".join(f"{m}={totals[m]}" for m in totals), flush=True)

    out_f.close()
    n = len(kb_items)
    print(f"\n==== 纯向量单层消融（shard {args.shard_id}，n={n}，有效 {len(done_cves) + (n - sum(1 for _ in done_cves))}） ====", flush=True)
    for m in list(MODES) + ["structured_only"]:
        print(f"{m}: {totals[m]}/{n} = {totals[m]/n:.1%}" if n else f"{m}: 0", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
