# -*- coding: utf-8 -*-
"""纯向量单视图消融（seed=2024 基准，离线重门控，不跑模型）。

与 vector_layer_only.py 逻辑一致（剔除结构化候选、按 vector_layer 过滤命中、
重门控后派生写入条目并判定 self 命中），但复用 weight_sensitivity_s.py 落盘的
gap 证据（wsens_dump*.gz），故无需重跑嵌入与检索。

配置：
  structured_only : 仅结构化候选（对照）
  vector_all      : 四视图向量候选（无结构化兜底）
  vector_solution / vector_full / vector_semantic / vector_code_pattern : 仅单视图
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import asyncio  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

from core.agents.ai_driven_second_pass_analysis_agent import (  # noqa: E402
    AIDrivenSecondPassAnalysisAgent as Agent,
)
from infrastructure.database.sqlite.service import DatabaseService  # noqa: E402
from utils.experiments.ablate_views_singlepass import (  # noqa: E402
    _gated_weaviate_candidates,
    _load_ci_to_pattern,
    _load_id_by_title,
    _own_in_findings,
    _shard_path,
)
from utils.experiments.hard_filter_ablation import install_caches  # noqa: E402
from utils.experiments.local_numpy_weaviate import NumpyVectorService  # noqa: E402

DB = ROOT / "infrastructure" / "database" / "mas.db"
KB_DUMP = ROOT / "reports" / "weaviate_kb_seed2024.jsonl"

MODES = {
    "vector_all": (None, False),
    "vector_full": ({"full"}, False),
    "vector_semantic": ({"semantic"}, False),
    "vector_code_pattern": ({"code_pattern"}, False),
    "vector_solution": ({"solution"}, False),
    "structured_only": (None, True),
}


def recall_mode(agent, gap, sqlite_patterns, layer_subset, keep_structured, own, fp, ci2p):
    re_gated = []
    for e in gap or []:
        if not isinstance(e, dict):
            continue
        ev = dict(e)
        cc = e.get("code_chunk")
        if isinstance(cc, dict):
            issue_like = agent._code_chunk_as_issue(cc)
        else:
            issue_like = {"description": e.get("issue_description"),
                          "line": None, "file": e.get("issue_file") or fp}
        hits = [h for h in (e.get("weaviate_hits") or []) if isinstance(h, dict)]
        if layer_subset is not None:
            hits = [h for h in hits
                    if str(h.get("vector_layer") or "").strip().lower() in layer_subset]
        wc = _gated_weaviate_candidates(
            agent, hits, str(issue_like.get("description") or ""),
            issue_like.get("file"), issue_like, sqlite_patterns)
        constant = []
        if keep_structured:
            constant = [c for c in (e.get("candidates") or [])
                        if isinstance(c, dict) and c.get("channel") != "weaviate"]
        ev["candidates"] = constant + wc
        ev["weaviate_hits"] = hits
        re_gated.append(ev)
    findings = agent._derive_new_findings_from_gap_evidence(re_gated, [], [], "vablate24", 1, fp)
    return _own_in_findings(findings, own, ci2p)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "vector_layer_only_seed2024.jsonl"))
    args = ap.parse_args()

    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    agent = Agent()
    agent.vector_service = NumpyVectorService(KB_DUMP)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    install_caches(agent)
    id_by_title = _load_id_by_title(str(DB))
    ci2p = _load_ci_to_pattern(str(DB))
    sp = await agent.db_service.get_issue_patterns(status="active")
    sp = sp[: getattr(agent, "max_sqlite_patterns", 2000)]

    dump = Path(f"{ROOT}/reports/wsens_dump.jsonl_shard{args.shard_id}.gz")
    out_f = open(_shard_path(args.out, args.shard_id, args.shard_count), "w", encoding="utf-8")
    totals = {m: 0 for m in MODES}
    n = 0
    t0 = time.time()
    with gzip.open(dump, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("role") != "kb":
                continue
            n += 1
            cve = rec.get("cve")
            own = id_by_title.get(cve)
            files = [(fe.get("file") or "", fe.get("gap") or []) for fe in rec.get("files") or []]
            admitted = {m: False for m in MODES}
            if own is not None:
                for fp, gap in files:
                    for m, (subset, keep) in MODES.items():
                        if admitted[m]:
                            continue
                        if recall_mode(agent, gap, sp, subset, keep, own, fp, ci2p):
                            admitted[m] = True
            out_f.write(json.dumps({"cve": cve, **admitted}, ensure_ascii=False) + "\n")
            out_f.flush()
            for m in MODES:
                if admitted[m]:
                    totals[m] += 1
            if n % 10 == 0:
                print(f"[{n}] {time.time()-t0:.0f}s " +
                      " ".join(f"{m}={totals[m]}" for m in MODES), flush=True)
    out_f.close()
    print(f"\n==== shard {args.shard_id}：库内组样本 {n} ====", flush=True)
    for m in MODES:
        print(f"{m}: {totals[m]}/{n} = {totals[m]/max(n,1)*100:.1f}%", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
