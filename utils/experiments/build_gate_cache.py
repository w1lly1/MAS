# -*- coding: utf-8 -*-
"""构建候选级门控缓存（含 s(x)/v(x)/a(x)/F(x)/self 标记），用于任意公式与任意阈值的秒级重放。

对每个样本做一次重门控，逐候选抽取分量后落盘为 npz：
    cve_id, s, v, a, F_pass, is_kb, is_self, formula_id
这样换公式（只需重建缓存）或换阈值（纯 numpy 重放）都不必再跑模型。

用法：
    python utils/experiments/build_gate_cache.py --shard-id 0 --shard-count 4 --formula old|new
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

import numpy as np  # noqa: E402

from core.agents.ai_driven_second_pass_analysis_agent import (  # noqa: E402
    AIDrivenSecondPassAnalysisAgent as Agent,
)
from infrastructure.database.sqlite.service import DatabaseService  # noqa: E402
from utils.experiments.ablate_views_singlepass import (  # noqa: E402
    _gated_weaviate_candidates,
    _load_ci_to_pattern,
    _load_id_by_title,
)
from utils.experiments.hard_filter_ablation import (  # noqa: E402
    install_caches,
    hard_filter_attribution,
)
from utils.experiments.local_numpy_weaviate import NumpyVectorService  # noqa: E402

DB = ROOT / "infrastructure" / "database" / "mas.db"
KB_DUMP = ROOT / "reports" / "weaviate_kb_seed2024.jsonl"

WEAK = frozenset({
    "phenomenon_in_description", "root_cause_in_description",
    "error_type_in_description", "error_type_in_source",
    "error_description_prefix", "problematic_pattern_prefix",
    "location_in_description", "pattern_in_snippet",
    "file_basename_in_description", "file_pattern",
})
LOCATE = frozenset({
    "file_basename_anchor", "basename_match",
    "class_pattern_in_code", "function_name_in_code",
    "function_in_description",
})


def apply_old() -> None:
    Agent._UNIFIED_STRUCT_FIELDS = {
        "error_code_clone": 0.5,
        "file_basename_anchor": 0.2, "basename_match": 0.2,
        "class_pattern_in_code": 0.25,
        "function_name_in_code": 0.25, "function_in_description": 0.25,
    }
    Agent._UNIFIED_WEAK_FIELDS = WEAK

    def _score(cls, mf):
        mf = set(mf or [])
        s = sum(w for f, w in cls._UNIFIED_STRUCT_FIELDS.items() if f in mf)
        if mf & cls._UNIFIED_WEAK_FIELDS:
            s += 0.1
        return min(1.0, s)

    Agent._unified_structured_score = classmethod(_score)


def apply_new() -> None:
    """0.5·I词元 + 0.4·I定位 + 0.1·I描述。"""
    Agent._UNIFIED_STRUCT_FIELDS = {}
    Agent._UNIFIED_WEAK_FIELDS = WEAK

    def _score(cls, mf):
        mf = set(mf or [])
        s = 0.0
        if "error_code_clone" in mf:
            s += 0.5
        if mf & LOCATE:
            s += 0.4
        if mf & cls._UNIFIED_WEAK_FIELDS:
            s += 0.1
        return min(1.0, s)

    Agent._unified_structured_score = classmethod(_score)


FORMULAS = {"old": apply_old, "new": apply_new}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--formula", type=str, default="new", choices=["old", "new"])
    args = ap.parse_args()

    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    agent = Agent()
    agent.vector_service = NumpyVectorService(KB_DUMP)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    install_caches(agent)
    FORMULAS[args.formula]()
    id_by_title = _load_id_by_title(str(DB))
    ci2p = _load_ci_to_pattern(str(DB))
    sp = await agent.db_service.get_issue_patterns(status="active")
    sp = sp[: getattr(agent, "max_sqlite_patterns", 2000)]

    dump = Path(f"{ROOT}/reports/wsens_dump.jsonl_shard{args.shard_id}.gz")
    cves, S, V, A, F, ISKB, SELF = [], [], [], [], [], [], []
    t0 = time.time()
    n = 0
    with gzip.open(dump, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            cve, role = rec.get("cve"), rec.get("role")
            is_kb = role == "kb"
            own = id_by_title.get(cve)
            for fe in rec.get("files") or []:
                fp = fe.get("file") or ""
                for e in fe.get("gap") or []:
                    if not isinstance(e, dict):
                        continue
                    cc = e.get("code_chunk")
                    il = agent._code_chunk_as_issue(cc) if isinstance(cc, dict) else {
                        "description": e.get("issue_description"), "line": None,
                        "file": e.get("issue_file") or fp}
                    hits = [h for h in (e.get("weaviate_hits") or []) if isinstance(h, dict)]
                    cands = [dict(c) for c in (e.get("candidates") or [])
                             if isinstance(c, dict) and c.get("channel") != "weaviate"]
                    cands += _gated_weaviate_candidates(
                        agent, hits, str(il.get("description") or ""),
                        il.get("file"), il, sp)
                    for c in cands:
                        mf = set(c.get("matched_fields") or [])
                        sid = c.get("sqlite_id")
                        if str(c.get("channel") or "").lower() == "curated_issue":
                            self_hit = bool(ci2p.get(sid) is not None and ci2p.get(sid) == own)
                        else:
                            self_hit = bool(own is not None and sid == own)
                        cves.append(str(cve))
                        S.append(float(agent._unified_structured_score(mf)))
                        V.append(float(c.get("semantic_score") or 0.0))
                        A.append(float(c.get("anchor_score") or 0.0))
                        F.append(not hard_filter_attribution(agent, c))
                        ISKB.append(is_kb)
                        SELF.append(self_hit)
    out = ROOT / "reports" / f"gate_cache_v3_{args.formula}_shard{args.shard_id}.npz"
    np.savez_compressed(
        out,
        cve=np.array(cves, dtype="<U32"),
        S=np.array(S, dtype=np.float32), V=np.array(V, dtype=np.float32),
        A=np.array(A, dtype=np.float32), F=np.array(F, dtype=bool),
        IS_KB=np.array(ISKB, dtype=bool), SELF=np.array(SELF, dtype=bool),
    )
    print(f"样本 {n}，候选 {len(S)}，写入 {out.name}（{time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
