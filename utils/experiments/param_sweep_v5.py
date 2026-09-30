# -*- coding: utf-8 -*-
"""第 4.8 节参数扫描（seed=2024 基准，离线重门控，不跑模型）。

按论文三项式 s(x) = θ_lex·I_lex + θ_loc·I_loc + θ_desc·I_desc 与
k(x) = [v(x) ≥ τ] ∧ [s(x) ≥ θ_w]、admit(x) = F(x) ∧ [s(x) ≥ θ_s ∨ k(x)]，
逐个参数做单变量全区间扫描（其余固定为定稿值）：
  θ_lex / θ_loc / θ_desc : 0.00 → 1.00（步长 0.05）
  θ_w                    : 0.00 → 1.00（步长 0.05）
  τ                      : 0.50 → 0.85（步长 0.05）
  θ_s                    : 0.30 → 1.00（步长 0.05）
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
    _load_ci_to_pattern,
    _load_id_by_title,
    _own_in_findings,
    _shard_path,
)
from utils.experiments.hard_filter_ablation import (  # noqa: E402
    install_caches,
    regate_variant,
    _ORIG_GATE,
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

BASE = {"lex": 0.5, "loc": 0.4, "desc": 0.1, "ts": 0.65, "tw": 0.2, "tau": 0.65}
LABEL = {"lex": "θ_lex", "loc": "θ_loc", "desc": "θ_desc", "tw": "θ_w",
         "tau": "τ", "ts": "θ_s"}


def build_configs():
    cfgs = [("定稿 θ_lex=0.50 θ_loc=0.40 θ_desc=0.10 θ_s=0.65 θ_w=0.20 τ=0.65", dict(BASE))]
    vals = [round(0.05 * i, 2) for i in range(21)]
    for key in ("lex", "loc", "desc", "tw"):
        for v in vals:
            if abs(v - BASE[key]) < 1e-9:
                continue
            c = dict(BASE)
            c[key] = v
            cfgs.append((f"{LABEL[key]}={v:.2f}", c))
    for v in [round(0.50 + 0.05 * i, 2) for i in range(8)]:      # τ 0.50~0.85
        if abs(v - BASE["tau"]) < 1e-9:
            continue
        c = dict(BASE)
        c["tau"] = v
        cfgs.append((f"τ={v:.2f}", c))
    for v in [round(0.30 + 0.05 * i, 2) for i in range(15)]:     # θ_s 0.30~1.00
        if abs(v - BASE["ts"]) < 1e-9:
            continue
        c = dict(BASE)
        c["ts"] = v
        cfgs.append((f"θ_s={v:.2f}", c))
    return cfgs


CONFIGS = build_configs()


def apply_cfg(agent, cfg: dict) -> None:
    loc, desc = cfg["loc"], cfg["desc"]
    lex = cfg["lex"]
    Agent._UNIFIED_STRUCT_FIELDS = {}
    Agent._UNIFIED_WEAK_FIELDS = WEAK

    def _score(cls, matched_fields):
        mf = set(matched_fields or [])
        s = 0.0
        if "error_code_clone" in mf:
            s += lex
        if mf & LOCATE:
            s += loc
        if mf & cls._UNIFIED_WEAK_FIELDS:
            s += desc
        return min(1.0, s)

    Agent._unified_structured_score = classmethod(_score)
    agent.gate_structured_threshold = cfg["ts"]
    agent.gate_weak_structure_threshold = cfg["tw"]
    agent.similarity_threshold = cfg["tau"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "param_sweep_v5.jsonl"))
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
    print(f"配置数 {len(CONFIGS)}，读入 {dump.name}", flush=True)
    idx = 0
    t0 = time.time()
    with gzip.open(dump, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            idx += 1
            if args.limit and idx > args.limit:
                break
            cve, role = rec.get("cve"), rec.get("role")
            own = id_by_title.get(cve)
            files = [(fe.get("file") or "", fe.get("gap") or []) for fe in rec.get("files") or []]
            row = {"cve": cve, "role": role, "prod_self": rec.get("prod_self"),
                   "prod_any": rec.get("prod_any"), "cfg": {}}
            for name, cfg in CONFIGS:
                apply_cfg(agent, cfg)
                self_hit = any_hit = False
                for fp, gap in files:
                    findings, _ = regate_variant(agent, gap, sp, fp, _ORIG_GATE)
                    if findings:
                        any_hit = True
                    if role == "kb" and not self_hit and _own_in_findings(findings, own, ci2p):
                        self_hit = True
                row["cfg"][name] = [self_hit, any_hit]
            apply_cfg(agent, dict(BASE))
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            if idx % 10 == 0:
                print(f"[{idx}] {time.time()-t0:.0f}s", flush=True)
    out_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
