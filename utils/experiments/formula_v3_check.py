# -*- coding: utf-8 -*-
"""用户提出的三项公式验证：
    s(x) = 0.5·I词元 + 0.4·I定位（同名文件/类名/函数名任一） + 0.1·I描述

与现状 5 项公式 0.5/0.2/0.25/0.25/0.1 做逐 CVE 对照（离线重门控，400 样本，生产口径）。
唯一预期差异：无词元但命中 2 项以上定位证据的候选（如 file+cls+func），
现状 s=0.2+0.25+0.25=0.7 ≥ 0.65 判 formal_hit；新式合并为单项 0.4 < 0.65，改由语义通道判定。
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


def apply_baseline() -> None:
    Agent._UNIFIED_STRUCT_FIELDS = {
        "error_code_clone": 0.5,
        "file_basename_anchor": 0.2, "basename_match": 0.2,
        "class_pattern_in_code": 0.25,
        "function_name_in_code": 0.25, "function_in_description": 0.25,
    }
    Agent._UNIFIED_WEAK_FIELDS = WEAK

    def _score(cls, matched_fields):
        mf = set(matched_fields or [])
        s = sum(w for f, w in cls._UNIFIED_STRUCT_FIELDS.items() if f in mf)
        if mf & cls._UNIFIED_WEAK_FIELDS:
            s += 0.1
        return min(1.0, s)

    Agent._unified_structured_score = classmethod(_score)


def apply_proposed(desc_w: float = 0.1) -> None:
    """0.5·I词元 + 0.4·I定位 + desc_w·I描述。"""
    Agent._UNIFIED_STRUCT_FIELDS = {}   # 不再逐项累加
    Agent._UNIFIED_WEAK_FIELDS = WEAK

    def _score(cls, matched_fields):
        mf = set(matched_fields or [])
        s = 0.0
        if "error_code_clone" in mf:
            s += 0.5
        if mf & LOCATE:
            s += 0.4
        if desc_w and (mf & cls._UNIFIED_WEAK_FIELDS):
            s += desc_w
        return min(1.0, s)

    Agent._unified_structured_score = classmethod(_score)


VARIANTS = [
    ("现状 5 项 0.5/0.2/0.25/0.25/0.1", apply_baseline),
    ("提案 3 项 0.5/0.4/0.1", lambda: apply_proposed(0.1)),
    ("提案去描述 0.5/0.4", lambda: apply_proposed(0.0)),
]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "formula_v3.jsonl"))
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
    idx = 0
    t0 = time.time()
    with gzip.open(dump, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            idx += 1
            cve, role = rec.get("cve"), rec.get("role")
            own = id_by_title.get(cve)
            files = [(fe.get("file") or "", fe.get("gap") or []) for fe in rec.get("files") or []]
            row = {"cve": cve, "role": role, "prod_self": rec.get("prod_self"),
                   "prod_any": rec.get("prod_any"), "v": {}}
            for name, applier in VARIANTS:
                applier()
                self_hit = any_hit = False
                nfind = 0
                for fp, gap in files:
                    findings, _ = regate_variant(agent, gap, sp, fp, _ORIG_GATE)
                    nfind += len(findings)
                    if findings:
                        any_hit = True
                    if role == "kb" and not self_hit and _own_in_findings(findings, own, ci2p):
                        self_hit = True
                row["v"][name] = {"self": self_hit, "any": any_hit, "n_findings": nfind}
            apply_baseline()
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            if idx % 25 == 0:
                print(f"[{idx}] {time.time()-t0:.0f}s", flush=True)
    out_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
