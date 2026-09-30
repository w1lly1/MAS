# -*- coding: utf-8 -*-
"""s(x) 公式减重验证：删项/合并项后库内召回与库外错配是否变化（离线重门控，400 样本）。

变体（生产口径，基线一致性已由 hard_filter_ablation 校验为 0/400 不一致）：
  1 基线        0.50·I词元 + 0.20·I文件 + 0.25·I类名 + 0.25·I函数 + 0.10·I描述
  2 删类名函数名  0.50/0.20/0/0/0.10
  3 删描述弱证据  0.50/0.20/0.25/0.25/0
  4 三项        0.50/0.20/0/0/0
  5 两项        0.50/0.20/0/0/0 且描述不计
  6 中证据合并为一项（类名或函数名命中记 0.25，不叠加）
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

VARIANTS = [
    ("基线 0.50/0.20/0.25/0.25/0.10", dict(lex=.50, file=.20, cls=.25, func=.25, desc=.10), False),
    ("删类名+函数名 0.50/0.20/0/0/0.10", dict(lex=.50, file=.20, cls=0.0, func=0.0, desc=.10), False),
    ("删描述弱证据 0.50/0.20/0.25/0.25/0", dict(lex=.50, file=.20, cls=.25, func=.25, desc=0.0), False),
    ("三项 0.50/0.20/0/0/0", dict(lex=.50, file=.20, cls=0.0, func=0.0, desc=0.0), False),
    ("两项 0.50/0.20（描述不计）", dict(lex=.50, file=.20, cls=0.0, func=0.0, desc=None), False),
    ("中证据合并为一项 0.25", dict(lex=.50, file=.20, cls=.25, func=.25, desc=.10), True),
]


def apply_weights(w: dict, merge_mid: bool) -> None:
    Agent._UNIFIED_STRUCT_FIELDS = {
        "error_code_clone": w["lex"],
        "file_basename_anchor": w["file"],
        "basename_match": w["file"],
        "class_pattern_in_code": w["cls"],
        "function_name_in_code": w["func"],
        "function_in_description": w["func"],
    }
    Agent._UNIFIED_WEAK_FIELDS = WEAK
    desc_w = w["desc"]

    if merge_mid:
        mid = max(w["cls"], w["func"])

        def _score(cls, matched_fields):
            mf = set(matched_fields or [])
            s = 0.0
            if "error_code_clone" in mf:
                s += w["lex"]
            if mf & {"file_basename_anchor", "basename_match"}:
                s += w["file"]
            if mf & {"class_pattern_in_code", "function_name_in_code", "function_in_description"}:
                s += mid
            if desc_w is not None and (mf & cls._UNIFIED_WEAK_FIELDS):
                s += desc_w
            return min(1.0, s)
    else:
        def _score(cls, matched_fields):
            mf = set(matched_fields or [])
            s = sum(wt for f, wt in cls._UNIFIED_STRUCT_FIELDS.items() if f in mf)
            if desc_w is not None and (mf & cls._UNIFIED_WEAK_FIELDS):
                s += desc_w
            return min(1.0, s)

    Agent._unified_structured_score = classmethod(_score)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "simplified_formula.jsonl"))
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
    out_path = _shard_path(args.out, args.shard_id, args.shard_count)
    out_f = open(out_path, "w", encoding="utf-8")
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
            for name, w, merge in VARIANTS:
                apply_weights(w, merge)
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
            apply_weights(dict(lex=.50, file=.20, cls=.25, func=.25, desc=.10), False)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            if idx % 20 == 0:
                print(f"[{idx}] {time.time()-t0:.0f}s", flush=True)
    out_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
