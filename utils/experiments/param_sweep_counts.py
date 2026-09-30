# -*- coding: utf-8 -*-
"""以【放行候选个数】为统计量重跑参数扫描（seed=2024）。

统计量（每样本、每配置）：
  adm       : 被放行的候选个数（formal_hit + explanatory_hit），即需交大模型裁决的候选数
  adm_self  : 其中命中该样本自身知识库条目的候选数（库内组有效）
  adm_other : 其余放行候选数
汇总时按库内组/库外组分别求和。
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
    _shard_path,
)
from utils.experiments.hard_filter_ablation import install_caches  # noqa: E402
from utils.experiments.local_numpy_weaviate import NumpyVectorService  # noqa: E402
from utils.experiments.param_sweep_fast import CONFIGS, BASE, gate, prepare_evidence  # noqa: E402

DB = ROOT / "infrastructure" / "database" / "mas.db"
KB_DUMP = ROOT / "reports" / "weaviate_kb_seed2024.jsonl"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "param_counts.jsonl"))
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
    out = open(_shard_path(args.out, args.shard_id, args.shard_count), "w", encoding="utf-8")
    print(f"配置 {len(CONFIGS)} 个，读入 {dump.name}", flush=True)
    n = 0
    t0 = time.time()
    with gzip.open(dump, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            role, cve = rec.get("role"), rec.get("cve")
            own = id_by_title.get(cve)
            prepared = [(fe.get("file") or "",
                         prepare_evidence(agent, fe.get("gap") or [], sp, fe.get("file") or ""))
                        for fe in rec.get("files") or []]
            row = {"cve": cve, "role": role, "cfg": {}}
            for name, cfg in CONFIGS:
                adm = adm_self = adm_other = 0
                for fp, ev in prepared:
                    for e in gate(agent, ev, cfg):
                        for c in e["candidates"]:
                            if c.get("gating_decision") not in ("formal_hit", "explanatory_hit"):
                                continue
                            adm += 1
                            sid = c.get("sqlite_id")
                            if str(c.get("channel") or "").lower() == "curated_issue":
                                is_self = ci2p.get(sid) == own
                            else:
                                is_self = own is not None and sid == own
                            if is_self:
                                adm_self += 1
                            else:
                                adm_other += 1
                row["cfg"][name] = [adm, adm_self, adm_other]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if n % 25 == 0:
                print(f"[{n}] {time.time()-t0:.0f}s", flush=True)
    out.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
