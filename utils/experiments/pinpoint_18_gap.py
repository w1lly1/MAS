# -*- coding: utf-8 -*-
"""pinpoint_18_gap：20 个差异样本的候选是否存在、solution 是否回填、code_already_fixed 判定。"""
import sys
from pathlib import Path
ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
sys.path.insert(0, str(ROOT / "local_libs")); sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import asyncio, gzip, json  # noqa: E402
from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent as Agent  # noqa: E402
from infrastructure.database.sqlite.service import DatabaseService  # noqa: E402
from utils.experiments import param_sweep_fast as PSF  # noqa: E402
from utils.experiments.ablate_views_singlepass import _load_ci_to_pattern, _load_id_by_title, _own_in_findings  # noqa: E402
from utils.experiments.hard_filter_ablation import install_caches  # noqa: E402
from utils.experiments.local_numpy_weaviate import NumpyVectorService  # noqa: E402

DB = ROOT / "infrastructure" / "database" / "mas.db"
KB = ROOT / "reports" / "weaviate_kb_seed2024.jsonl"
TARGET = ["CVE-2011-2804","CVE-2011-4913","CVE-2012-6647","CVE-2012-6701","CVE-2013-1943",
          "CVE-2013-6638","CVE-2014-2038","CVE-2014-3479","CVE-2015-0275","CVE-2015-1573",
          "CVE-2015-3138","CVE-2015-6031","CVE-2015-8863","CVE-2017-10661","CVE-2017-11665",
          "CVE-2017-12190","CVE-2017-12904","CVE-2019-15148","CVE-2019-15165","CVE-2019-5764"]

async def main():
    agent = Agent()
    agent.vector_service = NumpyVectorService(KB)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    install_caches(agent)
    id_by_title = _load_id_by_title(str(DB))
    ci2p = _load_ci_to_pattern(str(DB))
    sp = await agent.db_service.get_issue_patterns(status="active")
    sp = sp[: getattr(agent, "max_sqlite_patterns", 2000)]

    for sid in range(4):
        with gzip.open(ROOT / f"reports/wsens_dump.jsonl_shard{sid}.gz", "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["cve"] not in TARGET:
                    continue
                own = id_by_title.get(rec["cve"])
                print(f"\n===== {rec['cve']} (own={own}) prod_self={rec.get('prod_self')} =====")
                for fe in rec.get("files") or []:
                    for e in fe.get("gap") or []:
                        for c in (e.get("candidates") or []):
                            if c.get("sqlite_id") == own or c.get("channel") == "curated_issue":
                                sid2 = c.get("sqlite_id")
                                ch = c.get("channel")
                                sol = c.get("solution") or ""
                                has_err = "错误逻辑" in sol or "错误代码" in sol
                                fixed = agent._candidate_code_fixed(c)
                                mark = "★SELF" if (ci2p.get(sid2) == own or sid2 == own) else ""
                                print(f"  [{ch}] sqlite_id={sid2} solution_len={len(sol)} 含错误逻辑={has_err} "
                                      f"code_fixed={fixed} {mark}")
                for fe in rec.get("files") or []:
                    for e in fe.get("gap") or []:
                        for h in (e.get("weaviate_hits") or []):
                            if h.get("sqlite_id") == own:
                                print(f"  [weaviate_hit] sqlite_id={own} layer={h.get('vector_layer')} "
                                      f"solution_len={len(h.get('solution') or '')}")
                break

asyncio.run(main())
