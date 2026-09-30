# -*- coding: utf-8 -*-
"""rerun_second_pass_full：重新回填 solution（含 Remove incorrect logic 错误代码）后重门控，校验是否复现 144。"""
import sys
from pathlib import Path
ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
sys.path.insert(0, str(ROOT / "local_libs")); sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import asyncio, gzip, json, sqlite3  # noqa: E402
from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent as Agent  # noqa: E402
from infrastructure.database.sqlite.service import DatabaseService  # noqa: E402
from utils.experiments import param_sweep_fast as PSF  # noqa: E402
from utils.experiments.ablate_views_singlepass import _load_ci_to_pattern, _load_id_by_title, _own_in_findings  # noqa: E402
from utils.experiments.hard_filter_ablation import install_caches  # noqa: E402
from utils.experiments.local_numpy_weaviate import NumpyVectorService  # noqa: E402

DB = ROOT / "infrastructure" / "database" / "mas.db"
KB = ROOT / "reports" / "weaviate_kb_seed2024.jsonl"

# 预加载 mas.db 的 solution（issue_patterns by id）
con = sqlite3.connect(str(DB))
IP_SOL = dict(con.execute("SELECT id, solution FROM issue_patterns").fetchall())
CI_SOL = {r[0]: (r[1], r[2]) for r in con.execute("SELECT id, solution, code_snippet FROM curated_issues").fetchall()}
con.close()

async def main(limit=200):
    agent = Agent()
    agent.vector_service = NumpyVectorService(KB)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    install_caches(agent)
    id_by_title = _load_id_by_title(str(DB))
    ci2p = _load_ci_to_pattern(str(DB))
    sp = await agent.db_service.get_issue_patterns(status="active")
    sp = sp[: getattr(agent, "max_sqlite_patterns", 2000)]

    hit = n = 0
    for sid in range(4):
        with gzip.open(ROOT / f"reports/wsens_dump.jsonl_shard{sid}.gz", "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("role") != "kb":
                    continue
                n += 1
                own = id_by_title.get(rec["cve"])
                ok = False
                for fp, gap in [(fe.get("file") or "", fe.get("gap") or []) for fe in rec.get("files") or []]:
                    ev = PSF.prepare_evidence(agent, gap, sp, fp)
                    for item in ev:
                        for p in item["prepared"]:
                            c = p["c"]
                            sid2 = c.get("sqlite_id")
                            ch = str(c.get("channel") or "").lower()
                            # 关键修复：重新回填 solution（含 Remove incorrect logic 错误代码）
                            if ch == "weaviate" and sid2 in IP_SOL:
                                if "Remove incorrect logic" not in str(c.get("solution") or ""):
                                    c["solution"] = IP_SOL[sid2]
                            elif ch == "curated_issue" and sid2 in CI_SOL:
                                sol, snip = CI_SOL[sid2]
                                if "Remove incorrect logic" not in str(c.get("solution") or ""):
                                    c["solution"] = sol
                                if not c.get("code_snippet"):
                                    c["code_snippet"] = snip
                            # 重新门控（真实 _gate_candidate）
                            agent._gate_candidate(c)
                    rg = [{"candidates": [p["c"] for p in item["prepared"]],
                           "code_chunk": item["code_chunk"],
                           "issue_description": item["issue_description"],
                           "issue_file": item["issue_file"]} for item in ev]
                    fd = agent._derive_new_findings_from_gap_evidence(rg, [], [], "rerun", 1, fp)
                    if _own_in_findings(fd, own, ci2p):
                        ok = True
                if ok:
                    hit += 1
                if n >= limit:
                    break
        if n >= limit:
            break
    print(f"重新回填 solution 后重门控：前 {n} 个 kb 样本 self_hit = {hit}")

asyncio.run(main(200))
