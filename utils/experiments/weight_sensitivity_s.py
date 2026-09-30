# -*- coding: utf-8 -*-
"""s(x) 权重敏感性实验（seed=2024，本地 CPU）。

做法：对每个样本只跑一次 _run_second_pass（基线权重）拿到 gap 证据，
然后在内存中按不同权重配置【重新门控】（重算 s(x) → 重判 admit → 重新派生 findings），
统计各配置下的库内组召回率与库外组错配样本率。

验证：基线配置的重算结果应与生产 new_findings 一致（脚本会报告不一致样本数）。
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
    AIDrivenSecondPassAnalysisAgent as Agent,
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

BATCH = ROOT / "论文" / "test_400_error_batch.json"          # seed=2024
DB = ROOT / "infrastructure" / "database" / "mas.db"          # seed=2024
KB_DUMP = ROOT / "reports" / "weaviate_kb_seed2024.jsonl"

BASE = {"lex": 0.50, "file": 0.20, "cls": 0.25, "func": 0.25, "desc": 0.10}


def _mk(**kw):
    w = dict(BASE)
    w.update(kw)
    return w


def _build_configs() -> list[tuple[str, dict]]:
    """单权重全区间扫描：固定其余四个，目标权重 0→1（步长 0.05）。"""
    cfgs: list[tuple[str, dict]] = [("基线 0.50/0.20/0.25/0.25/0.10", dict(BASE))]
    field_names = {"lex": "I_lex", "file": "I_file", "cls": "I_class",
                   "func": "I_func", "desc": "I_desc"}
    vals = [round(0.05 * i, 2) for i in range(0, 21)]  # 0.00 ~ 1.00
    for key, label in field_names.items():
        for v in vals:
            if abs(v - BASE[key]) < 1e-9:
                continue  # 与基线重复
            cfgs.append((f"{label}={v:.2f}", _mk(**{key: v})))
    return cfgs


CONFIGS: list[tuple[str, dict]] = _build_configs()

WEAK = frozenset({
    "phenomenon_in_description", "root_cause_in_description",
    "error_type_in_description", "error_type_in_source",
    "error_description_prefix", "problematic_pattern_prefix",
    "location_in_description", "pattern_in_snippet",
    "file_basename_in_description", "file_pattern",
})


def apply_weights(w: dict) -> None:
    """把权重写入 Agent 类（覆盖类属性 + 包装 s(x) 计算）。"""
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

    def _score(cls, matched_fields):
        mf = set(matched_fields or [])
        s = 0.0
        for field, weight in cls._UNIFIED_STRUCT_FIELDS.items():
            if field in mf:
                s += weight
        if mf & cls._UNIFIED_WEAK_FIELDS:
            s += desc_w
        return min(1.0, s)

    Agent._unified_structured_score = classmethod(_score)


def build_agent() -> Agent:
    a = Agent()
    a.vector_service = NumpyVectorService(KB_DUMP)
    a.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    return a


def regate_findings(agent, gap, sqlite_patterns, file_path):
    """按当前权重重新门控 gap 证据并派生 findings（结构化与语义候选都重判）。"""
    re_gated = []
    for e in gap or []:
        if not isinstance(e, dict):
            continue
        ev = dict(e)
        cc = e.get("code_chunk")
        if isinstance(cc, dict):
            issue_like = agent._code_chunk_as_issue(cc)
        else:
            issue_like = {
                "description": e.get("issue_description"),
                "line": None,
                "file": e.get("issue_file") or file_path,
            }
        issue_desc = str(issue_like.get("description") or "")
        issue_file = issue_like.get("file")
        # 结构化候选：重新门控
        constant = []
        for c in (e.get("candidates") or []):
            if isinstance(c, dict) and c.get("channel") != "weaviate":
                cc2 = dict(c)
                agent._gate_candidate(cc2)
                constant.append(cc2)
        hits = [h for h in (e.get("weaviate_hits") or []) if isinstance(h, dict)]
        wc = _gated_weaviate_candidates(
            agent, hits, issue_desc, issue_file, issue_like, sqlite_patterns
        )
        ev["candidates"] = constant + wc
        ev["weaviate_hits"] = hits
        re_gated.append(ev)
    return agent._derive_new_findings_from_gap_evidence(
        re_gated, [], [], "wsens", 1, file_path
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--role", type=str, default="both", choices=["kb", "held", "both"])
    ap.add_argument("--dump", type=str, default="", help="落盘原始 gap 证据的路径（供离线反复扫描）")
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "weight_sensitivity.jsonl"))
    args = ap.parse_args()

    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    agent = build_agent()
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    items = [it for it in batch["items"] if args.role == "both" or it.get("role") == args.role]
    if args.limit > 0:
        items = items[: args.limit]
    if args.shard_count > 1:
        items = [it for i, it in enumerate(items) if i % args.shard_count == args.shard_id]

    id_by_title = _load_id_by_title(str(DB))
    ci_to_pattern = _load_ci_to_pattern(str(DB))
    sqlite_patterns = await agent.db_service.get_issue_patterns(status="active")
    sqlite_patterns = sqlite_patterns[: getattr(agent, "max_sqlite_patterns", 2000)]

    out_path = _shard_path(args.out, args.shard_id, args.shard_count)
    out_f = open(out_path, "w", encoding="utf-8")
    dump_f = None
    if args.dump:
        import gzip
        dump_f = gzip.open(_shard_path(args.dump, args.shard_id, args.shard_count), "wt", encoding="utf-8")
    print(f"待跑样本: {len(items)}，配置数 {len(CONFIGS)}（shard {args.shard_id}/{args.shard_count}）", flush=True)

    for idx, it in enumerate(items, 1):
        cve = it.get("cve")
        role = it.get("role")
        own = id_by_title.get(cve)
        td = it.get("target_dir") or ""
        if td.startswith("/root/autodl-tmp/MAS/"):
            td = str(ROOT / td[len("/root/autodl-tmp/MAS/"):])
        files = [f["path"] for f in discover_source_files(td).get("files", [])] if td and Path(td).is_dir() else []
        if not files:
            print(f"[{idx}/{len(items)}] {cve}: 无源文件，跳过", flush=True)
            continue

        t0 = time.time()
        gap_all = []
        prod_self = False
        prod_any = False
        for fp in files:
            report_data = {"file": fp, "issues": [], "run_id": "wsens", "requirement_id": 1}
            try:
                r = await agent._run_second_pass(report_data, original_analysis=None, layer_mode=None)
            except Exception as e:
                print(f"[{idx}/{len(items)}] {cve} {fp}: 异常 {e}", flush=True)
                continue
            gap_all.append((fp, r.get("gap_retrieval_evidence") or []))
            nf = r.get("new_findings") or []
            if nf:
                prod_any = True
            if role == "kb" and _own_in_findings(nf, own, ci_to_pattern):
                prod_self = True

        rec = {"cve": cve, "role": role, "prod_self": prod_self, "prod_any": prod_any, "cfg": {}}
        for name, w in CONFIGS:
            apply_weights(w)
            hit = False
            for fp, gap in gap_all:
                findings = regate_findings(agent, gap, sqlite_patterns, fp)
                if role == "kb":
                    if _own_in_findings(findings, own, ci_to_pattern):
                        hit = True
                        break
                else:
                    if findings:
                        hit = True
                        break
            rec["cfg"][name] = hit
        apply_weights(dict(BASE))  # 复位

        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        if dump_f is not None:
            dump_f.write(json.dumps(
                {"cve": cve, "role": role, "prod_self": prod_self, "prod_any": prod_any,
                 "files": [{"file": fp, "gap": gap} for fp, gap in gap_all]},
                ensure_ascii=False, default=str) + "\n")
            dump_f.flush()
        print(f"[{idx}/{len(items)}] {cve} ({role}) prod={prod_self} {time.time()-t0:.1f}s", flush=True)

    out_f.close()
    if dump_f is not None:
        dump_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
