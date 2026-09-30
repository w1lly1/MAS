# -*- coding: utf-8 -*-
"""第 4.8 节参数扫描（快速版）：参数无关量预计算一次，逐配置只重算 s(x) 与门控分支。

与 param_sweep_v5.py 等价但快约两个数量级（结构化候选每样本约 4400 条，但其
matched_fields 组合仅约 7 种，故把候选构建、已修复/跨文件判定、各分量都提到配置循环之外）。

门控分支与核心代码逐行一致；基线配置必须与生产结果逐样本一致（脚本会打印校验）。
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
from utils.experiments.hard_filter_ablation import install_caches  # noqa: E402
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
CODE_ANCHORS = {"class_pattern_in_code", "function_name_in_code",
                "file_basename_anchor", "file_pattern"}
GENERIC = {"threading", "insert", "update", "delete"}

BASE = {"lex": 0.5, "loc": 0.4, "desc": 0.1, "ts": 0.65, "tw": 0.2, "tau": 0.65}
LABEL = {"lex": "θ_lex", "loc": "θ_loc", "desc": "θ_desc", "tw": "θ_w",
         "tau": "τ", "ts": "θ_s"}
BASELINE_NAME = "A定稿"


def build_configs():
    cfgs = [(BASELINE_NAME, dict(BASE))]
    vals = [round(0.05 * i, 2) for i in range(21)]
    for key in ("lex", "loc", "desc", "tw"):
        for v in vals:
            if abs(v - BASE[key]) < 1e-9:
                continue
            c = dict(BASE)
            c[key] = v
            cfgs.append((f"{LABEL[key]}={v:.2f}", c))
    for v in [round(0.50 + 0.05 * i, 2) for i in range(8)]:
        if abs(v - BASE["tau"]) < 1e-9:
            continue
        c = dict(BASE)
        c["tau"] = v
        cfgs.append((f"τ={v:.2f}", c))
    for v in [round(0.30 + 0.05 * i, 2) for i in range(15)]:
        if abs(v - BASE["ts"]) < 1e-9:
            continue
        c = dict(BASE)
        c["ts"] = v
        cfgs.append((f"θ_s={v:.2f}", c))
    return cfgs


CONFIGS = build_configs()


def prepare_evidence(agent, gap, sp, file_path):
    """构建候选并预计算参数无关量。返回可直接用于逐配置门控的 evidence 列表。"""
    out = []
    for e in gap or []:
        if not isinstance(e, dict):
            continue
        cc = e.get("code_chunk")
        if isinstance(cc, dict):
            il = agent._code_chunk_as_issue(cc)
        else:
            il = {"description": e.get("issue_description"), "line": None,
                  "file": e.get("issue_file") or file_path}
        desc = str(il.get("description") or "")
        ifile = il.get("file")
        hits = [h for h in (e.get("weaviate_hits") or []) if isinstance(h, dict)]
        layer_cands = [agent._build_candidate_from_weaviate(h, desc, ifile, il) for h in hits]
        wc = []
        for cand in agent._merge_weaviate_candidates_by_sqlite_id(layer_cands):
            agent._backfill_weaviate_candidate_solution(
                cand, weaviate_hits=hits, sqlite_patterns=sp)
            agent._apply_file_function_anchors(cand, il, ifile)
            wc.append(cand)
        const = [dict(c) for c in (e.get("candidates") or [])
                 if isinstance(c, dict) and c.get("channel") != "weaviate"]
        prepared = []
        for c in const + wc:
            mf = frozenset(c.get("matched_fields") or [])
            ch = str(c.get("channel") or "").strip().lower()
            kb_base = agent._knowledge_file_basename(c) if ch == "weaviate" else ""
            ab = agent._normalize_source_basename(str(c.get("_analysis_file") or ""))
            prepared.append({
                "c": c, "mf": mf, "ch": ch,
                "anchor": bool(mf & CODE_ANCHORS),
                "cross": bool(kb_base and ab and kb_base != ab),
                "fixed": agent._candidate_code_fixed(c),
                "structured": float(c.get("structured_score") or 0.0),
                "semantic": float(c.get("semantic_score") or 0.0),
                "context": float(c.get("context_score") or 0.0),
                "anchor_s": float(c.get("anchor_score") or 0.0),
                "etype": str(c.get("error_type") or "").strip().lower(),
                "layer": str(c.get("vector_layer") or "").strip().lower(),
            })
        out.append({"code_chunk": cc, "issue_description": e.get("issue_description"),
                    "issue_file": e.get("issue_file"), "prepared": prepared})
    return out


def gate(agent, ev_list, cfg):
    """按配置重算门控，返回可送入派生流程的 evidence 列表。"""
    ts, tw, tau = cfg["ts"], cfg["tw"], cfg["tau"]
    theta_a = agent.gate_anchor_threshold
    req_sim = bool(getattr(agent, "layer_bonus_require_similarity_gate", True))
    lb_map = getattr(agent, "layer_bonus_map", {})
    re_gated = []
    for item in ev_list:
        cands = []
        for p in item["prepared"]:
            c, mf = p["c"], p["mf"]
            s = 0.0
            if "error_code_clone" in mf:
                s += cfg["lex"]
            if mf & LOCATE:
                s += cfg["loc"]
            if mf & WEAK:
                s += cfg["desc"]
            s = min(1.0, s)
            anchor_ok = p["anchor"]
            dec = rea = None
            if p["ch"] == "weaviate":
                if p["cross"] and not anchor_ok:
                    dec, rea = "discarded_hit", "cross_file_mismatch"
                elif s < tw and not anchor_ok:
                    if p["semantic"] >= tau:
                        dec, rea = "low_confidence_hit", "weak_structure_no_file_anchor"
                    else:
                        dec, rea = "discarded_hit", "low_confidence_or_generic"
            if dec is None:
                if p["fixed"]:
                    dec, rea = "discarded_hit", "code_already_fixed"
                elif s >= ts:
                    dec, rea = "formal_hit", ""
                elif p["semantic"] >= tau and p["anchor_s"] >= theta_a and s >= tw:
                    dec, rea = "explanatory_hit", ""
                elif p["semantic"] >= tau and s < tw:
                    dec, rea = "low_confidence_hit", "weak_structure_high_semantic"
                else:
                    dec, rea = "discarded_hit", "low_confidence_or_generic"
            c["gating_decision"] = dec
            c["rejection_reason"] = rea
            # 展示/排序用分项（派生流程以 structured_score、total_score 为排序键）
            ab_bonus = 0.0
            if "pattern_in_snippet" in mf:
                ab_bonus += 0.12
            if "file_pattern" in mf:
                ab_bonus += 0.08
            if "function_in_description" in mf:
                ab_bonus += 0.05
            if "location_in_description" in mf:
                ab_bonus += 0.03
            ab_bonus = min(0.2, ab_bonus)
            layer_bonus = float(lb_map.get(p["layer"], 0.0)) if p["layer"] else 0.0
            if req_sim and p["semantic"] < tau:
                layer_bonus = 0.0
            penalty = 0.0
            if p["etype"] in GENERIC and p["structured"] < 0.4:
                penalty += 0.2
            if p["semantic"] < tau and p["structured"] < 0.4:
                penalty += 0.1
            if p["anchor_s"] < 0.2 and p["structured"] < 0.4:
                penalty += 0.05
            w = (0.55, 0.0, 0.15, 0.20) if p["ch"] == "curated_issue" else (0.50, 0.35, 0.10, 0.05)
            total = (p["structured"] * w[0] + p["semantic"] * w[1] + p["context"] * w[2]
                     + p["anchor_s"] * w[3] + ab_bonus + layer_bonus - penalty)
            c["total_score"] = round(max(0.0, total), 4)
            c["penalty_score"] = round(penalty, 4)
            c["unified_structured_score"] = round(s, 4)
            cands.append(c)
        re_gated.append({"candidates": cands, "code_chunk": item["code_chunk"],
                         "issue_description": item["issue_description"],
                         "issue_file": item["issue_file"]})
    return re_gated


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "param_sweep_fast.jsonl"))
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
            prepared = [(fp, prepare_evidence(agent, gap, sp, fp)) for fp, gap in files]
            row = {"cve": cve, "role": role, "prod_self": rec.get("prod_self"),
                   "prod_any": rec.get("prod_any"), "cfg": {}}
            for name, cfg in CONFIGS:
                self_hit = any_hit = False
                for fp, ev_list in prepared:
                    findings = agent._derive_new_findings_from_gap_evidence(
                        gate(agent, ev_list, cfg), [], [], "psweep", 1, fp)
                    if findings:
                        any_hit = True
                    if role == "kb" and not self_hit and _own_in_findings(findings, own, ci2p):
                        self_hit = True
                row["cfg"][name] = [self_hit, any_hit]
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            if idx % 20 == 0:
                print(f"[{idx}] {time.time()-t0:.0f}s", flush=True)
    out_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
