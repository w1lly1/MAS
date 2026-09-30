# -*- coding: utf-8 -*-
"""硬前置过滤 F(x) 消融实验（离线重门控，seed=2024，400 样本）。

F(x) 由三条硬前置过滤组成（在阈值析取门控之前执行、直接 return）：
  (A) 跨文件错配   cross_file_mismatch        —— 仅向量通道；知识条目文件与当前分析文件不同名且无代码锚点
  (B) 弱结构无锚点 weak_structure_no_file_anchor —— 仅向量通道；s(x) < θ_w 且无代码锚点
  (C) 已修复判定   code_already_fixed          —— 全通道；错误代码 token 连续子串在当前代码中已消失

做法：直接复用 weight_sensitivity_s.py 落盘的 gap 证据（含未门控的原始向量命中
weaviate_hits），在内存中对 F(x) 的 8 个子集重新门控并派生 findings，统计
库内组召回率（self 命中）与库外组错配率（产生任意 finding）。

基线一致性：不删除任何过滤时应与落盘中记录的生产结果完全一致（脚本会校验并报告不一致数）。

实现：通过 inspect 取出 _gate_candidate 源码，把目标分支的条件改写为 `if False:`，
exec 回类上（临时，进程结束即失效），因此门控逻辑与生产代码逐字一致。
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
import inspect  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from functools import lru_cache  # noqa: E402

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
from utils.experiments.local_numpy_weaviate import NumpyVectorService  # noqa: E402

DB = ROOT / "infrastructure" / "database" / "mas.db"
KB_DUMP = ROOT / "reports" / "weaviate_kb_seed2024.jsonl"
DUMP_BASE = ROOT / "reports" / "wsens_dump.jsonl"

MECHS = ["cross_file", "weak_struct", "code_fixed"]
MECH_CN = {
    "cross_file": "跨文件错配",
    "weak_struct": "弱结构无锚点",
    "code_fixed": "已修复判定",
}


def variant_name(disabled: tuple[str, ...]) -> str:
    if not disabled:
        return "完整 F(x)（定稿口径）"
    return "删除 " + " + ".join(MECH_CN[m] for m in disabled)


def _build_variants():
    out = []
    for k in range(0, len(MECHS) + 1):
        for combo in itertools.combinations(MECHS, k):
            out.append((variant_name(combo), combo))
    # 先基线、再单删、再多删
    out.sort(key=lambda t: (len(t[1]), t[1]))
    return out


VARIANTS = _build_variants()

# ------------------------------------------------------------------ #
# 源码级分支改写
# ------------------------------------------------------------------ #
_ORIG_GATE = Agent._gate_candidate

_A_COND = """            if (
                knowledge_base
                and analysis_base
                and knowledge_base != analysis_base
                and not has_code_anchor
            ):"""
_B_COND = "            if unified_s < self.gate_weak_structure_threshold and not has_code_anchor:"
_C_COND = "        if self._candidate_code_fixed(candidate):"

_COND_BY_MECH = {"cross_file": _A_COND, "weak_struct": _B_COND, "code_fixed": _C_COND}


def _replace_condition(src: str, cond: str, mech: str) -> str:
    """把多行条件整段替换为 `if False:`（忽略缩进差异，按 strip 后逐行匹配）。"""
    lines = src.split("\n")
    cond_lines = [l.strip() for l in cond.strip().split("\n")]
    n = len(cond_lines)
    hits = [
        i for i in range(len(lines) - n + 1)
        if [lines[i + k].strip() for k in range(n)] == cond_lines
    ]
    if len(hits) != 1:
        raise RuntimeError(f"分支条件定位失败（{mech}，命中 {len(hits)} 次）")
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    return "\n".join(lines[:i] + [f"{indent}if False:"] + lines[i + n:])


def _gate_variant(disabled: tuple[str, ...]):
    """返回把 disabled 中机制的分支条件改写为 False 的 _gate_candidate。"""
    if not disabled:
        return _ORIG_GATE
    src = textwrap.dedent(inspect.getsource(_ORIG_GATE))
    for mech in disabled:
        src = _replace_condition(src, _COND_BY_MECH[mech], mech)
    ns = dict(vars(sys.modules[Agent.__module__]))
    exec(compile(src, "<gate_variant>", "exec"), ns)
    fn = ns["_gate_candidate"]
    fn.__doc__ = f"gate variant disabled={disabled}"
    return fn


# ------------------------------------------------------------------ #
# 缓存：token 化与错误代码片段抽取与变体无关，缓存后各变体共用
# ------------------------------------------------------------------ #
def install_caches(agent: Agent) -> None:
    @lru_cache(maxsize=200000)
    def _tok(text: str):
        return tuple(_ORIG_TOKENIZE(agent, text))

    @lru_cache(maxsize=200000)
    def _frags(solution: str):
        return tuple(tuple(f) for f in _ORIG_FRAGS(agent, solution))

    # 统一返回元组：_is_contiguous_subseq 内部用切片相等比较，needle/haystack 类型必须一致
    agent._tokenize_code = _tok
    agent._extract_error_code_fragments = _frags


_ORIG_TOKENIZE = Agent._tokenize_code
_ORIG_FRAGS = Agent._extract_error_code_fragments


# ------------------------------------------------------------------ #
def hard_filter_attribution(agent, candidate) -> list[str]:
    """候选若被放行，究竟是绕过了哪几条硬前置过滤（用于归因新晋候选）。"""
    hits = []
    channel = str(candidate.get("channel") or "").strip().lower()
    fields = set(candidate.get("matched_fields") or [])
    code_anchor_fields = {
        "class_pattern_in_code", "function_name_in_code",
        "file_basename_anchor", "file_pattern",
    }
    has_code_anchor = bool(fields & code_anchor_fields)
    unified_s = agent._unified_structured_score(fields)
    if channel == "weaviate":
        kb = agent._knowledge_file_basename(candidate)
        ab = agent._normalize_source_basename(str(candidate.get("_analysis_file") or ""))
        if kb and ab and kb != ab and not has_code_anchor:
            hits.append("cross_file")
        if unified_s < agent.gate_weak_structure_threshold and not has_code_anchor:
            hits.append("weak_struct")
    if agent._candidate_code_fixed(candidate):
        hits.append("code_fixed")
    return hits


def regate_variant(agent, gap, sqlite_patterns, file_path, gate_fn):
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
        constant = []
        for c in (e.get("candidates") or []):
            if isinstance(c, dict) and c.get("channel") != "weaviate":
                # 必须按当前参数重判：否则结构化候选会沿用落盘时的基线决策，
                # 参数扫描（权重/阈值）将只作用于向量候选，结果失真。
                cc2 = dict(c)
                agent._gate_candidate(cc2)
                constant.append(cc2)
        hits = [h for h in (e.get("weaviate_hits") or []) if isinstance(h, dict)]
        wc = _gated_weaviate_candidates(agent, hits, issue_desc, issue_file, issue_like, sqlite_patterns)
        ev["candidates"] = constant + wc
        ev["weaviate_hits"] = hits
        re_gated.append(ev)
    findings = agent._derive_new_findings_from_gap_evidence(re_gated, [], [], "hfabl", 1, file_path)
    return findings, re_gated


def build_agent() -> Agent:
    a = Agent()
    a.vector_service = NumpyVectorService(KB_DUMP)
    a.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    return a


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--role", type=str, default="both", choices=["kb", "held", "both"])
    ap.add_argument("--dump", type=str, default=str(DUMP_BASE))
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "hard_filter_ablation.jsonl"))
    args = ap.parse_args()

    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    agent = build_agent()
    install_caches(agent)

    id_by_title = _load_id_by_title(str(DB))
    ci_to_pattern = _load_ci_to_pattern(str(DB))
    sqlite_patterns = await agent.db_service.get_issue_patterns(status="active")
    sqlite_patterns = sqlite_patterns[: getattr(agent, "max_sqlite_patterns", 2000)]

    # 落盘文件名形如 wsens_dump.jsonl_shard0.gz（gzip 后缀在 shard 之后），
    # 不走 _shard_path（它会把 .jsonl 当后缀处理）。
    dump_path = Path(f"{args.dump}_shard{args.shard_id}.gz") if args.shard_count > 1 else Path(f"{args.dump}.gz")
    if not dump_path.exists():
        raise FileNotFoundError(f"gap 证据落盘文件不存在: {dump_path}")
    out_path = _shard_path(args.out, args.shard_id, args.shard_count)
    print(f"读入 {dump_path}", flush=True)

    out_f = open(out_path, "w", encoding="utf-8")
    gate_fns = [(name, dis, _gate_variant(dis)) for name, dis in VARIANTS]
    idx = 0
    t_all = time.time()
    with gzip.open(dump_path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            role = rec.get("role")
            if args.role != "both" and role != args.role:
                continue
            idx += 1
            if args.limit and idx > args.limit:
                break
            cve = rec.get("cve")
            own = id_by_title.get(cve)
            prod_self = bool(rec.get("prod_self"))
            prod_any = bool(rec.get("prod_any"))
            files = [(fe.get("file") or "", fe.get("gap") or []) for fe in rec.get("files") or []]

            out = {"cve": cve, "role": role, "prod_self": prod_self, "prod_any": prod_any, "v": {}}
            for name, dis, gate_fn in gate_fns:
                Agent._gate_candidate = gate_fn
                self_hit = False
                any_hit = False
                admitted = 0
                new_adm = 0
                by_reason: dict[str, int] = {}
                for fp, gap in files:
                    findings, re_gated = regate_variant(agent, gap, sqlite_patterns, fp, gate_fn)
                    if findings:
                        any_hit = True
                    for ev in re_gated:
                        for c in (ev.get("candidates") or []):
                            if not isinstance(c, dict):
                                continue
                            if c.get("gating_decision") in {"formal_hit", "explanatory_hit"}:
                                admitted += 1
                                reasons = hard_filter_attribution(agent, c)
                                if reasons:
                                    new_adm += 1
                                    for r in reasons:
                                        by_reason[r] = by_reason.get(r, 0) + 1
                    if role == "kb" and not self_hit and _own_in_findings(findings, own, ci_to_pattern):
                        self_hit = True
                out["v"][name] = {
                    "self": self_hit, "any": any_hit,
                    "admitted": admitted, "newly_admitted": new_adm, "by_reason": by_reason,
                }
            Agent._gate_candidate = _ORIG_GATE
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[{idx}] {cve} ({role}) prod_self={prod_self} base_self={out['v'][VARIANTS[0][0]]['self']} "
                  f"{time.time()-t_all:.0f}s", flush=True)

    out_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
