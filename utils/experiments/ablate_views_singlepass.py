# -*- coding: utf-8 -*-
"""离线视图消融（单遍检索 + 内存分层重算，纯 CPU，无需 GPU）。

动机
----
二次校验是非生成式的（enable_llm_second_pass=false），整条链路：
    CodeBERT 嵌入(CPU) + Weaviate 向量检索(CPU) + SQLite/curated 结构化匹配(CPU)
都不需要 GPU。之前的 ablate_views_offline.py 之所以慢，是因为它对每个 CVE
跑了 5 遍完整的 _run_second_pass（all + 4 个单层），其中结构化匹配
（200 issue_patterns + 327 curated_issues / 每代码分片）与向量层完全无关，
等于白跑 5 遍。

策略
----
1. 只在 layer_mode=None（4 层全开）下，对每个 CVE 跑 **1 遍** 完整二次校验，
   得到每个代码分片的 gap_retrieval_evidence；其中 weaviate_hits 逐条自带
   vector_layer（semantic/code_pattern/solution/full）。
2. 在内存里按 5 种层子集过滤 weaviate_hits，复用 agent 原有的
   候选构建 → 跨层融合 → 回填 → 文件/函数锚定 → 门控 →
   _derive_new_findings_from_gap_evidence，判断 own_id 是否被采纳。
3. 结果增量写盘（--resume 断点续跑）；原始 evidence 落盘供离线复核；
   --regate-only 可跳过检索、纯从 dump 重算（不连 Weaviate / 不连 DB）。

运行（无卡 CPU 机器）：
    export HF_HOME=/root/autodl-tmp/hf-cache
    python -u utils/experiments/ablate_views_singlepass.py --limit 200
断点续跑：
    python -u utils/experiments/ablate_views_singlepass.py --limit 200 --resume
仅离线重算（读 dump，不检索）：
    python -u utils/experiments/ablate_views_singlepass.py --regate-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.agents.ai_driven_second_pass_analysis_agent import (  # noqa: E402
    AIDrivenSecondPassAnalysisAgent,
)
from utils.scan_discovery import discover_source_files  # noqa: E402

# 128 核无卡机器上，PyTorch 多线程会让 distilbert 前向传播急剧变慢（线程越多越慢），
# 实测 threads=1 最快（~159ms/次 vs 8线程~1.1s/次、16线程~18s/次）。显式压到 1 线程。
import os as _os  # noqa: E402

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch as _torch  # noqa: F401

    _torch.set_num_threads(1)
    _torch.set_num_interop_threads(1)
except Exception:
    pass

BATCH_CANDIDATES = [
    ROOT / "utils" / "experiments" / "test_400_error_batch.json",
    ROOT / "论文" / "test_400_error_batch.json",
]
DB_DEFAULT = ROOT / "infrastructure" / "database" / "mas.db"
REPORTS = ROOT / "reports"

# 层子集 -> 实际保留的 vector_layer（None = 全部四层）
MODES = {
    "all(4层)": None,
    "full": {"full"},
    "semantic": {"semantic"},
    "code_pattern": {"code_pattern"},
    "solution": {"solution"},
}


def _own_in_findings(findings, own_id, ci_to_pattern=None) -> bool:
    """判断 own_id（issue_patterns.id）是否作为 self 命中出现在 new_findings。

    - weaviate / sqlite 通道：evidence.sqlite_id 即 issue_patterns.id，直接比较。
    - curated_issue 通道：evidence.sqlite_id 是 curated_issues.id，需经
      curated_issues.pattern_id 映射回 issue_patterns.id（与 evaluate_400.py 口径一致）。
    """
    for nf in findings or []:
        ev = nf.get("evidence") if isinstance(nf, dict) else {}
        sid = ev.get("sqlite_id")
        if sid is None:
            continue
        channel = str(ev.get("channel") or "").strip().lower()
        if channel == "curated_issue":
            if ci_to_pattern and ci_to_pattern.get(sid) == own_id:
                return True
        elif sid == own_id:
            return True
    return False


def _gated_weaviate_candidates(agent, weaviate_hits, issue_desc, issue_file, issue, sqlite_patterns):
    """与 _collect_evidence 内联路径完全一致：构建→融合→回填→锚定→门控。"""
    layer_candidates = [
        agent._build_candidate_from_weaviate(h, issue_desc, issue_file, issue)
        for h in weaviate_hits
        if isinstance(h, dict)
    ]
    gated = []
    for candidate in agent._merge_weaviate_candidates_by_sqlite_id(layer_candidates):
        agent._backfill_weaviate_candidate_solution(
            candidate, weaviate_hits=weaviate_hits, sqlite_patterns=sqlite_patterns
        )
        agent._apply_file_function_anchors(candidate, issue, issue_file)
        agent._gate_candidate(candidate)
        gated.append(candidate)
    return gated


def _re_gate_evidence(agent, evidence, sqlite_patterns, layer_subset):
    """返回该分片在某层子集下的新 evidence（candidates 按层过滤重算）。"""
    ev = dict(evidence)
    code_chunk = evidence.get("code_chunk")
    if isinstance(code_chunk, dict):
        issue_like = agent._code_chunk_as_issue(code_chunk)
    else:
        issue_like = {
            "description": evidence.get("issue_description"),
            "line": None,
            "file": evidence.get("issue_file"),
        }
    issue_desc = str(issue_like.get("description") or "")
    issue_file = issue_like.get("file")

    # 结构化命中（sqlite + curated）与向量层无关，恒定保留
    constant = [
        c for c in (evidence.get("candidates") or [])
        if isinstance(c, dict) and c.get("channel") != "weaviate"
    ]
    if layer_subset is None:
        hits = [h for h in (evidence.get("weaviate_hits") or []) if isinstance(h, dict)]
    else:
        hits = [
            h for h in (evidence.get("weaviate_hits") or [])
            if isinstance(h, dict)
            and str(h.get("vector_layer") or "").strip().lower() in layer_subset
        ]
    weaviate_cands = _gated_weaviate_candidates(
        agent, hits, issue_desc, issue_file, issue_like, sqlite_patterns
    )
    ev["candidates"] = constant + weaviate_cands
    ev["weaviate_hits"] = hits
    return ev


def _recall_for_subset(agent, gap_evidence, sqlite_patterns, layer_subset, own_id, file_path, ci_to_pattern=None) -> bool:
    re_gated = [
        _re_gate_evidence(agent, e, sqlite_patterns, layer_subset)
        for e in (gap_evidence or [])
        if isinstance(e, dict)
    ]
    findings = agent._derive_new_findings_from_gap_evidence(
        re_gated, [], [], "ablate", 1, file_path
    )
    return _own_in_findings(findings, own_id, ci_to_pattern)


def _load_id_by_title(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT id, title FROM issue_patterns")
    mapping = {t: i for i, t in cur.fetchall()}
    con.close()
    return mapping


def _load_ci_to_pattern(db_path: str):
    """curated_issues.id -> curated_issues.pattern_id（用于 curated 通道的 self 判定）。"""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT id, pattern_id FROM curated_issues")
    mapping = {int(i): p for i, p in cur.fetchall()}
    con.close()
    return mapping


def _remap_target_dir(target_dir: str) -> str:
    """batch 里 target_dir 若是 Windows 路径（E:/MyOwn/ProgramStudy/MAS/...），
    自动把该前缀重映射为当前 ROOT（远程 /root/autodl-tmp/MAS/...）。
    已是 Linux 路径时原样返回。"""
    if not target_dir:
        return target_dir
    if "E:/" in target_dir or "E:\\" in target_dir:
        idx = target_dir.find("MAS")
        if idx >= 0:
            return str(ROOT) + target_dir[idx + 3:]
    return target_dir


def _find_batch(arg_batch: str) -> Path:
    if arg_batch:
        p = Path(arg_batch)
    else:
        p = next((c for c in BATCH_CANDIDATES if c.exists()), BATCH_CANDIDATES[0])
    return p


def _shard_path(p: str, shard_id: int, shard_count: int) -> Path:
    if shard_count <= 1:
        return Path(p)
    p = Path(p)
    return p.with_name(f"{p.stem}_shard{shard_id}{p.suffix}")


def _merge_summary(results_dir: Path, base_name: str) -> None:
    """合并所有 shard 结果文件，输出总召回。"""
    results = []
    seen = set()
    for f in sorted(results_dir.glob(f"{base_name}_shard*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cve = rec.get("cve")
            if cve in seen:
                continue
            seen.add(cve)
            results.append(rec)

    n = len(results)
    print(f"\n==== 合并汇总（{n} 个 CVE，{len(list(results_dir.glob(base_name + '_shard*.jsonl')))} 个 shard 文件） ====")
    for m in MODES:
        hits = sum(1 for r in results if r.get(m))
        pct = f"{hits / n:.1%}" if n else "-"
        print(f"{m}: 召回 {hits}/{n} = {pct}")


def _summary_from_results(results_path: Path) -> None:
    results = []
    seen = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        cve = rec.get("cve")
        if cve in seen:
            continue
        seen.add(cve)
        results.append(rec)

    n = len(results)
    print(f"\n==== 汇总（{n} 个 CVE） ====")
    for m in MODES:
        hits = sum(1 for r in results if r.get(m))
        pct = f"{hits / n:.1%}" if n else "-"
        print(f"{m}: 召回 {hits}/{n} = {pct}")


async def _run_retrieval(args) -> None:
    batch_path = _find_batch(args.batch)
    if not batch_path.exists():
        print(f"[fatal] batch 不存在: {batch_path}", flush=True)
        sys.exit(1)

    out_path = _shard_path(args.out, args.shard_id, args.shard_count)
    dump_path = _shard_path(args.dump, args.shard_id, args.shard_count)
    sqlite_sidecar = _shard_path(args.sqlite_sidecar, args.shard_id, args.shard_count)

    agent = AIDrivenSecondPassAnalysisAgent()
    await agent.initialize()

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    kb_items = [it for it in batch["items"] if it.get("role") == "kb"][: args.limit]
    if args.shard_count > 1:
        kb_items = [it for i, it in enumerate(kb_items) if i % args.shard_count == args.shard_id]
    id_by_title = _load_id_by_title(args.db)
    ci_to_pattern = _load_ci_to_pattern(args.db)

    sqlite_patterns = await agent.db_service.get_issue_patterns(status="active")
    sqlite_patterns = sqlite_patterns[: getattr(agent, "max_sqlite_patterns", 2000)]

    # 存侧车（供 --regate-only 复用，无需再连 DB）
    sqlite_sidecar.write_text(
        json.dumps(
            {"sqlite_patterns": sqlite_patterns, "ci_to_pattern": ci_to_pattern},
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    done_cves = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done_cves.add(json.loads(line)["cve"])
            except Exception:
                pass
        if done_cves:
            print(f"[resume] 已跳过 {len(done_cves)} 个 CVE", flush=True)

    mode = "a" if args.resume else "w"
    out_f = open(out_path, mode, encoding="utf-8")
    dump_f = open(dump_path, mode, encoding="utf-8")

    print(f"kb 样本数: {len(kb_items)}（--limit {args.limit}）", flush=True)
    totals = {m: 0 for m in MODES}
    done = 0
    no_own = 0
    no_files = 0
    for idx, it in enumerate(kb_items, 1):
        cve = it.get("cve")
        if cve in done_cves:
            continue
        own = id_by_title.get(cve)
        if own is None:
            no_own += 1
            print(f"[{idx}/{len(kb_items)}] {cve}: own 缺失，跳过", flush=True)
            continue

        t0 = time.time()
        target_dir = _remap_target_dir(it["target_dir"])
        files = [f["path"] for f in discover_source_files(target_dir).get("files", [])]
        if not files:
            no_files += 1
            print(f"[{idx}/{len(kb_items)}] {cve}: 无源文件，跳过", flush=True)
            continue

        admitted = {m: False for m in MODES}
        for fp in files:
            report_data = {"file": fp, "issues": [], "run_id": "ablate", "requirement_id": 1}
            try:
                r = await agent._run_second_pass(report_data, original_analysis=None, layer_mode=None)
            except Exception as e:
                print(f"[{idx}/{len(kb_items)}] {cve} {fp}: _run_second_pass 异常 {e}", flush=True)
                continue
            gap = r.get("gap_retrieval_evidence") or []
            # 一致性校验：重算的 all(4层) 应与生产 new_findings 的自命中一致
            prod_all = _own_in_findings(r.get("new_findings") or [], own, ci_to_pattern)
            my_all = _recall_for_subset(agent, gap, sqlite_patterns, None, own, fp, ci_to_pattern)
            if prod_all != my_all:
                print(
                    f"    [warn] all 重算与生产不一致: prod={prod_all} regate={my_all} "
                    f"(cve={cve} file={fp})",
                    flush=True,
                )
            dump_f.write(
                json.dumps(
                    {"cve": cve, "own_id": own, "file": fp, "gap_evidence": gap},
                    ensure_ascii=False,
                    default=str,
                ) + "\n"
            )
            dump_f.flush()
            for m, subset in MODES.items():
                if not admitted[m] and _recall_for_subset(
                    agent, gap, sqlite_patterns, subset, own, fp, ci_to_pattern
                ):
                    admitted[m] = True

        rec = {"cve": cve, "own_id": own, **{m: admitted[m] for m in MODES}}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        done += 1
        for m in MODES:
            if admitted[m]:
                totals[m] += 1
        dt = time.time() - t0
        flags = " ".join("+" if admitted[m] else "." for m in MODES)
        print(
            f"[{idx}/{len(kb_items)}] {cve} done {dt:.1f}s [{flags}] "
            + " ".join(f"{m}={totals[m]}" for m in MODES),
            flush=True,
        )

    out_f.close()
    dump_f.close()
    await agent.stop()

    print(f"\n本进程完成 {done} 个；own 缺失 {no_own}；无源文件 {no_files}", flush=True)
    _summary_from_results(out_path)


def _run_regate_only(args) -> None:
    """纯离线重算：读 dump JSONL，不连 Weaviate / 不加载模型（DB 仅读侧车）。"""
    dump_path = Path(args.dump)
    if not dump_path.exists():
        print(f"[fatal] dump 不存在: {dump_path}", flush=True)
        sys.exit(1)

    sqlite_patterns = []
    ci_to_pattern = {}
    sqlite_sidecar = Path(args.sqlite_sidecar)
    if sqlite_sidecar.exists():
        sidecar = json.loads(sqlite_sidecar.read_text(encoding="utf-8"))
        sqlite_patterns = sidecar.get("sqlite_patterns") or []
        ci_to_pattern = {int(k): v for k, v in (sidecar.get("ci_to_pattern") or {}).items()}

    # 仅构造 agent（不 initialize），纯内存门控/派生方法无需 Weaviate/模型
    agent = AIDrivenSecondPassAnalysisAgent()

    # 按 CVE 聚合（同一 CVE 可能有多个文件的 dump 行）
    cve_admitted = {}   # cve -> {own_id, mode: bool}
    cve_order = []
    n = 0
    t0 = time.time()
    for line in dump_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cve = rec.get("cve")
        own = rec.get("own_id")
        gap = rec.get("gap_evidence") or []
        if cve not in cve_admitted:
            cve_admitted[cve] = {"own_id": own, **{m: False for m in MODES}}
            cve_order.append(cve)
        for m, subset in MODES.items():
            if not cve_admitted[cve][m] and _recall_for_subset(
                agent, gap, sqlite_patterns, subset, own, rec.get("file"), ci_to_pattern
            ):
                cve_admitted[cve][m] = True
        n += 1
        if n % 50 == 0:
            print(f"[regate] {n} 行已重算，耗时 {time.time()-t0:.1f}s", flush=True)

    out_path = Path(args.out)
    out_f = open(out_path, "w", encoding="utf-8")
    for cve in cve_order:
        r = cve_admitted[cve]
        out_f.write(
            json.dumps({"cve": cve, "own_id": r["own_id"], **{m: r[m] for m in MODES}}, ensure_ascii=False) + "\n"
        )
    out_f.close()
    print(f"[regate] 完成 {len(cve_order)} 个 CVE（{n} 行），总耗时 {time.time()-t0:.1f}s", flush=True)
    _summary_from_results(out_path)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch", type=str, default="")
    ap.add_argument("--db", type=str, default=str(DB_DEFAULT))
    ap.add_argument("--out", type=str, default=str(REPORTS / "ablate_views_singlepass_results.jsonl"))
    ap.add_argument("--dump", type=str, default=str(REPORTS / "ablate_views_evidence.jsonl"))
    ap.add_argument("--sqlite-sidecar", type=str, default=str(REPORTS / "ablate_views_sqlite_patterns.json"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--regate-only", action="store_true")
    ap.add_argument("--shard-id", type=int, default=0, help="分片编号（0 起）")
    ap.add_argument("--shard-count", type=int, default=1, help="总分片数（>1 时按 stride 取子集，输出文件带 _shardN 后缀）")
    ap.add_argument("--merge", action="store_true", help="合并所有 shard 结果并输出总召回")
    args = ap.parse_args()

    if args.merge:
        _merge_summary(REPORTS, "ablate_views_singlepass_results")
    elif args.regate_only:
        _run_regate_only(args)
    else:
        await _run_retrieval(args)


if __name__ == "__main__":
    asyncio.run(main())
