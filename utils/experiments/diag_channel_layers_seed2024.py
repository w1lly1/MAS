# -*- coding: utf-8 -*-
"""seed=2024 通道归因 + 语义通道逐层拆解（本地 CPU，numpy 复刻检索）。

按《实验数据汇总.md》口径（diag_channel_attribution.py 同款定义）：
  - 词法通道 = finding.matched_fields 含 "error_code_clone"（错误代码连续子串匹配）
  - 语义通道 = finding.channel == "weaviate"
  - 语义通道独立命中 = 有 weaviate 自命中 且 无 error_code_clone 自命中

额外：对每个「语义独立命中」的 CVE，记录其 weaviate 自命中来自哪几层
（matched_layers / vector_layer），回答「语义通道独立召回是通过哪几层」。

产出：reports/channel_layers_seed2024.jsonl + 汇总
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
import sqlite3  # noqa: E402
import time  # noqa: E402

from core.agents.ai_driven_second_pass_analysis_agent import (  # noqa: E402
    AIDrivenSecondPassAnalysisAgent,
)
from infrastructure.database.sqlite.service import DatabaseService  # noqa: E402
from infrastructure.database.weaviate.service import WeaviateVectorService  # noqa: E402
from infrastructure.embeddings.codebert_embedder import embed_text  # noqa: E402
from utils.experiments.local_numpy_weaviate import NumpyVectorService  # noqa: E402
from utils.scan_discovery import discover_source_files  # noqa: E402

BATCH = ROOT / "论文" / "test_400_error_batch.json"          # seed=2024
DB = ROOT / "infrastructure" / "database" / "mas.db"          # seed=2024
KB_DUMP = ROOT / "reports" / "weaviate_kb_seed2024.jsonl"


def build_seed2024_kb() -> None:
    """用 seed=2024 的 200 条 issue_patterns 重建 KB dump（4 层 × 200 = 800 对象）。"""
    if KB_DUMP.exists():
        return
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute(
        "SELECT id, title, error_type, severity, language, framework, error_description, "
        "problematic_pattern, solution, file_pattern, class_pattern FROM issue_patterns"
    )
    rows = cur.fetchall()
    con.close()
    svc = WeaviateVectorService()  # 复用 layer_text 构建器（stub 满足 import）
    out = open(KB_DUMP, "w", encoding="utf-8")
    for r in rows:
        props = {
            "sqlite_id": r[0], "title": r[1] or "", "error_type": r[2] or "", "severity": r[3] or "",
            "language": r[4] or "", "framework": r[5] or "", "error_description": r[6] or "",
            "problematic_pattern": r[7] or "", "solution": r[8] or "", "file_pattern": r[9] or "",
            "class_pattern": r[10] or "",
        }
        for layer in ("semantic", "code_pattern", "solution", "full"):
            lt = svc._build_enhanced_issue_pattern_text(props, layer)
            vec = embed_text(lt, layer)
            rec = dict(props)
            rec["vector_layer"] = layer
            rec["layer_text"] = lt
            rec["_vector"] = vec
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    print(f"KB dump 已生成: {KB_DUMP}", flush=True)


def _build_agent() -> AIDrivenSecondPassAnalysisAgent:
    agent = AIDrivenSecondPassAnalysisAgent()
    agent.vector_service = NumpyVectorService(KB_DUMP)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    return agent


def _remap(target_dir: str) -> str:
    if not target_dir:
        return target_dir
    if target_dir.startswith("/root/autodl-tmp/MAS/"):
        return str(ROOT / target_dir[len("/root/autodl-tmp/MAS/"):])
    return target_dir


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "channel_layers_seed2024.jsonl"))
    args = ap.parse_args()

    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    build_seed2024_kb()
    agent = _build_agent()

    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    kb = [it for it in batch["items"] if it.get("role") == "kb"]
    if args.limit > 0:
        kb = kb[: args.limit]
    if args.shard_count > 1:
        kb = [it for i, it in enumerate(kb) if i % args.shard_count == args.shard_id]

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id, title FROM issue_patterns")
    id_by_title = {t: i for i, t in cur.fetchall()}
    cur.execute("SELECT id, pattern_id FROM curated_issues")
    ci_to_p = {i: p for i, p in cur.fetchall()}
    con.close()

    def is_self(ch, sid, own):
        if own is None:
            return False
        if ch == "curated_issue":
            return ci_to_p.get(sid) == own
        return sid == own

    from utils.experiments.ablate_views_singlepass import _shard_path
    out_path = _shard_path(args.out, args.shard_id, args.shard_count)
    out_f = open(out_path, "w", encoding="utf-8")

    total = lex_total = sem_total = 0
    only_lex = only_sem = both = 0
    sem_layers = {}  # CVE -> layers（语义独立命中的逐层）
    print(f"kb 样本数: {len(kb)}", flush=True)
    for idx, it in enumerate(kb, 1):
        cve = it.get("cve")
        own = id_by_title.get(cve)
        files = [f["path"] for f in discover_source_files(_remap(it["target_dir"])).get("files", [])]
        if not files:
            print(f"[{idx}/{len(kb)}] {cve}: 无源文件", flush=True)
            continue
        t0 = time.time()
        has_lex = has_sem = False
        cve_layers = set()
        for fp in files:
            report_data = {"file": fp, "issues": [], "run_id": "ablate", "requirement_id": 1}
            try:
                r = await agent._run_second_pass(report_data, original_analysis=None, layer_mode=None)
            except Exception as e:
                print(f"[{idx}/{len(kb)}] {cve} {fp}: 异常 {e}", flush=True)
                continue
            for nf in r.get("new_findings") or []:
                ev = nf.get("evidence") if isinstance(nf, dict) else {}
                if not is_self(ev.get("channel"), ev.get("sqlite_id"), own):
                    continue
                mf = ev.get("matched_fields") or []
                ch = str(ev.get("channel") or "").lower()
                if "error_code_clone" in mf:
                    has_lex = True
                if ch == "weaviate":
                    has_sem = True
                    for l in (ev.get("matched_layers") or [ev.get("vector_layer")]):
                        if l:
                            cve_layers.add(str(l).strip().lower())
        if not (has_lex or has_sem):
            out_f.write(json.dumps({"cve": cve, "hit": False}, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[{idx}/{len(kb)}] {cve} miss {time.time()-t0:.1f}s", flush=True)
            continue
        total += 1
        if has_lex:
            lex_total += 1
        if has_sem:
            sem_total += 1
        if has_lex and has_sem:
            both += 1
        elif has_lex:
            only_lex += 1
        else:
            only_sem += 1
            sem_layers[cve] = sorted(cve_layers)
        out_f.write(json.dumps({
            "cve": cve, "hit": True, "lex": has_lex, "sem": has_sem,
            "sem_layers": sorted(cve_layers),
        }, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"[{idx}/{len(kb)}] {cve} hit lex={has_lex} sem={has_sem} layers={sorted(cve_layers)} {time.time()-t0:.1f}s", flush=True)

    out_f.close()
    print("\n==== seed=2024 通道归因 ====")
    print(f"总召回: {total}")
    print(f"词法(error_code_clone): {lex_total}")
    print(f"语义(weaviate): {sem_total}")
    print(f"  两者重叠: {both}")
    print(f"  只词法: {only_lex}")
    print(f"  只语义(独立命中): {only_sem}")
    print("\n语义独立命中的 CVE 与逐层:")
    for cve, layers in sorted(sem_layers.items()):
        print(f"  {cve}: {layers}")


if __name__ == "__main__":
    asyncio.run(main())
