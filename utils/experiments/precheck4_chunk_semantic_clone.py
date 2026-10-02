# -*- coding: utf-8 -*-
"""**A4 预检**：语义重现到底存不存在——把粒度从"整文件"降到"代码块"再量一次。

## 为什么这是决定性实验
预检 2 的结论是：文件级均值池化向量**表示不了那 2 个字符的补丁**（中位改动仅 0.98%），
所以"文件级语义"只能匹配话题、匹配不了缺陷。据此你已拍板 **B2：语义通道改块级**。

但"改成块级"值不值得做，取决于一个**必须先量的事实**：
**在块级上，同一个缺陷的语义相似度是否显著高于随机？**（即"语义重现"是否真实存在）

## 三个探针（都用生产实现）
* **A4-1 块级天花板**：样本文件按生产切分器切成块 → 每块嵌成查询向量（`_query_embed`，
  含 C2 偏移修正）→ 与**自己那条 KB 条目的 4 层索引向量**比 → 取全块最大值。
  和文件级（预检 2：max 0.573、0/30 过 τ）对比：**块级能不能把分数抬起来**。
* **A4-2 命中指向**：每块在自己**全部 200 条** KB 条目里找最像的那条 → 看它是不是"自己那条"
  （块级 top-1 命中率）；这直接说明"用块当查询能不能定位到正确的知识"。
* **A4-3 跨文件语义克隆**（回答"数据集里有没有语义重现场景"）：
  对每个样本，找出**含针的块**（针用生产 `_extract_error_code_fragments` 抽、`_is_contiguous_subseq` 判命中），
  再在**其它样本**的所有块里找最像的一块 → 统计有多少样本存在"块级语义克隆"，
  并用随机块对做对照算 z 分数。

用法:
    python -X utf8 utils/experiments/precheck4_chunk_semantic_clone.py --limit 30
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from utils.kb_coverage import SOURCE_EXT  # noqa: E402

LAYERS = ("semantic", "code_pattern", "solution", "full")
TAU = 0.65
DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"


def cos(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return num / (na * nb) if na and nb else 0.0


def load_dump(path: Path):
    vecs, meta = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        layer = str(r.get("vector_layer") or "")
        if layer not in LAYERS:
            continue
        sid = int(r["sqlite_id"])
        vecs[(sid, layer)] = [float(x) for x in (r.get("_vector") or [])]
        meta[sid] = {"file_pattern": str(r.get("file_pattern") or ""),
                     "solution": str(r.get("solution") or "")}
    return vecs, meta


def pick_file(cve: str, kb_base: str, part: str = "before"):
    d = DS / part / cve
    if not d.is_dir():
        return None
    files = [f for f in sorted(d.rglob("*")) if f.is_file() and f.suffix.lower() in SOURCE_EXT]
    if not files:
        return None
    if kb_base:
        for f in files:
            if f.name.lower() == kb_base.lower():
                return f
    return files[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path,
                    default=ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl")
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_rebuild_candidate_v2.db")
    ap.add_argument("--runs", type=Path, default=ROOT / "reports/arm1_runs.txt")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--json-out", type=Path,
                    default=ROOT / "reports/precheck4_chunk_semantic_clone.json")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    vecs, meta = load_dump(args.dump)
    all_sids = sorted({s for s, _ in vecs})
    print("索引 (条目,层) 对: %d，条目数: %d；查询偏移修正: %s"
          % (len(vecs), len(all_sids), agent.query_offset_correction))

    con = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    con.close()

    cves = [l.strip().split("/")[0] for l in args.runs.read_text(encoding="utf-8").splitlines()
            if l.strip()][: args.limit]
    own_of = [(c, id_by_title.get(c.upper())) for c in cves if id_by_title.get(c.upper())]

    # 逐样本切块 + 嵌向量（缓存起来给 A4-3 用）
    per_sample = []
    for cve, sid in own_of:
        kb_base = Path((meta.get(sid) or {}).get("file_pattern") or "").name
        f = pick_file(cve, kb_base)
        if not f:
            continue
        chunks = agent._split_file_into_context_chunks(str(f))
        entry = {"cve": cve, "sid": sid, "file": f.name, "chunks": []}
        for ch in chunks:
            text = ch["text"]
            if len(text.strip()) < 40:
                continue
            vec = {L: agent._query_embed(text, L) for L in LAYERS}
            entry["chunks"].append({"start_line": ch["start_line"], "end_line": ch["end_line"],
                                    "text": text, "vec": vec})
        per_sample.append(entry)
        print("  %-16s %-40s 块数 %d" % (cve, f.name[:40], len(entry["chunks"])))

    # ---------- A4-1 块级天花板 ----------
    print("\n" + "=" * 96)
    print("A4-1 块级天花板：每块 vs 自己那条 KB 条目的 4 层索引向量，取全块最大")
    print("     （对照：预检 2 的文件级结果是 max 0.573、0/30 过 τ）")
    print("=" * 96)
    print("  %-16s %8s %-14s %8s %9s %9s %6s" %
          ("CVE", "块级max", "最佳层/块", "文件级", "对照均", "对照σ", "z"))
    rows = []
    for s in per_sample:
        best, where, best_ctrl = 0.0, "", []
        rng = random.Random(hash(s["cve"]) & 0xFFFF)
        others = [x for x in all_sids if x != s["sid"]]
        ctrl_sids = rng.sample(others, min(20, len(others)))
        for ch in s["chunks"]:
            for L in LAYERS:
                v = vecs.get((s["sid"], L))
                if v:
                    sc = cos(ch["vec"][L], v)
                    if sc > best:
                        best, where = sc, "%s#%d" % (L, ch["start_line"])
        # 对照：**必须与"取最大值"这个操作对齐** ——
        # 对每个随机条目，同样取"全块 × 4 层"的最大值，得到一串"别人能达到的最大值"，
        # 再和"自己那条的最大值"比。第一版把对照写成"所有 (块,层,别人) 配对的扁平列表"，
        # 那等于拿"自己的最大值"去比"别人配对的中位数"，z 会被系统性抬高（选择偏差）。
        ctrl_maxes = []
        for other in ctrl_sids:
            m = 0.0
            for ch in s["chunks"]:
                for L in LAYERS:
                    v = vecs.get((other, L))
                    if v:
                        m = max(m, cos(ch["vec"][L], v))
            ctrl_maxes.append(m)
        best_ctrl = ctrl_maxes
        mean = sum(best_ctrl) / len(best_ctrl) if best_ctrl else 0.0
        var = sum((x - mean) ** 2 for x in best_ctrl) / len(best_ctrl) if best_ctrl else 0.0
        std = math.sqrt(var)
        z = (best - mean) / std if std > 1e-9 else None
        rows.append({"cve": s["cve"], "chunk_max": round(best, 4), "where": where,
                     "ctrl_mean": round(mean, 4), "ctrl_std": round(std, 4),
                     "z": round(z, 2) if z is not None else None})
        print("  %-16s %8.4f %-14s %8s %9.4f %9.4f %6s"
              % (s["cve"], best, where, "0.573(max)", mean, std,
                 ("%.2f" % z) if z is not None else "-"))

    cm = [r["chunk_max"] for r in rows]
    zs = [r["z"] for r in rows if r["z"] is not None]
    print("\n  块级 max 分布: min %.3f  p50 %.3f  max %.3f（文件级 p50 0.183 / max 0.573）"
          % (min(cm), sorted(cm)[len(cm) // 2], max(cm)))
    print("  ≥ τ(%.2f): %d/%d   z ≥ 2: %d/%d   z ≥ 3: %d/%d"
          % (TAU, sum(1 for x in cm if x >= TAU), len(cm),
             sum(1 for z in zs if z >= 2), len(zs), sum(1 for z in zs if z >= 3), len(zs)))

    # ---------- A4-2 命中指向 ----------
    print("\n" + "=" * 96)
    print("A4-2 命中指向：每块在全部 200 条 KB 条目里最像的是不是自己那条")
    print("=" * 96)
    top1_own = top1_tot = 0
    for s in per_sample:
        for ch in s["chunks"]:
            best_sid, best_sc = None, -9.0
            for L in LAYERS:
                q = ch["vec"][L]
                for sid in all_sids:
                    v = vecs.get((sid, L))
                    if not v:
                        continue
                    sc = cos(q, v)
                    if sc > best_sc:
                        best_sc, best_sid = sc, sid
            top1_tot += 1
            if best_sid == s["sid"]:
                top1_own += 1
    print("  块级 top-1 命中自己那一条的比例: %d/%d = %.1f%%" % (top1_own, top1_tot, 100.0 * top1_own / max(top1_tot, 1)))

    # ---------- A4-3 跨文件语义克隆 ----------
    print("\n" + "=" * 96)
    print("A4-3 跨文件语义克隆：含针的块，在**别的样本**里能不能找到语义克隆")
    print("=" * 96)
    # 先把所有块汇总（用于跨样本搜索）
    all_chunks = [(s["cve"], s["sid"], ch) for s in per_sample for ch in s["chunks"]]
    hits = 0
    for s in per_sample:
        sol = (meta.get(s["sid"]) or {}).get("solution") or ""
        frags = agent._extract_error_code_fragments(sol)
        if not frags:
            continue
        # 找到含针的块
        target = None
        for ch in s["chunks"]:
            toks = agent._tokenize_code(ch["text"])
            if any(agent._is_contiguous_subseq(f, toks) for f in frags):
                target = ch
                break
        if not target:
            continue
        best, best_cve, best_l = 0.0, None, ""
        for cve2, sid2, ch2 in all_chunks:
            if cve2 == s["cve"]:
                continue
            for L in LAYERS:
                sc = cos(target["vec"][L], ch2["vec"][L])
                if sc > best:
                    best, best_cve, best_l = sc, cve2, L
        # 对照：**同样取"块 × 4 层"的最大值**（与本行上面的搜索侧对齐，避免选择偏差）。
        # 第一版对照只用 full 单层，而搜索侧取了 4 层最大值 → z 被系统性抬高。
        rng = random.Random(hash(s["cve"]) & 0xFFFF)
        ctrl = []
        for _ in range(60):
            cve2, sid2, ch2 = rng.choice(all_chunks)
            if cve2 == s["cve"]:
                continue
            m = max(cos(target["vec"][L], ch2["vec"][L]) for L in LAYERS)
            ctrl.append(m)
        mean = sum(ctrl) / len(ctrl) if ctrl else 0.0
        std = (sum((x - mean) ** 2 for x in ctrl) / len(ctrl)) ** 0.5 if ctrl else 0.0
        z = (best - mean) / std if std > 1e-9 else None
        hit = z is not None and z >= 2
        hits += 1 if hit else 0
        print("  %-16s 针块 L%-4d 最像 → %-16s 层=%-12s cos=%.3f  z=%s  %s"
              % (s["cve"], target["start_line"], best_cve or "-", best_l, best,
                 ("%.2f" % z) if z is not None else "-", "**语义克隆**" if hit else ""))
    print("\n  存在跨样本「语义克隆」（z ≥ 2）的样本: %d 个" % hits)

    out = {"tau": TAU, "a4_1": rows, "a4_2": {"top1_own": top1_own, "total": top1_tot},
           "a4_3": {"clones": hits}}
    args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写入 %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
