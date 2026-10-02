# -*- coding: utf-8 -*-
"""**预检 2**：语义通道的"天花板"到底在哪——是门限标定错了，还是数据里没有语义重现？

## 为什么必须做这个预检
上一轮实测发现：语义（weaviate）通道 17,518 个候选里**最高分 0.625**，而门限 **τ = 0.65**
⇒ 语义那一支判据在现有向量空间里**永远不可能满足**。但"永远不可能"有两种完全不同的原因，
对应两条完全不同的修法：

* **标定问题**：连"拿知识条目自己的文本当查询"都对不上自己的索引向量（自相似度就 0.6x）
  ⇒ 是**门限/偏移变换的刻度问题**，改标定即可（文献：按通道归一化 / 排名融合）；
* **数据问题**：自相似度其实是 1.0（管线自洽），只是**真实查询文本与索引文本差太远**
  ⇒ 要改**索引文本/查询文本**（doc2query、RM3 那一类），或者承认数据集里没有语义重现。

## 三个探针
* **P1 自洽性（标定）**：`cos(偏移修正后的「条目自己的 layer_text」, 该条目自己的索引向量)`
  —— 这是**理想查询**下的上界。若这个值都低于 τ，那 τ 就是不可达的（纯标定问题）。
* **P2 样本侧天花板**：对每个 kb-self 样本 `cos(偏移修正后的「被分析文件全文」, 自己那条的 4 层索引向量)`
  —— "若语义检索真的能召回自己的存量知识，分数会是多少"。
* **P3 对照**：同一批样本用 **after（已打补丁）文件**当查询；再对每个样本取若干**随机条目**
  作对照，算 z 分数，排除"整个空间都挤在一起"的假象。

用的都是**生产实现**：`embed_text`（distilbert + 逐层白化）→ `_apply_query_offset`（C2 的查询偏移）；
索引侧向量直接取**线上导出的 dump**（`weaviate_kb_dump_postswitch.jsonl`），不重新嵌入，避免口径漂移。

用法:
    python -X utf8 utils/experiments/precheck2_semantic_ceiling.py --limit 30
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
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
    """{(sqlite_id, layer): vector} + {sqlite_id: {layer: layer_text}}"""
    vecs, texts, meta = {}, {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid, layer = int(r["sqlite_id"]), str(r.get("vector_layer") or "")
        if layer not in LAYERS:
            continue
        vecs[(sid, layer)] = [float(x) for x in (r.get("_vector") or [])]
        texts.setdefault(sid, {})[layer] = str(r.get("layer_text") or "")
        meta[sid] = {"file_pattern": r.get("file_pattern") or "", "title": None}
    return vecs, texts, meta


def p1_self_consistency(agent, vecs, texts, sample_n: int, seed: int = 2025):
    """P1：拿条目自己的 layer_text 当查询（偏移修正后），对自己的索引向量算相似度。"""
    sids = sorted({s for s, _ in vecs})
    rng = random.Random(seed)
    pick = rng.sample(sids, min(sample_n, len(sids))) if sample_n else sids
    rows = []
    for sid in pick:
        for layer in LAYERS:
            v = vecs.get((sid, layer))
            t = (texts.get(sid) or {}).get(layer) or ""
            if not v or not t:
                continue
            q_raw = agent._default_embed(t, layer)
            q_off = agent._query_embed(t, layer)
            rows.append({"sid": sid, "layer": layer,
                         "cos_raw": cos(q_raw, v), "cos_offset": cos(q_off, v)})
    return rows


def _pick_file(cve: str, kb_basename: str, part: str = "before"):
    d = DS / part / cve
    if not d.is_dir():
        return None
    files = [f for f in sorted(d.rglob("*")) if f.is_file() and f.suffix.lower() in SOURCE_EXT]
    if not files:
        return None
    if kb_basename:
        for f in files:
            if f.name.lower() == kb_basename.lower():
                return f
    return files[0]


def p2_sample_ceiling(agent, vecs, meta, own_of, limit: int, parts=("before", "after")):
    """P2/P3：样本文件当查询，对自己那条的 4 层向量算相似度；并做随机条目对照。"""
    out = []
    all_sids = sorted({s for s, _ in vecs})
    for cve, sid in own_of[:limit]:
        kb_base = (meta.get(sid) or {}).get("file_pattern") or ""
        kb_base = Path(kb_base).name
        rec = {"cve": cve, "sid": sid}
        for part in parts:
            f = _pick_file(cve, kb_base, part)
            if not f:
                rec[part] = None
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")[:4000]
            best, best_layer, per_layer = 0.0, "", {}
            for layer in LAYERS:
                v = vecs.get((sid, layer))
                if not v:
                    continue
                q = agent._query_embed(text, layer)
                s = cos(q, v)
                per_layer[layer] = round(s, 4)
                if s > best:
                    best, best_layer = s, layer
            # 随机对照：同一批层、别的条目
            rng = random.Random(hash(cve) & 0xFFFF)
            ctrl = []
            for other in rng.sample(all_sids, min(30, len(all_sids))):
                if other == sid:
                    continue
                v = vecs.get((other, best_layer or "full"))
                if not v:
                    continue
                q = agent._query_embed(text, best_layer or "full")
                ctrl.append(cos(q, v))
            mean = sum(ctrl) / len(ctrl) if ctrl else 0.0
            var = sum((x - mean) ** 2 for x in ctrl) / len(ctrl) if ctrl else 0.0
            std = math.sqrt(var)
            rec[part] = {"file": f.name, "best": round(best, 4), "best_layer": best_layer,
                         "per_layer": per_layer, "ctrl_mean": round(mean, 4),
                         "ctrl_std": round(std, 4),
                         "z": round((best - mean) / std, 2) if std > 1e-9 else None}
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path,
                    default=ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl")
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_rebuild_candidate_v2.db")
    ap.add_argument("--runs", type=Path, default=ROOT / "reports/arm1_runs.txt")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--p1-sample", type=int, default=40)
    ap.add_argument("--json-out", type=Path, default=ROOT / "reports/precheck2_semantic_ceiling.json")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent

    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    print("查询偏移修正开关: %s（生产配置）" % agent.query_offset_correction)

    vecs, texts, meta = load_dump(args.dump)
    print("索引向量: %d 个 (条目,层) 对，覆盖 %d 个条目" % (len(vecs), len({s for s, _ in vecs})))

    # ---------- P1 ----------
    print("\n" + "=" * 96)
    print("P1 自洽性：拿条目**自己的 layer_text** 当查询，对自己的索引向量")
    print("   （偏移修正前的 cos 是「管线上限」；修正后的 cos 是「生产实际拿到的分」）")
    print("=" * 96)
    rows = p1_self_consistency(agent, vecs, texts, args.p1_sample)
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)
    print("  %-14s %6s %14s %14s %10s" % ("层", "样本", "cos(原始)", "cos(偏移修正后)", "≥τ 的比例"))
    for layer in LAYERS:
        rs = by_layer.get(layer) or []
        if not rs:
            continue
        raw = sum(r["cos_raw"] for r in rs) / len(rs)
        off = sum(r["cos_offset"] for r in rs) / len(rs)
        over = sum(1 for r in rs if r["cos_offset"] >= TAU) / len(rs)
        print("  %-14s %6d %14.4f %14.4f %9.0f%%" % (layer, len(rs), raw, off, 100 * over))
    best_p1 = max((r["cos_offset"] for r in rows), default=0.0)
    print("  全部层的最大自相似度（偏移修正后）= %.4f" % best_p1)

    # ---------- P2/P3 ----------
    import sqlite3
    con = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    con.close()
    cves = [l.strip().split("/")[0] for l in args.runs.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    own_of = [(c, id_by_title.get(c.upper())) for c in cves if id_by_title.get(c.upper())]
    print("\n" + "=" * 96)
    print("P2/P3 样本侧天花板（%d 个 kb-self 样本；before=有漏洞文件，after=已修文件）" % len(own_of))
    print("=" * 96)
    res = p2_sample_ceiling(agent, vecs, meta, own_of, args.limit)
    print("  %-16s %8s %-12s %8s %8s %8s %6s" %
          ("CVE", "before", "最佳层", "after", "对照均", "对照σ", "z"))
    for r in res:
        b = r.get("before") or {}
        a = r.get("after") or {}
        print("  %-16s %8s %-12s %8s %8s %8s %6s"
              % (r["cve"], b.get("best", "-"), b.get("best_layer", "-"), a.get("best", "-"),
                 b.get("ctrl_mean", "-"), b.get("ctrl_std", "-"), b.get("z", "-")))

    befores = [ (r.get("before") or {}).get("best") for r in res ]
    befores = [x for x in befores if x is not None]
    afters = [ (r.get("after") or {}).get("best") for r in res ]
    afters = [x for x in afters if x is not None]
    zs = [ (r.get("before") or {}).get("z") for r in res ]
    zs = [x for x in zs if x is not None]

    def stat(v):
        if not v:
            return "-"
        return "min %.3f  p50 %.3f  max %.3f" % (min(v), sorted(v)[len(v) // 2], max(v))

    print("\n  before 文件对自己条目的最高相似度: %s" % stat(befores))
    print("  after  文件对自己条目的最高相似度: %s" % stat(afters))
    print("  z 分数（相对随机条目）: %s" % stat(zs))
    print("  ≥ τ(%.2f) 的样本数: before %d/%d，after %d/%d"
          % (TAU, sum(1 for x in befores if x >= TAU), len(befores),
             sum(1 for x in afters if x >= TAU), len(afters)))
    print("  z ≥ 2 的样本数: %d/%d" % (sum(1 for x in zs if x >= 2), len(zs)))

    out = {"tau": TAU, "p1": {"rows": rows, "max_cos_offset": best_p1},
           "p2": res,
           "summary": {"before_best": befores, "after_best": afters, "z": zs}}
    args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写入 %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
