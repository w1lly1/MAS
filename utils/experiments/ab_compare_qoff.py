#!/usr/bin/env python
"""C2 修复端到端 A/B：对比「基线」与「开启 query_offset_correction 后」的生产实测指标。

与 Stage B/D 不同，本步不使用任何代理查询文本——直接读取两次运行落盘的
weaviate_hits（含真实 top-5 与 similarity），因此在生产查询集上直接可比。

指标：
  · 相似度分布 mean/σ（判定力）
  · hubness：N_5 的 max / 偏度 / anti-hub / top-10 集中度
  · 被召回 KB 行数
  · 跨全部被分析文件共现的行数（目标 0）
  · 精度：命中行的同文件% / 同项目%（KB file_pattern 取 SQLite 真值，随机基线作对照）
  · 门控：formal/explanatory 晋升数、evidence_hits 数、rejection_reason 分布
"""
import glob
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

BASELINE_RUNS = {
    "f5fbeb25-8c47-4b11-b3a1-870ceafe10bb", "42efd1e7-b6fb-4874-af25-11531b9779b1",
    "a78ea346-820c-4a87-9b7e-e9a96ad5f098", "66f9ad5f-4407-4ce0-a772-12a9a737ec7c",
    "9a5a2f2d-4b21-4763-ab10-6730bde0bc48", "ba139961-bc84-45e0-8b62-39246b6928fe",
    "902dc158-5c26-4f45-acf6-4d9acd10dfdd", "a1ca5b0b-0c8a-4cdc-bb43-044d0c16cb9e",
}
LAYERS = ["semantic", "code_pattern", "solution", "full"]


def norm_path(p):
    s = str(p or "").replace("\\", "/").strip().lower()
    b = os.path.basename(s)
    if "__" in b:
        s = (os.path.dirname(s) + "/" if os.path.dirname(s) else "") + b.replace("__", "/")
    return s


def base_of(p):
    return os.path.basename(norm_path(p))


def proj_of(p):
    n = norm_path(p)
    return n.split("/")[0] if "/" in n else ""


def skew(x):
    x = np.asarray(x, float)
    s = x.std()
    return 0.0 if s == 0 or x.size < 3 else float(((x - x.mean()) ** 3).mean() / s ** 3)


def fp_by_sid():
    c = sqlite3.connect(os.path.join(ROOT, "infrastructure", "database", "mas.db"))
    return {str(r[0]): (r[1] or "") for r in
            c.execute("select id, file_pattern from issue_patterns").fetchall()}


def collect(prefix, runs, fpmap):
    """返回 (recorded_hits, candidates, evidence)"""
    hits, cands, ev = [], [], 0
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        run = f.split("/")[3] if len(f.split("/")) > 3 else ""
        if run not in runs:
            continue
        d = json.load(open(f, encoding="utf-8"))
        ana = os.path.basename(str(d.get("file") or ""))
        for b in ("retrieval_evidence", "gap_retrieval_evidence"):
            for qi, it in enumerate(d.get(b) or []):
                ev += len(it.get("evidence_hits") or [])
                for h in (it.get("weaviate_hits") or []):
                    hits.append({"ana": ana, "b": b, "qi": qi, "sid": h.get("sqlite_id"),
                                 "layer": str(h.get("vector_layer")),
                                 "sim": h.get("similarity"), "dist": h.get("distance")})
                for c in (it.get("candidates") or []):
                    cands.append({"ana": ana, "ch": str(c.get("channel")),
                                  "gd": str(c.get("gating_decision")),
                                  "rr": str(c.get("rejection_reason")),
                                  "mf": c.get("matched_fields") or []})
    return hits, cands, ev


def report(name, hits, cands, ev, fpmap):
    print("\n" + "=" * 78)
    print("### %s" % name)
    files = sorted({h["ana"] for h in hits})
    print("  文件 %d  命中 %d  文件均命中 %.0f" % (len(files), len(hits), len(hits) / max(1, len(files))))
    sims = np.array([float(h["sim"]) for h in hits if isinstance(h["sim"], (int, float))])
    if sims.size:
        print("  相似度(recorded=1-d/2): mean=%.4f σ=%.4f min=%.4f max=%.4f" % (
            sims.mean(), sims.std(), sims.min(), sims.max()))
    # 精度（用 SQLite 真值 file_pattern）
    nf = npr = tot = 0
    for h in hits:
        fp = fpmap.get(str(h["sid"]), "")
        if not fp:
            continue
        tot += 1
        if base_of(fp) == base_of(h["ana"]):
            nf += 1
        if proj_of(fp) and proj_of(fp) == proj_of(h["ana"]):
            npr += 1
    if tot:
        fs = [base_of(f) for f in files]
        pr = [proj_of(f) for f in files]
        allfp = list(fpmap.values())
        ef = float(np.mean([sum(1 for p in allfp if base_of(p) == b) / len(allfp) for b in fs]))
        ep = float(np.mean([sum(1 for p in allfp if proj_of(p) == p0) / len(allfp) for p0 in pr]))
        print("  同文件 %.2f%%（随机期望 %.2f%%）  同项目 %.2f%%（随机期望 %.2f%%）" % (
            100 * nf / tot, 100 * ef, 100 * npr / tot, 100 * ep))
    # hubness：按 (查询, 层) 的 top-5
    byq = defaultdict(list)
    for h in hits:
        byq[(h["ana"], h["b"], h["qi"], h["layer"])].append(h["sid"])
    N = Counter()
    for k, sids in byq.items():
        for s in set(sids):
            N[s] += 1
    nv = np.array(list(N.values()), float) if N else np.array([0.0])
    slots = len(byq)
    tot_hits_rows = sum(len(set(v)) for v in byq.values())
    od = np.sort(nv)[::-1]
    print("  查询层实例 %d  被召回 KB 行 %d/200  每行平均被命中 %.2f" % (
        slots, len(N), tot_hits_rows / max(1, len(N))))
    print("  hubness: N_max=%.0f  N_skew=%.2f  anti-hub=%d  top10 集中度=%.3f" % (
        nv.max(), skew(nv), int((nv == 0).sum()), od[:10].sum() / max(1.0, nv.sum())))
    # 跨文件共现
    rowfiles = defaultdict(set)
    for (ana, b, qi, L), sids in byq.items():
        for s in set(sids):
            rowfiles[s].add(ana)
    allf = len(files)
    co = sorted([(s, len(v)) for s, v in rowfiles.items() if len(v) >= allf], key=lambda x: -x[1])
    print("  跨全部 %d 文件共现的 KB 行: %d %s" % (allf, len(co), [s for s, _ in co][:8]))
    # 门控
    gd = Counter(c["gd"] for c in cands)
    rr = Counter(c["rr"] for c in cands)
    print("  门控: 候选=%d  evidence_hits=%d  判定=%s" % (len(cands), ev, dict(gd)))
    print("        拒绝原因=%s" % dict(rr.most_common(5)))
    mf = Counter(m for c in cands for m in c["mf"])
    print("        强字段命中=%s" % {k: mf.get(k, 0) for k in
          ("error_code_clone", "file_basename_anchor", "basename_match",
           "class_pattern_in_code", "function_name_in_code")})
    return {"files": len(files), "hits": len(hits), "rows": len(N),
            "top10": float(od[:10].sum() / max(1.0, nv.sum())),
            "coexist": len(co), "admit": gd.get("formal_hit", 0) + gd.get("explanatory_hit", 0),
            "ev": ev,
            "same_file": 100.0 * nf / max(1, tot), "same_proj": 100.0 * npr / max(1, tot)}


def main():
    fpmap = fp_by_sid()
    cur = set()
    for f in glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json"):
        cur.add(f.split("/")[3])
    before = set()
    bp = "/root/autodl-tmp/_runs_before2.txt"
    if os.path.exists(bp):
        before = set(x.strip() for x in open(bp) if x.strip())
    # 新 run = 当前全部 run 减去「启动本轮前已存在」的 run
    new_runs = cur - before if before else (cur - BASELINE_RUNS)
    print("基线 run 命中 %d 个 | 启动前已有 %d 个 | 本轮新增 %d 个" % (
        len(BASELINE_RUNS & cur), len(before), len(new_runs)))
    print("本轮新增 run:", sorted(new_runs))

    b_h, b_c, b_e = collect("base", BASELINE_RUNS, fpmap)
    n_h, n_c, n_e = collect("new", new_runs, fpmap)
    rb = report("基线（query_offset_correction = false）", b_h, b_c, b_e, fpmap)
    rn = report("A/B（query_offset_correction = true）", n_h, n_c, n_e, fpmap)

    print("\n" + "=" * 78)
    print("### 汇总对比")
    print("%-22s %12s %12s %10s" % ("指标", "基线", "偏移修正", "变化"))
    for k, label, better in (("top10", "top-10 集中度", "down"), ("coexist", "跨文件共现行数", "down"),
                             ("rows", "被召回 KB 行数", "up"), ("admit", "晋升数", "up"),
                             ("ev", "evidence_hits", "up"),
                             ("same_file", "同文件命中%", "up"), ("same_proj", "同项目命中%", "up")):
        a, b = rb[k], rn[k]
        if isinstance(a, float):
            print("%-22s %12.4f %12.4f %10s" % (label, a, b, "↓" if b < a else ("↑" if b > a else "=")))
        else:
            print("%-22s %12s %12s %10s" % (label, a, b, "↓" if b < a else ("↑" if b > a else "=")))
    json.dump({"baseline": rb, "qoff": rn}, open("/root/autodl-tmp/ab_qoff.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n已写出 /root/autodl-tmp/ab_qoff.json")


if __name__ == "__main__":
    sys.exit(main())
