#!/usr/bin/env python
"""通用两组 A/B 对比：比较任意两组 run 的生产实测指标。

用法：
    python ab_compare_sets.py <A_run_ids_file> <B_run_ids_file> [A标签] [B标签]

指标与 ab_compare_qoff.py 一致：相似度分布、同文件/同项目精度（含随机期望）、
hubness（N_5）、被召回 KB 行数、跨全部文件共现、门控判定与拒绝原因分布。
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


def collect(runs, fpmap):
    hits, cands, ev = [], [], 0
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        if f.split("/")[3] not in runs:
            continue
        d = json.load(open(f, encoding="utf-8"))
        ana = os.path.basename(str(d.get("file") or ""))
        for b in ("retrieval_evidence", "gap_retrieval_evidence"):
            for qi, it in enumerate(d.get(b) or []):
                ev += len(it.get("evidence_hits") or [])
                for h in (it.get("weaviate_hits") or []):
                    hits.append({"ana": ana, "b": b, "qi": qi, "sid": h.get("sqlite_id"),
                                 "layer": str(h.get("vector_layer")), "sim": h.get("similarity")})
                for c in (it.get("candidates") or []):
                    cands.append({"ch": str(c.get("channel")), "gd": str(c.get("gating_decision")),
                                  "rr": str(c.get("rejection_reason")),
                                  "mf": c.get("matched_fields") or []})
    return hits, cands, ev


def report(name, hits, cands, ev, fpmap):
    print("\n" + "=" * 80)
    print("### %s" % name)
    files = sorted({h["ana"] for h in hits})
    sims = np.array([float(h["sim"]) for h in hits if isinstance(h["sim"], (int, float))])
    print("  文件 %d  命中 %d" % (len(files), len(hits)))
    if sims.size:
        print("  相似度 mean=%.4f σ=%.4f min=%.4f max=%.4f" % (
            sims.mean(), sims.std(), sims.min(), sims.max()))
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
    allfp = list(fpmap.values())
    fs = [base_of(f) for f in files]
    pr = [proj_of(f) for f in files]
    ef = float(np.mean([sum(1 for p in allfp if base_of(p) == b) / len(allfp) for b in fs]))
    ep = float(np.mean([sum(1 for p in allfp if proj_of(p) == p0) / len(allfp) for p0 in pr]))
    if tot:
        print("  同文件 %.2f%%（随机 %.2f%%）  同项目 %.2f%%（随机 %.2f%%）" % (
            100 * nf / tot, 100 * ef, 100 * npr / tot, 100 * ep))
    byq = defaultdict(set)
    for h in hits:
        byq[(h["ana"], h["b"], h["qi"], h["layer"])].add(h["sid"])
    N = Counter()
    rowfiles = defaultdict(set)
    for (ana, b, qi, L), sids in byq.items():
        for s in sids:
            N[s] += 1
            rowfiles[s].add(ana)
    nv = np.array(list(N.values()), float) if N else np.array([0.0])
    od = np.sort(nv)[::-1]
    allf = len(files)
    co = [s for s, v in rowfiles.items() if len(v) >= allf]
    ks = [len({h["sid"] for h in hits if h["ana"] == f}) for f in files]
    exp = 200 * float(np.prod([k / 200.0 for k in ks])) if ks else 0.0
    print("  查询层实例 %d  被召回行 %d/200  hubness: N_max=%.0f skew=%.2f top10=%.3f" % (
        len(byq), len(N), nv.max(), skew(nv), od[:10].sum() / max(1.0, nv.sum())))
    print("  跨全部 %d 文件共现: %d 行  零模型期望=%.3f  超出=%.0f×" % (
        allf, len(co), exp, (len(co) / exp) if exp > 0 else float("inf")))
    gd = Counter(c["gd"] for c in cands)
    rr = Counter(c["rr"] for c in cands)
    mf = Counter(m for c in cands for m in c["mf"])
    print("  门控: 候选=%d evidence_hits=%d 判定=%s" % (len(cands), ev, dict(gd)))
    print("        拒绝=%s" % dict(rr.most_common(5)))
    print("        强字段=%s" % {k: mf.get(k, 0) for k in
          ("error_code_clone", "file_basename_anchor", "basename_match",
           "class_pattern_in_code", "function_name_in_code")})
    return {"files": len(files), "hits": len(hits), "rows": len(N),
            "top10": float(od[:10].sum() / max(1.0, nv.sum())),
            "coexist": len(co), "excess": (len(co) / exp) if exp > 0 else None,
            "admit": gd.get("formal_hit", 0) + gd.get("explanatory_hit", 0), "ev": ev,
            "same_file": 100.0 * nf / max(1, tot), "same_proj": 100.0 * npr / max(1, tot),
            "sim_std": float(sims.std()) if sims.size else 0.0}


def load_run_ids(p):
    """run id 文件 → 裸 uuid 集合（文件里写的是 `CVE-x/<uuid>`，遍历路径拿到的是 `<uuid>`）。"""
    out = set()
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            out.add(line.split("/", 1)[1] if "/" in line else line)
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit("用法: ab_compare_sets.py <A文件> <B文件> [A标签] [B标签]")
    A = load_run_ids(sys.argv[1])
    B = load_run_ids(sys.argv[2])
    la = sys.argv[3] if len(sys.argv) > 3 else "A"
    lb = sys.argv[4] if len(sys.argv) > 4 else "B"
    print("A=%s (%d runs)   B=%s (%d runs)" % (la, len(A), lb, len(B)))
    fpmap = fp_by_sid()
    ha, ca, ea = collect(A, fpmap)
    hb, cb, eb = collect(B, fpmap)
    ra = report(la, ha, ca, ea, fpmap)
    rb = report(lb, hb, cb, eb, fpmap)
    print("\n" + "=" * 80)
    print("### 汇总对比  A=%s  ->  B=%s" % (la, lb))
    rows = [("top10", "top-10 集中度"), ("coexist", "跨文件共现行数"), ("excess", "共现超出随机倍数"),
            ("rows", "被召回 KB 行数"), ("same_file", "同文件命中%"), ("same_proj", "同项目命中%"),
            ("admit", "晋升数"), ("ev", "evidence_hits"), ("sim_std", "相似度 σ")]
    for k, label in rows:
        a, b = ra[k], rb[k]
        if a is None or b is None:
            print("%-24s %12s %12s" % (label, a, b)); continue
        arrow = "↓" if b < a else ("↑" if b > a else "=")
        if isinstance(a, float):
            print("%-24s %12.4f %12.4f  %s" % (label, a, b, arrow))
        else:
            print("%-24s %12s %12s  %s" % (label, a, b, arrow))
    json.dump({la: ra, lb: rb}, open("/root/autodl-tmp/ab_sets.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n已写出 /root/autodl-tmp/ab_sets.json")


if __name__ == "__main__":
    sys.exit(main())
