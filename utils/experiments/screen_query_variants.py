#!/usr/bin/env python
"""C3 变体预筛工具（忠实版，0 GPU）

设计原则：**工具算的必须和生产跑的一模一样。**
  · 查询文本一律通过生产方法 `_build_query_text()` 生成，变体通过 `gap_query_source`
    开关切换 —— 预筛里测的就是将来上线跑的那条代码路径。
  · 真值校验：生产运行时会把每个查询文本的 sha1 落盘（`dump_query_vectors`）。
    本工具重建后用 sha1 **逐一比对**；覆盖率不到 100% 就说明重建不忠实，结果不可信。

为什么旧版不可信（已修正）：
  · 旧版自己手工拼 issue 字典（漏了 severity / location / tool 等字段），
    且只重建了 gap 通道、丢掉了 validation 通道 —— 两处都让查询文本与生产不同。
  · 旧版锚点核对：raw 实测 prod=0.0634（生产约 0.00）、intent_v1 实测 0.5064（生产 0.29~0.34），明显偏离。

输出：每个变体 × 每层的 查询间相似度（加偏移前/后）、前10名集中度、
      可检索到的条目数、以及**精度**（同文件命中率 / 同项目命中率，含随机基线）。
"""
import argparse
import glob
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

KB = "/root/autodl-tmp/kb_vectors.json"
OFFSET = os.path.join(ROOT, "infrastructure", "embeddings", "query_offset_transform.json")
DUMP_FOR = {
    "code_chunk": "/root/autodl-tmp/query_vectors_qoff.jsonl",
    "code_intent": "/root/autodl-tmp/query_vectors_c3.jsonl",
}
LAYERS = ["semantic", "code_pattern", "solution", "full"]
MODES = ["code_chunk", "code_intent", "code_augment"]


def l2(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def qq(M):
    M = l2(np.asarray(M, float))
    if len(M) < 3:
        return float("nan")
    iu = np.triu_indices(len(M), 1)
    return float((M @ M.T)[iu].mean())


def base_of(p):
    """取文件名。要同时兼容两类写法：
      · 数据集里的 ag 名字：`hphp__runtime__ext__ext_hash.cpp` → `ext_hash.cpp`
      · 知识库里的真实路径：`coders/tiff.c` → `tiff.c`
    """
    s = str(p or "").replace("\\", "/").strip().lower()
    b = os.path.basename(s)
    if "__" in b:
        return b.replace("__", "/").split("/")[-1]
    return b


def proj_of(p):
    """取"项目/顶层目录"。
    注意：被分析文件在产物里是**绝对路径**（/root/.../hphp__runtime__ext__ext_hash.cpp），
    直接 split("/")[0] 会得到空串 —— 这是个曾把"同项目命中率"打成 0.00% 的 bug。
    """
    s = str(p or "").replace("\\", "/").strip().lower()
    b = os.path.basename(s)
    if "__" in b:                      # ag 名字：项目 = 第一个 __ 之前的部分
        return b.split("__")[0]
    return s.split("/")[0] if "/" in s else ""


def sha16(t):
    return hashlib.sha1((t or "").encode("utf-8", "ignore")).hexdigest()[:16]


def load_run_set(path):
    return set(x.strip() for x in open(path) if x.strip()) if os.path.exists(path) else set()


def rebuild_queries(agent, runs, mode):
    """按生产代码路径重建查询文本。返回 [(text, analyzed_file, channel)]（按 sha 去重）。"""
    agent.gap_query_source = mode
    out, seen = [], set()
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        if f.split("/")[3] not in runs:
            continue
        d = json.load(open(f, encoding="utf-8"))
        # 通道 1：validation（首轮生成的问题）。生产里 original_issues 与 retrieval_evidence 一一对应
        for issue, _item in zip(d.get("original_issues") or [], d.get("retrieval_evidence") or []):
            if not isinstance(issue, dict):
                continue
            fl = str(issue.get("file") or "")
            t = agent._build_query_text(issue, fl)
            k = sha16(t)
            if k not in seen:
                seen.add(k)
                out.append((t, fl, "validation"))
        # 通道 2：gap（源码分片）
        for item in (d.get("gap_retrieval_evidence") or []):
            cc = item.get("code_chunk") or {}
            if not str(cc.get("text") or "").strip():
                continue
            issue = agent._code_chunk_as_issue(cc)
            fl = str(issue.get("file") or "")
            t = agent._build_query_text(issue, fl)
            k = sha16(t)
            if k not in seen:
                seen.add(k)
                out.append((t, fl, "gap"))
    return out


def validate(texts, mode):
    """与生产落盘的 sha1 比对，并按通道拆分覆盖率（定位是哪一路没重建对）。"""
    p = DUMP_FOR.get(mode)
    if not p or not os.path.exists(p):
        return None
    prod = set()
    for line in open(p, encoding="utf-8"):
        if line.strip():
            prod.add(json.loads(line)["text_sha1"])
    per_ch, overall = {}, {}
    for t, _, c in texts:
        k = sha16(t)
        d = per_ch.setdefault(c, {"n": 0, "hit": 0})
        d["n"] += 1
        if k in prod:
            d["hit"] += 1
    mine = set(sha16(t) for t, _, _ in texts)
    inter = mine & prod
    overall = {"mine": len(mine), "prod": len(prod), "match": len(inter),
               "recall": len(inter) / max(1, len(prod)),
               "precision": len(inter) / max(1, len(mine)), "per_channel": per_ch}
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/root/autodl-tmp/_runs_qoff2.txt")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--no-offset", action="store_true")
    args = ap.parse_args()

    runs = load_run_set(args.runs)
    print("运行组: %s （%d 个 run）" % (args.runs, len(runs)))

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent.__new__(AIDrivenSecondPassAnalysisAgent)
    agent._query_offset_cache = json.load(open(OFFSET, encoding="utf-8"))

    objs = json.load(open(KB, encoding="utf-8"))
    idx, sid_by_layer = defaultdict(list), defaultdict(list)
    for o in objs:
        p = o.get("properties") or {}
        v = (o.get("vectors") or {}).get("default") or []
        if v:
            idx[p.get("vector_layer")].append(np.asarray(v, float))
            sid_by_layer[p.get("vector_layer")].append(p.get("sqlite_id"))
    Xby = {L: l2(np.vstack(idx[L])) for L in LAYERS if idx.get(L)}

    sq = sqlite3.connect(os.path.join(ROOT, "infrastructure", "database", "mas.db"))
    fpm = {str(r[0]): (r[1] or "") for r in
           sq.execute("select id, file_pattern from issue_patterns").fetchall()}
    allfp = list(fpm.values())

    from infrastructure.embeddings.codebert_embedder import embed_text

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print("\n" + "=" * 100)
    print("第一步：忠实性校验（重建查询文本 sha1  vs  生产落盘）")
    print("=" * 100)
    rebuilt = {}
    for m in modes:
        texts = rebuild_queries(agent, runs, m)
        rebuilt[m] = texts
        v = validate(texts, m)
        ch = defaultdict(int)
        for _, _, c in texts:
            ch[c] += 1
        print("  %-14s 重建 %-5d 条 (validation=%d, gap=%d)" % (
            m, len(texts), ch["validation"], ch["gap"]))
        if v:
            ok = v["recall"] > 0.999
            print("                 生产落盘 %d 条；重合 %d → 覆盖率 %.1f%%  纯洁度 %.1f%%  %s" % (
                v["prod"], v["match"], 100 * v["recall"], 100 * v["precision"],
                "✅ 忠实（可采信）" if ok else "⚠️ 不忠实（结果不可信）"))
            for c, d in sorted(v["per_channel"].items()):
                print("                   通道 %-11s 重建 %-4d 命中落盘 %-4d → %.1f%%" % (
                    c, d["n"], d["hit"], 100 * d["hit"] / max(1, d["n"])))
        else:
            print("                 （无对应落盘文件，跳过校验）")
        if m == "code_augment":
            same = [t for t, _, c in texts if c == "gap"][:1]
            if same:
                print("                 增强样例尾部: ...%s" % repr(same[0][-160:]))

    print("\n" + "=" * 100)
    print("第二步：各变体指标（偏移 = 生产偏移，%s）" % ("关" if args.no_offset else "开"))
    print("=" * 100)
    print("%-14s %-14s %8s %8s %8s %8s %9s %10s %10s" % (
        "变体", "层", "QQ前", "QQ后", "top10", "命中行", "中位长度", "同文件%", "同项目%"))
    results = {}
    for m in modes:
        texts = [t for t, _, _ in rebuilt[m]]
        files = [fl for _, fl, _ in rebuilt[m]]
        med = int(np.median([len(t) for t in texts])) if texts else 0
        results[m] = {"n": len(texts), "median_chars": med, "layers": {}}
        for L in LAYERS:
            Z = l2(np.asarray([embed_text(t, L) for t in texts], float))
            pre = qq(Z)
            if args.no_offset:
                Zp = Z
            else:
                off = np.asarray(agent._query_offset_cache[L]["offset"], float)
                k = min(len(off), Z.shape[1])
                Zp = Z.copy()
                Zp[:, :k] = Z[:, :k] - off[:k]
                Zp = l2(Zp)
            post = qq(Zp)
            S = Zp @ Xby[L].T
            top = np.argsort(-S, axis=1)[:, :5]
            sids = sid_by_layer[L]
            N = np.zeros(S.shape[1])
            nf = npr = tot = 0
            for qi, row in enumerate(top):
                for j in set(row.tolist()):
                    N[j] += 1
                fb, fpj = base_of(files[qi]), proj_of(files[qi])
                for j in row.tolist():
                    fpx = fpm.get(str(sids[j]), "")
                    if not fpx:
                        continue
                    tot += 1
                    if fb and base_of(fpx) == fb:
                        nf += 1
                    if proj_of(fpx) and proj_of(fpx) == fpj:
                        npr += 1
            od = np.sort(N)[::-1]
            t10 = float(od[:10].sum() / max(1.0, N.sum()))
            sf = 100.0 * nf / max(1, tot)
            sp = 100.0 * npr / max(1, tot)
            results[m]["layers"][L] = {"qq_pre": pre, "qq_post": post, "top10": t10,
                                       "rows": int((N > 0).sum()),
                                       "same_file": sf, "same_proj": sp}
            print("%-14s %-14s %8.4f %8.4f %8.3f %8d %9d %9.2f%% %9.2f%%" % (
                m if L == LAYERS[0] else "", L, pre, post, t10,
                int((N > 0).sum()), med, sf, sp))
        print()

    allf = sorted({f for _, f, _ in rebuilt[modes[0]]})
    ef = float(np.mean([sum(1 for p in allfp if base_of(p) == base_of(f)) / len(allfp) for f in allf]))
    ep = float(np.mean([sum(1 for p in allfp if proj_of(p) == proj_of(f)) / len(allfp) for f in allf]))
    print("随机基线：同文件 %.2f%%   同项目 %.2f%%" % (100 * ef, 100 * ep))
    print("\n=== 四层平均汇总 ===")
    print("%-14s %8s %8s %8s %8s %10s %10s" % ("变体", "QQ前", "QQ后", "top10", "命中行", "同文件%", "同项目%"))
    for m in modes:
        rs = results[m]["layers"]
        if not rs:
            continue
        print("%-14s %8.4f %8.4f %8.3f %8.0f %9.2f%% %9.2f%%" % (
            m, np.mean([rs[L]["qq_pre"] for L in rs]), np.mean([rs[L]["qq_post"] for L in rs]),
            np.mean([rs[L]["top10"] for L in rs]), np.mean([rs[L]["rows"] for L in rs]),
            np.mean([rs[L]["same_file"] for L in rs]), np.mean([rs[L]["same_proj"] for L in rs])))

    json.dump(results, open("/root/autodl-tmp/screen_variants_v2.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n已写出 /root/autodl-tmp/screen_variants_v2.json")


if __name__ == "__main__":
    sys.exit(main())
