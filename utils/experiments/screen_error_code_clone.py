# -*- coding: utf-8 -*-
"""预筛：把"错误代码克隆"这把尺子从**分片片段**放到**整个文件**之后，到底会不会乱命中？

## 背景（问题 6/7）

`error_code_clone` 是门控里权重最高的证据（0.5 分）：把知识库那条记录里"修复前的错误代码"
切成 token 序列，看它是否**连续原样**出现在当前代码里。实测在 3.3 万条候选里只命中 1 次——
因为历史实现只在"约 5 行的代码片段"里找。

把搜索范围扩大到整个文件，命中率必然上升。但**上升的是"对的命中"还是"错的命中"**，
必须在跑 GPU 之前先用离线实验问清楚，否则可能一边提高召回、一边把误报也抬起来。

## 这个实验怎么设计

做一个 400 × 200 的命中矩阵，每一格回答："第 i 个样本文件里，能不能找到第 j 条知识记录的
错误代码？"

    · 对角线（样本自己那条记录）→ 这叫**该命中的**（召回的上限收益）
    · 非对角线（别人的记录）    → 这叫**不该命中的**（误报的风险）

关键指标是**非对角线的命中率**：如果它跟对角线一样高，说明这把尺子分不清敌我，扩大范围
只会伤到自己；如果它很低，说明"连续 4 个以上 token 原样出现"确实是个强判据。

再单独看一个**最危险的子集**：知识条目的文件名 basename 恰好出现在样本文件路径里
（例如记录讲 `inode.c`，样本是 `fs/udf/inode.c`）。门控里的 `file_basename_anchor`
在这种情况会给出 0.2 分锚点，于是"异文件候选"能绕过跨文件拦截——所以这一子集的
命中率决定了扩大范围会不会**打开跨文件闸门**。

最后给出 `min_tokens`（最少连续 token 数）的权衡曲线，为"要不要调严"提供依据。

## 用法（MAS 根目录，本地可跑，不需要 GPU / 向量库）

    python utils/experiments/screen_error_code_clone.py
    python utils/experiments/screen_error_code_clone.py --csv reports/clone_prescreen.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT, normalize_key  # noqa: E402
from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()   # 本机没装 weaviate 时也能离线跑（门控逻辑不碰向量库）

DB = ROOT / "infrastructure/database/mas.db"
DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
MANIFEST = ROOT / "reports/negative_exp_manifest_400_error.json"


def sample_source_text(cve: str, ds_root: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    """样本的源码正文（把该样本目录下的源码文件拼起来）。

    与门控 `_resolve_current_code` 的口径一致：**整个文件**，不是片段。
    大小上限也与它一致（`_CURRENT_CODE_MAX_BYTES` = 4 MB）——超过上限的文件在线上会
    退回"分片片段"，预筛必须用同一个上限，否则会高估这把尺子的可得性。
    """
    d = ds_root / "before" / cve
    if not d.exists():
        return ""
    buf = []
    for f in sorted(d.rglob("*")):
        try:
            if f.is_file() and f.suffix.lower() in SOURCE_EXT:
                if f.stat().st_size > max_bytes:
                    continue
                buf.append(f.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(buf)


def make_fast_clone(agent, toks: List[str]):
    """给一个文件构造"快速版"克隆判定，**语义与线上完全一致**。

    线上 `_is_contiguous_subseq` 是朴素实现（逐个起点比较切片），对 400×200 的命中矩阵
    太慢（要跑好几个小时）。这里用"先按前 4 个 token 建索引、命中后再逐点核对"做等价加速：

      · 长度 ≥4 的片段：前 4 个 token 的 4-gram 命中 → 才去那几处核对整段
      · 片段长度必须 ≥ min_tokens(=4)，所以 4-gram 一定存在

    **等价性必须实测**，不能假设（历史上"复现脚本自己写一份"就踩过坑）——
    调用方 `assert_fast_equals_slow()` 会随机抽样本对照线上实现。
    """
    idx = {}
    for i in range(len(toks) - 3):
        idx.setdefault((toks[i], toks[i + 1], toks[i + 2], toks[i + 3]), []).append(i)

    def fast(solution: str) -> bool:
        frags = agent._extract_error_code_fragments(solution)
        if not frags:
            return False
        for f in frags:
            if len(f) < 4:
                # 线上的 min_tokens 默认为 4；若被调小，退回朴素实现以保证等价
                if agent._is_contiguous_subseq(f, toks):
                    return True
                continue
            key = (f[0], f[1], f[2], f[3])
            for pos in idx.get(key, ()):  # 只在候选位置核对整段
                if toks[pos:pos + len(f)] == f:
                    return True
        return False

    return fast


def assert_fast_equals_slow(agent, solutions, toks_by_cve, files, n: int = 300) -> None:
    """随机抽查：快速版与线上实现必须逐例一致，否则整个预筛结论无效。"""
    import random

    rng = random.Random(20240930)
    cves = [c for c in toks_by_cve if files.get(c)]
    checked = mismatch = 0
    for _ in range(n):
        cve = rng.choice(cves)
        sol = rng.choice(solutions)
        toks = toks_by_cve[cve]
        slow = agent._error_code_clone_matched_tokens(sol, toks)
        fast = make_fast_clone(agent, toks)(sol)
        checked += 1
        if slow != fast:
            mismatch += 1
    print("  [等价性自检] 快速版 vs 线上实现：抽查 %d 例，不一致 %d 例 %s"
          % (checked, mismatch, "✅" if mismatch == 0 else "❌ 结论不可用"))
    if mismatch:
        raise SystemExit("快速版与线上实现不等价，预筛结论无效")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--dataset-root", type=Path, default=DS)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--min-tokens", type=int, nargs="*", default=[4, 5, 6, 8],
                    help="要比较的『最少连续 token 数』")
    ap.add_argument("--max-samples", type=int, default=80,
                    help="抽样多少个样本建矩阵（kb/held 各半）。全矩阵 400×200 太慢，"
                         "而这里问的是『比率』，分层抽样足够")
    ap.add_argument("--seed", type=int, default=20240930)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    # 直接复用被测实现，保证预筛与线上判定是同一套逻辑（历史上"复现脚本自己写一份"踩过坑）
    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    con = sqlite3.connect(str(args.db))
    kb = [{"id": r[0], "cve": str(r[1] or "").strip().upper(),
           "file_pattern": str(r[2] or ""), "solution": str(r[3] or "")}
          for r in con.execute("select id, title, file_pattern, solution from issue_patterns")]
    con.close()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))["rows"]
    samples = [{"cve": r["cve"], "role": r["role"]} for r in rows]
    # 分层抽样：kb / held 各取一半（两群的"文件撞车"概率差别很大，不能只抽一边）
    import random
    rng = random.Random(args.seed)
    kb_pool = [s for s in samples if s["role"] == "kb"]
    held_pool = [s for s in samples if s["role"] == "held"]
    half = max(1, args.max_samples // 2)
    if args.max_samples < len(samples):
        samples = (rng.sample(kb_pool, min(half, len(kb_pool)))
                   + rng.sample(held_pool, min(half, len(held_pool))))
    print("知识库 %d 条，样本 %d 个（kb %d / held %d）%s"
          % (len(kb), len(samples),
             sum(1 for s in samples if s["role"] == "kb"),
             sum(1 for s in samples if s["role"] == "held"),
             "（分层抽样，seed=%d）" % args.seed if len(samples) < len(rows) else ""))

    # 先看有多少条知识记录真的带得动这把尺子
    with_frag = [k for k in kb if agent._extract_error_code_fragments(k["solution"])]
    nfrag = [len(agent._extract_error_code_fragments(k["solution"])) for k in with_frag]
    print("含『Remove incorrect logic』错误代码片段的知识记录: %d / %d；每条的片段数 "
          "中位 %d 最多 %d" % (len(with_frag), len(kb),
                            sorted(nfrag)[len(nfrag) // 2] if nfrag else 0,
                            max(nfrag) if nfrag else 0))
    if not with_frag:
        raise SystemExit("没有任何知识记录带错误代码片段 → 这把尺子在知识库上根本用不了")

    # 建立矩阵（token 化每个样本文件只做一次）
    names = sorted({t for t in args.min_tokens})
    files, file_keys, file_bases, toks_by_cve = {}, {}, {}, {}
    print("\n正在读取并 token 化 %d 个样本文件..." % len(samples))
    for s in samples:
        cve = s["cve"]
        text = sample_source_text(cve, args.dataset_root)
        files[cve] = bool(text)
        toks_by_cve[cve] = agent._tokenize_code(text)
        main = sorted({Path(f).name for f in
                       [str(p) for p in (args.dataset_root / "before" / cve).rglob("*")]
                       if Path(f).suffix.lower() in SOURCE_EXT})
        file_keys[cve] = {normalize_key(n, 2) for n in main}
        file_bases[cve] = {agent._normalize_source_basename(n) for n in main}
    missing = [c for c in files if not files[c]]
    print("  可读样本 %d 个；读不到源码的 %d 个" % (len(samples) - len(missing), len(missing)))

    results = {}
    own_by_cve = {k["cve"]: k for k in kb}
    # 等价性自检：快速版必须与线上实现逐例一致，否则后面所有数字都不作数
    assert_fast_equals_slow(agent, [k["solution"] for k in with_frag], toks_by_cve, files)

    for mt in names:
        agent.error_code_clone_min_tokens = mt
        fast_by_cve = {cve: make_fast_clone(agent, toks_by_cve[cve])
                       for cve in files if files[cve]}
        diag_hit = diag_n = 0                # 对角线：样本自己那条记录
        off_hit = off_n = 0                  # 非对角线且文件不同：不该命中
        base_off_hit = base_off_n = 0        # 上面里面"basename 撞车"的危险子集
        sf_hit = sf_n = 0                    # 文件相同、但不是自己那条（兄弟条目）
        per_cve_own = {}
        for s in samples:
            cve = s["cve"]
            fast = fast_by_cve.get(cve)
            if fast is None:
                continue
            keys = file_keys[cve]
            bases = file_bases[cve]
            own = own_by_cve.get(cve)
            for k in kb:
                same_file = bool(keys & {normalize_key(k["file_pattern"], 2)})
                hit = fast(k["solution"])
                if own and k["id"] == own["id"]:
                    diag_n += 1
                    diag_hit += int(hit)
                    per_cve_own[cve] = hit
                elif same_file:
                    sf_n += 1
                    sf_hit += int(hit)
                else:
                    off_n += 1
                    off_hit += int(hit)
                    if bases & {agent._normalize_source_basename(k["file_pattern"])}:
                        base_off_n += 1
                        base_off_hit += int(hit)
        results[mt] = {"diag_hit": diag_hit, "diag_n": diag_n,
                       "off_hit": off_hit, "off_n": off_n,
                       "base_off_hit": base_off_hit, "base_off_n": base_off_n,
                       "sf_hit": sf_hit, "sf_n": sf_n,
                       "own_hit": sum(1 for v in per_cve_own.values() if v),
                       "own_n": len(per_cve_own)}

    print("\n" + "=" * 104)
    print("错误代码克隆：扩大搜索范围到『整个文件』之后，该命中的与不该命中的")
    print("=" * 104)
    print("  %-9s %-21s %-23s %-22s %s" % (
        "最少token", "对角线（该命中）", "同文件·非自己条目", "异文件（不该命中）",
        "其中 basename 撞车"))
    for mt in names:
        r = results[mt]
        print("  %-9d %5d/%-4d=%5.1f%%  %6d/%-6d=%5.1f%%  %7d/%-7d=%5.2f%%  %6d/%-6d=%5.2f%%" % (
            mt, r["diag_hit"], r["diag_n"], 100 * r["diag_hit"] / max(1, r["diag_n"]),
            r["sf_hit"], r["sf_n"], 100 * r["sf_hit"] / max(1, r["sf_n"]),
            r["off_hit"], r["off_n"], 100 * r["off_hit"] / max(1, r["off_n"]),
            r["base_off_hit"], r["base_off_n"], 100 * r["base_off_hit"] / max(1, r["base_off_n"])))
    print("\n  注：对角线只统计『样本自己那条记录』；『同文件·非自己条目』就是"
          "held-same-file 的机理（同一个文件、库里的记录挂着别的 CVE）；")
    print("      『异文件』是主要风险面——它们的命中会被跨文件规则拦掉**除非**同时拿到"
          "file_basename_anchor（最后一列）。")

    print("\n" + "=" * 96)
    print("怎么读这张表（判据）")
    print("=" * 96)
    for mt in names:
        r = results[mt]
        d = r["diag_hit"] / max(1, r["diag_n"])
        o = r["off_hit"] / max(1, r["off_n"])
        print("  最少 %d token：该命中 %.1f%%  不该命中 %.2f%%  区分度（倍数）%s"
              % (mt, 100 * d, 100 * o, ("%.0f×" % (d / o)) if o else "∞"))

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["min_tokens", "diag_hit", "diag_n", "diag_rate",
                        "off_hit", "off_n", "off_rate",
                        "base_off_hit", "base_off_n", "base_off_rate"])
            for mt in names:
                r = results[mt]
                w.writerow([mt, r["diag_hit"], r["diag_n"],
                            round(100 * r["diag_hit"] / max(1, r["diag_n"]), 2),
                            r["off_hit"], r["off_n"],
                            round(100 * r["off_hit"] / max(1, r["off_n"]), 3),
                            r["base_off_hit"], r["base_off_n"],
                            round(100 * r["base_off_hit"] / max(1, r["base_off_n"]), 3)])
        print("\n明细已写出: %s" % args.csv)

    out = ROOT / "reports/clone_prescreen.json"
    out.write_text(json.dumps({"results": {str(k): v for k, v in results.items()},
                               "kb_entries_with_fragments": len(with_frag),
                               "kb_entries_total": len(kb)},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("汇总已写出: %s" % out)


if __name__ == "__main__":
    main()
