# -*- coding: utf-8 -*-
"""第二轮预筛：`v3_paren0` 切分 **+ 剪掉"太含糊"的针**，能不能把风险压回来？

## 这一轮要回答的问题

上一轮（`screen_needle_split.py`）测出：`v3_paren0`（只在**圆括号**深度 0 的 `;` 处切）
能把失分 case 修好、把"一条都命中不了的条目"从 35 降到 2、命中自己 +23，
**但**它把尺子放松了：跨文件命中对数 97→204、`df>=5` 占比 2.02%→3.31%，越过了风险线。

假设：**那些"在好多个文件里都出现"的针本来就没有身份信息**，剪掉它们应当能
"留住收益、去掉风险"。这一轮就是验证这个假设。

## 方法：拟合与测量必须用**不相交**的语料（否则数字是自我美化的）

| 语料 | 用途 | 组成 |
|---|---|---|
| 自有 196 个文件 | 只用于**测量**（`own_hit` 只能在自有文件上测） | 196 条知识条目对应的 CVE 目录 |
| `fit` 抽样 N 个文件 | **只用于**算 `df`、决定剪哪些针 | 从"非知识库 CVE"里随机抽 |
| `measure` 抽样 M 个文件 | **只用于**报告跨文件命中与 `df` | 从"非知识库 CVE"里随机抽，与 fit **不相交** |

脚本会断言 fit 与 measure 无交集；剪针阈值只在 fit 上定，风险指标只在 measure 上报。

## 先写死的验收口径（基准 = 现实现 `current`）

**必须全部满足**

1. `own_hit` ≥ 基准 + 5（收益不能丢）；
2. 跨文件命中对数 ≤ 基准 × 2；
3. `df>=5` 占比 ≤ 基准 × 1.5（相对线；上一轮已说明"绝对 ≤2%"这条基准自己就违反）；
4. `df==1` 占比 ≥ 70%；
5. 一条都命中不了的条目 ≤ 5（基准 35）；
6. **目标 case `CVE-2018-20854` 必须仍然被修好**（至少留一条针，且能命中）。

## 用法

    python utils/experiments/screen_needle_prune.py
"""
from __future__ import annotations

import argparse
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT  # noqa: E402
from utils.offline_imports import install_weaviate_stub  # noqa: E402
from utils.experiments.audit_clone_fragment_split import DS, payload_of  # noqa: E402
from utils.experiments.screen_needle_split import split_paren0  # noqa: E402

install_weaviate_stub()

CASE = "CVE-2018-20854"


# --------------------------------------------------------------------------- #
# 匹配：把 token 序列拼成"空格分隔的字符串"，连续子串匹配 == 连续 token 序列匹配
# （token 里不可能有空格，所以边界严格）。比自己写 n-gram/滚动哈希快得多，
# 也没有哈希碰撞隐患。下面有与生产实现 `_is_contiguous_subseq` 的一致性自检。
# --------------------------------------------------------------------------- #
def hay_of(tokens: list) -> str:
    return " " + " ".join(tokens) + " "


def needle_str(tokens) -> str:
    return " " + " ".join(tokens) + " "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--fit", type=int, default=150, help="拟合语料文件数")
    ap.add_argument("--measure", type=int, default=150, help="测量语料文件数（与拟合不相交）")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--ks", default="3,5,8,12", help="剪针阈值：df>=K 的针丢掉")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent

    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    min_tokens = agent.error_code_clone_min_tokens
    generic = agent._CODE_GENERIC_TOKENS
    tokenize = agent._tokenize_code

    con = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    rows = [(i, (t or "").strip().upper(), s or "") for i, t, s in
            con.execute("select id, title, solution from issue_patterns")]
    con.close()

    def extract(payload: str, splitter) -> list:
        out = []
        for part in splitter(payload):
            toks = tokenize(part)
            if len(toks) < min_tokens:
                continue
            if all(t.lower() in generic for t in toks):
                continue
            out.append(toks)
        return out

    current_needles, v3_needles = {}, {}
    for _id, cve, solution in rows:
        payload = payload_of(solution)
        if not payload:
            continue
        current_needles[cve] = extract(payload, lambda p: re.split(r";;", p))
        v3_needles[cve] = extract(payload, split_paren0)

    # ---- 语料（拟合与测量不相交） ------------------------------------------ #
    own = sorted(current_needles)
    all_dirs = sorted(d.name.upper() for d in (DS / "before").iterdir() if d.is_dir())
    pool = [c for c in all_dirs if c not in set(own)]
    rnd = random.Random(args.seed)
    rnd.shuffle(pool)
    fit_files = pool[: args.fit]
    measure_files = pool[args.fit: args.fit + args.measure]
    overlap = set(fit_files) & set(measure_files)
    print("语料：自有 %d（只测量）+ 拟合 %d + 测量 %d；拟合∩测量 = %d 个（必须为 0）"
          % (len(own), len(fit_files), len(measure_files), len(overlap)))
    assert not overlap, "拟合与测量语料重叠，数字会自我美化"

    raw_cache: dict = {}
    hay_cache: dict = {}

    def raw(cve: str) -> str:
        if cve not in raw_cache:
            d = DS / "before" / cve
            txt = ""
            if d.exists():
                txt = "\n".join(
                    f.read_text(encoding="utf-8", errors="ignore")
                    for f in sorted(d.rglob("*"))
                    if f.is_file() and f.suffix.lower() in SOURCE_EXT
                )
            raw_cache[cve] = txt
        return raw_cache[cve]

    def hay(cve: str) -> str:
        if cve not in hay_cache:
            txt = raw(cve)
            hay_cache[cve] = hay_of(tokenize(txt)) if txt else ""
        return hay_cache[cve]

    # ---- 自检：字符串匹配 与 生产实现 必须逐例一致 ------------------------- #
    checks, neg_ok = [], True
    for cve in own[:8]:
        toks = tokenize(raw(cve))
        for nt in v3_needles[cve][:2]:
            a = needle_str(nt) in hay(cve)
            b = agent._is_contiguous_subseq(nt, toks)
            checks.append(a == b)
    neg_ok = needle_str(["zzq_impossible_%d" % i for i in range(10)]) not in hay(own[0])
    print("\n一致性自检（字符串子串 vs 生产 _is_contiguous_subseq）：%d/%d 一致 %s"
          % (sum(checks), len(checks), "OK" if all(checks) else "*** 不一致，结果不可信 ***"))
    print("负控：人造不可能针判不命中 -> %s" % ("OK" if neg_ok else "*** 失败 ***"))

    # ---- 每根针的命中集合只算一次，各方案复用 ------------------------------ #
    def hits_over(needles: dict, files: list) -> dict:
        """(cve, idx) -> 命中的文件集合（含它自己那个文件）。"""
        out = {}
        for cve, frags in needles.items():
            h_own = hay(cve)
            for idx, nt in enumerate(frags):
                needle = needle_str(nt)
                found = {cve} if (h_own and needle in h_own) else set()
                for other in files:
                    if needle in hay(other):
                        found.add(other)
                out[(cve, idx)] = found
        return out

    print("\n在拟合语料上统计 df（%d 个文件）..." % len(fit_files))
    df_fit_v3 = {k: len(v) for k, v in hits_over(v3_needles, fit_files).items()}

    print("在测量语料上统计命中（自有 %d + 抽样 %d）..." % (len(own), len(measure_files)))
    hits_current = hits_over(current_needles, measure_files)
    hits_v3 = hits_over(v3_needles, measure_files)

    # ---- 评估各方案 -------------------------------------------------------- #
    def evaluate(label: str, needles: dict, hits: dict, keep=None) -> dict:
        own_hit, foreign, df1, df5 = set(), 0, 0, 0
        lens, impossible, n = [], set(), 0
        case_ok = False
        for cve, frags in needles.items():
            for idx, nt in enumerate(frags):
                if keep is not None and not keep(cve, idx, nt):
                    continue
                n += 1
                lens.append(len(nt))
                found = hits[(cve, idx)]
                if cve in found:
                    own_hit.add(cve)
                    if cve == CASE:
                        case_ok = True
                foreign += len(found - {cve})
                df1 += len(found) == 1
                df5 += len(found) >= 5
                if not found:
                    impossible.add(cve)
        d = n or 1
        return dict(label=label, needles=n, own_hit=len(own_hit), foreign=foreign,
                    df1_share=df1 / d, df5_share=df5 / d,
                    median_len=statistics.median(lens) if lens else 0,
                    impossible=len(impossible), case_ok=case_ok)

    results = [
        evaluate("current(基准)", current_needles, hits_current),
        evaluate("v3_paren0（不剪）", v3_needles, hits_v3),
    ]
    for k in [int(x) for x in args.ks.split(",") if x.strip()]:
        results.append(evaluate(
            "v3+df<%d（剪针）" % k, v3_needles, hits_v3,
            keep=lambda cve, idx, nt, k=k: df_fit_v3.get((cve, idx), 0) < k))

    print("\n%-18s %6s %9s %11s %9s %9s %8s %10s %8s" %
          ("方案", "针数", "命中自己", "跨文件对数", "df==1", "df>=5", "中位长", "命中不了", "case修好"))
    print("-" * 104)
    for r in results:
        print("%-18s %6d %9d %11d %8.1f%% %8.2f%% %8s %10d %8s" %
              (r["label"], r["needles"], r["own_hit"], r["foreign"],
               r["df1_share"] * 100, r["df5_share"] * 100, r["median_len"],
               r["impossible"], "是" if r["case_ok"] else "**否**"))

    # ---- 判定 -------------------------------------------------------------- #
    base = results[0]
    print("\n" + "=" * 104)
    print("按先写死的口径判定（基准 current：命中自己 %d，跨文件 %d，df>=5 %.2f%%，命中不了 %d 条）"
          % (base["own_hit"], base["foreign"], base["df5_share"] * 100, base["impossible"]))
    print("=" * 104)
    for r in results[1:]:
        checks = [
            ("1 own_hit >= 基准+5", r["own_hit"] >= base["own_hit"] + 5,
             "%d vs 基准 %d" % (r["own_hit"], base["own_hit"])),
            ("2 跨文件对数 <= 基准x2", r["foreign"] <= base["foreign"] * 2,
             "%d（%.2f 倍）" % (r["foreign"], r["foreign"] / max(1, base["foreign"]))),
            ("3 df>=5 <= 基准x1.5", r["df5_share"] <= base["df5_share"] * 1.5,
             "%.2f%% vs %.2f%%" % (r["df5_share"] * 100, base["df5_share"] * 100)),
            ("4 df==1 >= 70%", r["df1_share"] >= 0.70, "%.1f%%" % (r["df1_share"] * 100)),
            ("5 命中不了 <= 5 条", r["impossible"] <= 5, "%d 条" % r["impossible"]),
            ("6 目标 case 仍修好", r["case_ok"], "是" if r["case_ok"] else "否"),
        ]
        ok = all(c[1] for c in checks)
        print("\n%s -> %s" % (r["label"], "**通过**" if ok else "不通过"))
        for label, good, detail in checks:
            print("   [%s] %-22s %s" % ("OK" if good else "NG", label, detail))


if __name__ == "__main__":
    main()
