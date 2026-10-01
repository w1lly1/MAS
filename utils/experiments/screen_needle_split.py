# -*- coding: utf-8 -*-
"""预筛：把「错误代码片段」的切分从 `;;` 改成语句边界，收益和风险各是多少？

**这一步只测量，不改生产代码。** 改切分器要等这份数字说话。

## 背景（为什么现在的切分有问题）

`_extract_error_code_fragments()` 用 `re.split(r";;", payload)` 切 `solution` 里
"Remove incorrect logic: …" 那段。但数据集里的 payload 常常是**用单个分号**把两行连起来的：

    Remove incorrect logic: for (i = 0; i <= SERDES_MAX; i++) {; for (i = 0; i <= SERDES_MAX; i++) {.

于是两行被**粘成一条长针**。这条针要"连续、原样"出现在受检文件里才算命中，
而两处代码在文件里隔着几百个 token —— **永远不可能命中**。后果有两个方向：

* **证据侧**：`error_code_clone`（权重 0.5）拿不到，curated 通道被封顶在 0.4（低于 0.45 门限）；
* **否决侧**：`_candidate_code_fixed()` 判"错误代码已经找不到了" ⇒ 认定**已修复** ⇒ 误杀召回。

## 三种切法（本脚本对比）

| 代号 | 规则 | 特点 |
|---|---|---|
| `current` | `re.split(r";;")` | 现实现 |
| `v1_keyword` | 在 `;` 后面紧跟语句关键字（for/if/while/return/…）处切 | 保守：只切开"明显是两条语句"的地方 |
| `v2_depth0` | 在**括号深度为 0** 且不在字符串/字符字面量里的 `;` 处切 | 激进：切出更多、更短的针 |

片段过滤沿用**生产实现**（`_tokenize_code` + `error_code_clone_min_tokens` + 通用 token 过滤），
不另写一套。

## 先写死的验收口径（先定标准，再看数字）

**收益**
1. `own_hit`（至少一条针能命中"它自己来的那个文件"的条目数）比现实现**增加 ≥ 5 条**；
2. 现实现里"有针但一条都命中不了自己文件"的条目应当被修好。

**风险（任一条不满足就不接受该切法）**
3. 跨文件误命中（针命中了**别人**文件的 (条目,文件) 对数）相对现实现**不超过 2 倍**；
4. `df>=5`（在 ≥5 个文件里出现）的针占全部针的比例 **≤ 2%**；
5. `df==1`（只在自己那个文件出现，最有判别力）的针占比 **≥ 70%**；
6. 针长度中位数 **≥ 6 个 token**。

## 用法

    python utils/experiments/screen_needle_split.py
    python utils/experiments/screen_needle_split.py --extra 600    # 加大语料，df 更稳
"""
from __future__ import annotations

import argparse
import random
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT  # noqa: E402
from utils.offline_imports import install_weaviate_stub  # noqa: E402
from utils.experiments.audit_clone_fragment_split import DS, payload_of  # noqa: E402

install_weaviate_stub()

KEYWORDS = (
    "for|if|while|return|switch|case|do|else|break|continue|goto|"
    "int|char|void|short|long|float|double|unsigned|signed|struct|union|enum|"
    "static|const|size_t|uint8_t|uint16_t|uint32_t|uint64_t|int32_t|int64_t"
)
V1_RE = re.compile(r";\s*(?=(?:%s)\b)" % KEYWORDS)


def split_depth0(payload: str) -> list:
    """按「括号深度为 0 且不在字符串/字符字面量里的 `;`」切分。

    注意：这里把 `{ } [ ]` 也算进深度，所以 `{; for (...)` 这种"两行被 `;` 连接、
    且前一行以 `{` 结尾"的情况**切不开** —— 实测它对 CVE-2018-20854 无效（见主测量）。
    """
    parts, buf, depth, quote = [], [], 0, ""
    i, n = 0, len(payload)
    while i < n:
        ch = payload[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(payload[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == ";" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def split_paren0(payload: str) -> list:
    """只跟踪**圆括号**深度（`{}`/`[]` 不算）的 `;` 切分。

    为什么只算圆括号：
    * `for (i = 0; i < n; i++)` 里的分号在圆括号内 ⇒ 不会被切（这是必须的）；
    * 而两行被 `;` 连接时，前一行常以 `{` 结尾（`... ) {; for (...`），
      若把 `{}` 也算进深度，这个 `;` 就切不开 —— 恰恰是我们最想修的那一类。
    """
    parts, buf, depth, quote = [], [], 0, ""
    i, n = 0, len(payload)
    while i < n:
        ch = payload[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(payload[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == ";" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


VARIANTS = {
    "current": (lambda p: re.split(r";;", p), None),
    "v1_keyword": (lambda p: V1_RE.split(p), None),
    "v2_depth0": (split_depth0, None),
    "v3_paren0": (split_paren0, None),
    # 旋钮实验：把最短针长从 4 提到 6，看能不能把"针太短 ⇒ 判别力下降"的风险压回去
    "v3_paren0_min6": (split_paren0, 6),
}


def ngram_set(tokens: list, n: int) -> set:
    """文件里所有长度 n 的连续 token 元组（C 层切片+zip，比 Python 循环快得多）。"""
    if n <= 0 or n > len(tokens):
        return set()
    return set(zip(*(tokens[i:] for i in range(n))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--extra", type=int, default=200, help="额外抽多少个文件来估 df")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--max-tokens", type=int, default=400_000, help="超大文件跳过（保护运行时间）")
    ap.add_argument("--case", default="CVE-2018-20854")
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

    # ---- 每条目 × 每种切法，抽针（过滤沿用生产实现，除非该变体自带门槛） --- #
    per_entry = {}
    for _id, cve, solution in rows:
        payload = payload_of(solution)
        if not payload:
            continue
        variants = {}
        for name, (fn, min_override) in VARIANTS.items():
            floor = min_override or min_tokens
            frags = []
            for part in fn(payload):
                toks = tokenize(part)
                if len(toks) < floor:
                    continue
                if all(t.lower() in generic for t in toks):
                    continue
                frags.append(toks)
            variants[name] = frags
        per_entry[cve] = variants

    needles = {name: [] for name in VARIANTS}          # [(src_cve, idx, tokens)]
    for cve, variants in per_entry.items():
        for name, frags in variants.items():
            for idx, nt in enumerate(frags):
                needles[name].append((cve, idx, nt))

    lengths = {len(nt) for name in needles for _, _, nt in needles[name]}
    print("条目 %d；参与统计的针长度 %d 种（%d..%d）"
          % (len(per_entry), len(lengths), min(lengths), max(lengths)))
    for name in VARIANTS:
        lens = [len(nt) for _, _, nt in needles[name]]
        print("  切法 %-10s 针数 %4d  针长 中位 %-5s 最短 %-3s 最长 %-4s"
              % (name, len(lens), statistics.median(lens) if lens else "-",
                 min(lens) if lens else "-", max(lens) if lens else "-"))

    # ---- 语料 -------------------------------------------------------------- #
    own_cves = list(per_entry)
    all_dirs = sorted(d.name.upper() for d in (DS / "before").iterdir() if d.is_dir())
    rnd = random.Random(args.seed)
    extra = [c for c in all_dirs if c not in set(own_cves)]
    rnd.shuffle(extra)
    corpus = own_cves + extra[: args.extra]
    print("语料：自有 %d + 抽样 %d = %d 个文件\n" % (len(own_cves), len(corpus) - len(own_cves), len(corpus)))

    token_cache: dict = {}

    def cve_tokens(cve: str) -> list:
        if cve not in token_cache:
            d = DS / "before" / cve
            txt = ""
            if d.exists():
                txt = "\n".join(
                    f.read_text(encoding="utf-8", errors="ignore")
                    for f in sorted(d.rglob("*"))
                    if f.is_file() and f.suffix.lower() in SOURCE_EXT
                )
            token_cache[cve] = tokenize(txt) if txt else []
        return token_cache[cve]

    # ---- 控制组：先证明这套测量能失败、也能命中 --------------------------- #
    print("=" * 96)
    print("控制组")
    print("=" * 96)
    ref_toks = cve_tokens(own_cves[0])
    impossible = ["zzq_impossible_%d" % i for i in range(12)]
    ok_neg = tuple(impossible) not in ngram_set(ref_toks, 12)
    print("  负控：人造不可能针（12 个乱造 token）判不命中 -> %s" % ("OK" if ok_neg else "*** 失败 ***"))
    if needles["current"]:
        src0, _, nt0 = needles["current"][0]
        ok_pos = tuple(nt0) in ngram_set(cve_tokens(src0), len(nt0))
        print("  正控：现实现第一条针 %-16s 命中它自己的文件 -> %s"
              % (src0, "命中" if ok_pos else "不命中（该条可能就是粘针）"))

    print("\n  单例细看 %s（冒烟里唯一失分的样本）:" % args.case)
    case_toks = cve_tokens(args.case.upper())
    for name in VARIANTS:
        frags = per_entry.get(args.case.upper(), {}).get(name, [])
        marks = []
        for nt in frags:
            marks.append("%d token %s" % (len(nt), "命中" if tuple(nt) in ngram_set(case_toks, len(nt)) else "**不命中**"))
        print("    %-11s %s" % (name, " | ".join(marks) if marks else "（没有片段）"))

    # ---- 主测量 ------------------------------------------------------------ #
    print("\n" + "=" * 96)
    print("主测量：每种切法的针在整个语料里的命中情况")
    print("=" * 96)
    hits = {name: defaultdict(set) for name in VARIANTS}   # (src,idx) -> {cve}
    skipped = []
    for cve in corpus:
        toks = cve_tokens(cve)
        if not toks:
            continue
        if len(toks) > args.max_tokens:
            skipped.append(cve)
            continue
        cache_sets = {}
        for name in VARIANTS:
            for src, idx, nt in needles[name]:
                n = len(nt)
                if n not in cache_sets:
                    cache_sets[n] = ngram_set(toks, n)
                if tuple(nt) in cache_sets[n]:
                    hits[name][(src, idx)].add(cve)
    if skipped:
        print("  （跳过 %d 个超大文件：%s）" % (len(skipped), ", ".join(skipped[:5])))

    print("\n%-11s %6s %9s %10s %12s %9s %10s %8s" %
          ("切法", "针数", "覆盖条目", "命中自己", "跨文件对数", "df==1", "df>=5", "中位长"))
    print("-" * 96)
    summary = {}
    for name in VARIANTS:
        n_all = needles[name] or [(None, None, [])]
        own_hit, foreign, df1, df5 = set(), 0, 0, 0
        for src, idx, nt in needles[name]:
            files = hits[name].get((src, idx), set())
            df = len(files)
            if src in files:
                own_hit.add(src)
            foreign += len(files - {src})
            df1 += df == 1
            df5 += df >= 5
        count = len(needles[name]) or 1
        med = statistics.median([len(nt) for _, _, nt in needles[name]]) if needles[name] else 0
        summary[name] = dict(needles=len(needles[name]), own_hit=len(own_hit), foreign=foreign,
                             df1=df1 / count, df5=df5 / count, med=med)
        print("%-11s %6d %9d %10d %12d %8.1f%% %9.2f%% %8s" %
              (name, summary[name]["needles"], len(per_entry), len(own_hit), foreign,
               summary[name]["df1"] * 100, summary[name]["df5"] * 100, med))

    # ---- 按先写死的口径判定 ------------------------------------------------ #
    print("\n" + "=" * 96)
    print("按先写死的口径判定（基准 = current）")
    print("=" * 96)
    base = summary["current"]
    print("基准 current：own_hit %d，跨文件对数 %d，df>=5 %.2f%%，df==1 %.1f%%\n"
          % (base["own_hit"], base["foreign"], base["df5"] * 100, base["df1"] * 100))
    for name in VARIANTS:
        if name == "current":
            continue
        s = summary[name]
        checks = [
            ("1 own_hit 增加 >=5 条", s["own_hit"] - base["own_hit"] >= 5,
             "%+d（%d -> %d）" % (s["own_hit"] - base["own_hit"], base["own_hit"], s["own_hit"])),
            ("3 跨文件对数 <= 基准 2 倍", s["foreign"] <= max(1, base["foreign"]) * 2,
             "%d vs 基准 %d（%.2f 倍）" % (s["foreign"], base["foreign"],
                                            s["foreign"] / max(1, base["foreign"]))),
            ("4(df>=5) 占比 <=2%", s["df5"] <= 0.02, "%.2f%%" % (s["df5"] * 100)),
            ("4b(df>=5) 不超过基准 1.5 倍", s["df5"] <= base["df5"] * 1.5,
             "%.2f%% vs 基准 %.2f%%" % (s["df5"] * 100, base["df5"] * 100)),
            ("5 df==1 占比 >=70%", s["df1"] >= 0.70, "%.1f%%" % (s["df1"] * 100)),
            ("6 针长中位数 >=6", s["med"] >= 6, str(s["med"])),
        ]
        hard = [c for c in checks if not c[0].startswith("4b")]
        verdict = "接受" if all(c[1] for c in hard) else "不接受"
        print("\n%s -> %s" % (name, verdict))
        for label, ok, detail in checks:
            print("   [%s] %-28s %s" % ("OK" if ok else "NG", label, detail))

    # ---- 现实现下"一条都命中不了自己文件"的条目 --------------------------- #
    print("\n" + "=" * 96)
    print("『有针、却一条都命中不了任何文件』的条目（= 不可能命中的长针）")
    print("=" * 96)
    for name in VARIANTS:
        bad = sorted({src for src, idx, _ in needles[name] if not hits[name].get((src, idx))})
        print("  %-11s %2d 条：%s" % (name, len(bad), ", ".join(bad[:12]) or "（无）"))


if __name__ == "__main__":
    main()
