# -*- coding: utf-8 -*-
"""③ 知识库重建（路线 A：**只改数据，不改运行时代码**）—— 先干跑。

## 这份脚本做什么

在一份**数据库副本**上算出重建后的样子，并打印"到底会改多少、影响哪些层的向量"。
**绝不碰线上那份库**（脚本会断言 out != source）。

四项重建内容：

1. **needle 片段重排**（《01》问题 4 之二，今天已预筛定案）：
   用 `v3_paren0`（只在**圆括号**深度 0 的 `;` 处切）切 `Remove incorrect logic:` 那段，
   再剪掉"在 ≥K 个文件里都出现过"的针（K 默认 5，拟合语料 = 数据集中**非知识库**的 CVE 目录，
   与知识库 200 条不相交）。
   **关键**：重排后把保留片段用 `;;` 连接写回去 —— 运行时的 `;;` 切分正好切出这批片段，
   所以**生产代码一行不改**。
2. **写入 `llm_semantic`**（索引侧语义文本，来自 `reports/ls_index_v2.json`）。
3. **`error_type` 重算**（用修好的规则 + 数据集 metadata 的 cwe/分类/摘要）——可选，`--apply-error-type`。
4. **`problematic_pattern` 跟随 `error_type` 重算**（它与分类同源）。

同时报告 **`class_pattern` 是否只是"旧规则算漏了"**：重算一遍看有多少条能从空变成非空
（如果一条都变不了，说明空是因为摘要里根本没有函数名 —— 那是"原料问题"，得换来源，不是重算能救的）。

## 为什么要先干跑

重建会**改变索引文本**，进而让对应层的向量失效、旧基线作废。所以先算清楚**爆炸半径**：
每个条目有几层的层文本会变、总共要重嵌多少条向量。看清楚再动手。

## 用法

    python utils/experiments/rebuild_kb_candidate.py                # 干跑（只写副本 + 打印报告）
    python utils/experiments/rebuild_kb_candidate.py --apply-error-type
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT  # noqa: E402
from utils.offline_imports import install_weaviate_stub  # noqa: E402
from utils.experiments.audit_clone_fragment_split import DS, payload_of  # noqa: E402
from utils.experiments.screen_needle_split import split_paren0  # noqa: E402

install_weaviate_stub()

LAYERS = ("semantic", "code_pattern", "solution", "full")
PAYLOAD_RE = re.compile(r"Remove incorrect logic:\s*(.+?)(\.\s*Ensure corrected path:|$)", re.DOTALL)


def rewrite_solution_payload(solution, *, tokenize, min_tokens, generic, hay_of, fit_cves, min_df):
    """把 `Remove incorrect logic:` 那段重排成"保留片段用 `;;` 连接"。

    **这是唯一实现，两张表都走它**：同一份 payload 在 `issue_patterns.solution` 与
    `curated_issues.solution` 里**各存了一份**，只修一张表等于没修 ——
    实测就是这样：针修好了，但门控读的是 curated 那份旧文本，端到端零效果。

    规则：`v3_paren0` 切分（只在**圆括号**深度 0 的 `;` 处切）→ 过滤太短的/全通用词的片段
    → 剪掉"在 >=min_df 个文件里都出现"的针 → 保留片段用 `;;` 连回去
    （运行时现有的 `;;` 切分正好切出这批片段，**生产代码一行不用改**）。

    返回 (new_solution, kept_texts, dropped[(text, df)])。取不到 payload 或保留为空时原样返回。
    """
    payload = payload_of(solution)
    if not payload:
        return solution, [], []
    frags, seen = [], set()
    for part in split_paren0(payload):
        p = part.strip()
        if not p or p in seen:
            continue
        seen.add(p)
        toks = tokenize(p)
        if len(toks) < min_tokens:
            continue
        if all(t.lower() in generic for t in toks):
            continue
        frags.append((p, toks))
    kept, dropped = [], []
    for text, toks in frags:
        needle = " " + " ".join(toks) + " "
        df = sum(1 for c in fit_cves if needle in hay_of(c))
        if df >= min_df:
            dropped.append((text, df))
        else:
            kept.append(text)
    if not kept:
        return solution, [], dropped
    m = PAYLOAD_RE.search(solution)
    if not m:
        return solution, kept, dropped
    suffix = m.group(2) or ""
    rebuilt = "Remove incorrect logic: %s%s" % (";;".join(kept), suffix)
    return solution[:m.start()] + rebuilt + solution[m.end():], kept, dropped


def payload_selfcheck(new_solution, *, tokenize, min_tokens, generic, agent):
    """自检：重排后用**生产实现**重抽的针，是否等于"我们打算保留的那批"。

    必须用同一个 payload 提取规则（`payload_of`）取重排后那段 —— 拿整条 solution 去 split
    会把 ". Ensure corrected path: …" 也算成片段，把自检自己搞失败（第一版就这么错的）。
    """
    intended = []
    for part in payload_of(str(new_solution)).split(";;"):
        toks = tuple(tokenize(part))
        if len(toks) < min_tokens:
            continue
        if all(x.lower() in generic for x in toks):
            continue
        intended.append(toks)
    got = [tuple(t) for t in agent._extract_error_code_fragments(str(new_solution))]
    return sorted(intended) == sorted(got), len(intended), len(got)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/mas_rebuild_candidate.db")
    ap.add_argument("--dataset-root", type=Path, default=DS)
    ap.add_argument("--index-semantic", type=Path, default=ROOT / "reports/ls_index_v2.json")
    ap.add_argument("--needle-min-df", type=int, default=5, help="在 >=K 个文件里出现的针剪掉")
    ap.add_argument("--apply-error-type", action="store_true",
                    help="同时把 error_type / problematic_pattern 按修好的规则重算")
    ap.add_argument("--show", type=int, default=8, help="打印多少条明细")
    ap.add_argument("--include-curated", action="store_true",
                    help="同时重排 curated_issues.solution（**同一份 payload 的第二份副本**；"
                         "不修它的话，curated 通道的候选仍会拿旧粘针去判『已修复』）")
    ap.add_argument("--report", type=Path, default=ROOT / "reports/kb_rebuild_dryrun.json")
    args = ap.parse_args()

    assert args.out.resolve() != args.db.resolve(), "输出不能覆盖源库（干跑只写副本）"

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    from infrastructure.database.weaviate.service import WeaviateVectorService
    from utils.bigvul_ingest.rules import (
        derive_error_type, derive_problematic_pattern, extract_function_name_from_summary,
    )

    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    svc = WeaviateVectorService()
    min_tokens = agent.error_code_clone_min_tokens
    generic = agent._CODE_GENERIC_TOKENS
    tokenize = agent._tokenize_code

    idx_sem = json.loads(args.index_semantic.read_text(encoding="utf-8"))

    con = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    cols = [r[1] for r in con.execute("pragma table_info(issue_patterns)")]
    rows = [dict(zip(cols, r)) for r in con.execute("select * from issue_patterns")]
    con.close()
    print("源库 %s：%d 条，列 %s" % (args.db.name, len(rows), cols))

    # ---- 拟合语料：数据集中「非知识库」的 CVE 目录（与库里 200 条不相交） ---- #
    kb_cves = {(r.get("title") or "").strip().upper() for r in rows}
    pool = [d.name.upper() for d in (args.dataset_root / "before").iterdir() if d.is_dir()]
    fit_cves = [c for c in pool if c not in kb_cves]
    print("拟合语料（用于算 df）：%d 个非知识库 CVE 目录（库内 %d 条不参与）" % (len(fit_cves), len(kb_cves)))

    hay_cache: dict = {}

    def hay(cve: str) -> str:
        if cve not in hay_cache:
            d = args.dataset_root / "before" / cve
            txt = ""
            if d.exists():
                txt = "\n".join(f.read_text(encoding="utf-8", errors="ignore")
                                for f in sorted(d.rglob("*"))
                                if f.is_file() and f.suffix.lower() in SOURCE_EXT)
            hay_cache[cve] = (" " + " ".join(tokenize(txt)) + " ") if txt else ""
        return hay_cache[cve]

    def meta(cve: str) -> dict:
        p = args.dataset_root / "metadata" / cve / "cve_metadata.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ---- 逐条：切分 + 剪针 + 重排 ------------------------------------------ #
    stat = Counter()
    examples = []
    layer_blast = Counter()          # 有几层的层文本会变
    new_rows = []
    for row in rows:
        cve = (row.get("title") or "").strip().upper()
        solution = str(row.get("solution") or "")
        props_old = dict(row)
        props_new = dict(row)

        # (1) needle 重排（与 curated_issues 共用同一个实现）
        changed_solution, _kept, dropped = rewrite_solution_payload(
            solution,
            tokenize=tokenize, min_tokens=min_tokens, generic=generic,
            hay_of=hay, fit_cves=fit_cves, min_df=args.needle_min_df,
        )
        if payload_of(solution):
            stat["payload_entries"] += 1
            stat["needles_dropped"] += len(dropped)
        if changed_solution != solution:
            stat["solution_changed"] += 1
            if len(examples) < args.show:
                examples.append({"cve": cve, "dropped": [(t[:60], d) for t, d in dropped[:3]]})
        props_new["solution"] = changed_solution

        # (2) llm_semantic
        sem = idx_sem.get(cve, "")
        if sem:
            props_new["llm_semantic"] = sem
            stat["llm_semantic_set"] += 1

        # (3) error_type / problematic_pattern（可选）
        m = meta(cve)
        if args.apply_error_type and m:
            new_et = derive_error_type(str(m.get("cwe_id") or ""),
                                       str(m.get("vulnerability_classification") or ""),
                                       str(m.get("summary") or row.get("error_description") or ""))
            if new_et != row.get("error_type"):
                stat["error_type_changed"] += 1
            props_new["error_type"] = new_et
            props_new["problematic_pattern"] = derive_problematic_pattern(
                new_et, str(m.get("summary") or row.get("error_description") or ""))
        # class_pattern 是否只是"旧规则算漏了"（只统计，不改；必须独立于开关）
        if not str(row.get("class_pattern") or "").strip():
            stat["class_pattern_empty"] += 1
            if extract_function_name_from_summary(str(m.get("summary") or "")):
                stat["class_pattern_recoverable"] += 1

        # (4) 爆炸半径：哪些层的层文本会变
        n_changed_layers = 0
        for L in LAYERS:
            before = svc._build_enhanced_issue_pattern_text(dict(props_old, sqlite_id=row.get("id"), status="active"), L)
            after = svc._build_enhanced_issue_pattern_text(dict(props_new, sqlite_id=row.get("id"), status="active"), L)
            if before != after:
                n_changed_layers += 1
                stat["layer_changed_" + L] += 1
        layer_blast[n_changed_layers] += 1
        new_rows.append(props_new)

    # ---- 自检：重排后的 solution 用**生产实现**切出来的针 == 我们打算保留的针 ---- #
    bad = 0
    bad_examples = []
    for row, new in zip(rows, new_rows):
        ok, n_intended, n_got = payload_selfcheck(
            new["solution"], tokenize=tokenize, min_tokens=min_tokens, generic=generic, agent=agent)
        if not ok:
            bad += 1
            if len(bad_examples) < 3:
                bad_examples.append((row.get("title"), n_intended, n_got))
    print("\n自检（issue_patterns）：重排后用生产实现重抽的针 与 预期保留集合 不一致的条目数 = %d %s"
          % (bad, "OK" if bad == 0 else "*** 需要检查 ***"))
    for t, a, b in bad_examples:
        print("    %s：预期 %d 条针，实际抽到 %d 条" % (t, a, b))

    # ---- curated_issues：同一份 payload 的第二份副本（默认不动，--include-curated 才动）---- #
    curated_rows, curated_new = [], []
    curated_bad = 0
    if args.include_curated:
        con0 = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
        ccols = [r[1] for r in con0.execute("pragma table_info(curated_issues)")]
        curated_rows = [dict(zip(ccols, r)) for r in con0.execute("select * from curated_issues")]
        con0.close()
        for crow in curated_rows:
            new_sol, _kept, dropped = rewrite_solution_payload(
                str(crow.get("solution") or ""),
                tokenize=tokenize, min_tokens=min_tokens, generic=generic,
                hay_of=hay, fit_cves=fit_cves, min_df=args.needle_min_df,
            )
            if new_sol != str(crow.get("solution") or ""):
                stat["curated_solution_changed"] += 1
            stat["curated_needles_dropped"] += len(dropped)
            ok, n_intended, n_got = payload_selfcheck(
                new_sol, tokenize=tokenize, min_tokens=min_tokens, generic=generic, agent=agent)
            if not ok:
                curated_bad += 1
            curated_new.append(dict(crow, solution=new_sol))
        print("\n自检（curated_issues）：%d 条里自检不一致 %d 条 %s"
              % (len(curated_rows), curated_bad, "OK" if curated_bad == 0 else "*** 需要检查 ***"))

    # ---- 写副本 ------------------------------------------------------------ #
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.db, args.out)
    con = sqlite3.connect(str(args.out))
    cur = con.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row, new in zip(rows, new_rows):
        sets, vals = [], []
        for field in ("solution", "llm_semantic", "error_type", "problematic_pattern"):
            if field in cols and str(new.get(field) or "") != str(row.get(field) or ""):
                sets.append("%s=?" % field)
                vals.append(new.get(field))
        if not sets:
            continue
        sets.append("updated_at=?")
        vals.append(now)
        vals.append(row.get("id"))
        cur.execute("update issue_patterns set %s where id=?" % ",".join(sets), vals)
    if args.include_curated and curated_rows:
        ccols_out = [r[1] for r in cur.execute("pragma table_info(curated_issues)")]
        for crow, new in zip(curated_rows, curated_new):
            if str(new.get("solution") or "") == str(crow.get("solution") or ""):
                continue
            sets, vals = ["solution=?"], [new.get("solution")]
            if "updated_at" in ccols_out:
                sets.append("updated_at=?")
                vals.append(now)
            vals.append(crow.get("id"))
            cur.execute("update curated_issues set %s where id=?" % ",".join(sets), vals)
    con.commit()
    con.close()

    # ---- 报告 -------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("干跑报告（只写了副本：%s）" % args.out)
    print("=" * 100)
    print("  有『Remove incorrect logic』payload 的条目 : %d" % stat["payload_entries"])
    print("  被剪掉的针（在 >=%d 个文件里出现）          : %d" % (args.needle_min_df, stat["needles_dropped"]))
    print("  solution 发生变化的条目                    : %d" % stat["solution_changed"])
    print("  写入 llm_semantic 的条目                   : %d" % stat["llm_semantic_set"])
    if args.apply_error_type:
        print("  error_type 变化的条目                      : %d" % stat["error_type_changed"])
    else:
        print("  error_type 未重算（加 --apply-error-type 才做）")
        print("  class_pattern 为空                         : %d（重算能从摘要里救回的：%d）"
              % (stat["class_pattern_empty"], stat["class_pattern_recoverable"]))

    if args.include_curated:
        print("  curated_issues：solution 变化的行                 : %d / %d（被剪掉的针 %d）"
              % (stat["curated_solution_changed"], len(curated_rows), stat["curated_needles_dropped"]))
    else:
        print("  curated_issues 未处理（加 --include-curated 才做；不加则 curated 通道仍读旧粘针）")

    print("\n  爆炸半径：每个条目有几层的层文本会变（决定要重嵌多少条向量）")
    for k in sorted(layer_blast):
        print("    %d 层变了 : %d 条" % (k, layer_blast[k]))
    print("    分层明细：" + "  ".join("%s=%d" % (L, stat["layer_changed_" + L]) for L in LAYERS))
    print("    要重嵌的向量条数 ≈ %d 条（每层各 %d 条目）"
          % (sum(stat["layer_changed_" + L] for L in LAYERS), len(rows)))

    if examples:
        print("\n  变化样例（剪掉了哪些针、df 多大）：")
        for e in examples[: args.show]:
            print("    %-16s %s" % (e["cve"], e["dropped"]))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "source_db": str(args.db), "candidate_db": str(args.out),
        "stats": dict(stat), "layer_blast": {str(k): v for k, v in layer_blast.items()},
        "needle_min_df": args.needle_min_df, "apply_error_type": args.apply_error_type,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n报告已写出: %s" % args.report)
    print("\n下一步（本轮没做）：① 你确认后把副本写回线上库；② 重嵌受影响的层；③ 同步 Weaviate；④ 开 GPU 跑冒烟。")


if __name__ == "__main__":
    main()
