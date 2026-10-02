# -*- coding: utf-8 -*-
"""离线验证 T1 的预测：把 `curated_issues.solution` 也重排后，那条候选会不会被放行？

## 做法（用**生产代码**，不是自己再算一遍）

对 `CVE-2018-20854` 的那条 curated 候选，分别用**旧**和**新**的 `solution` 构造候选字典，
调用**真正的门控函数** `AIDrivenSecondPassAnalysisAgent._gate_candidate()`，看决策与拒因：

    旧 solution（粘针） → 预期 `code_already_fixed`（**这同时是重建正确性的对照**：
                        必须复现出运行日志里观测到的那个拒因，说明候选字典构造是对的）
    新 solution（重排） → 预期放行（`formal_hit`），因为 clone 命中让 s(x) 从 0.2 涨到 0.7 >= 0.65

⚠️ **一个必须记住的教训**：候选进门前还有一步计算**不能漏** —— "错误代码克隆命中"
（权重 0.5）是单独算出来写进 `matched_fields` 的（curated 通道里在 `_match_curated_issue()`
内部完成）。第一版只调了 `_gate_candidate()`，于是新 solution 明明针命中了、分数却没涨，
得到"预测不成立"的**假阴性**。这里单独调 `_apply_error_code_clone_evidence()` 与生产**数值等价**
（同一 haystack、同一匹配函数），并由 `predict_t2_gain.py` 用 **143 条真实证据逐条重放一致**验证过。
详见《03_踩过的坑.md》坑 30（文档目录 `MasOptimize/`）。

候选字典的字段取自**真实运行落盘的证据**（`second_pass/consolidated/*_r2.json`），
只有 `solution` 与"文件身份"由本地构造：

* `solution`：旧取 `reports/mas_live.db`，新取重建后的候选库；
* `file_pattern` / `_analysis_file`：都设成"被分析的那个文件"。
  **依据**：观测到的拒因是 `code_already_fixed`，而门控只在"同文件"前提下才会给出这个理由
  （见 `_gate_candidate` 里 `same_target` 的分支），所以当时 `same_file` 必定成立。

## 用法

    python utils/experiments/verify_curated_fix_prediction.py --cve CVE-2018-20854
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT  # noqa: E402
from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cve", default="CVE-2018-20854")
    ap.add_argument("--runs", type=Path, default=ROOT / "reports/arm1_runs.txt",
                    help="从哪个臂的产物里取「真实候选证据」（默认新系统臂）")
    ap.add_argument("--old-db", type=Path, default=ROOT / "reports/mas_live.db")
    ap.add_argument("--new-db", type=Path, default=ROOT / "reports/mas_rebuild_candidate_v2.db")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent

    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    # ---- 1) 被分析的文件（haystack）与目标条目的 id ----
    d = DS / "before" / args.cve
    files = [f for f in sorted(d.rglob("*")) if f.is_file() and f.suffix.lower() in SOURCE_EXT]
    analyzed = files[0] if files else None
    if analyzed is None:
        raise SystemExit("数据集里找不到 %s 的源码" % args.cve)
    current_code = analyzed.read_text(encoding="utf-8", errors="ignore")
    current_toks = agent._tokenize_code(current_code)

    con = sqlite3.connect("file:%s?mode=ro" % args.old_db.as_posix(), uri=True)
    own = list(con.execute("select id, solution from issue_patterns where upper(title)=?", (args.cve.upper(),)))
    ci = list(con.execute("select id, pattern_id, solution from curated_issues where pattern_id=?",
                          (own[0][0] if own else -1,)))
    con.close()
    if not ci:
        raise SystemExit("curated_issues 里找不到 %s 对应条目" % args.cve)
    curated_id, pattern_id, old_solution = ci[0][0], ci[0][1], ci[0][2]
    print("目标：%s  条目 id=%s  curated id=%s" % (args.cve, pattern_id, curated_id))

    con = sqlite3.connect("file:%s?mode=ro" % args.new_db.as_posix(), uri=True)
    new_solution = con.execute("select solution from curated_issues where id=?", (curated_id,)).fetchone()[0]
    con.close()

    # ---- 2) 从真实运行产物里取候选的原始分数与证据 ----
    run_line = None
    for line in args.runs.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(args.cve + "/"):
            run_line = line.strip()
            break
    if not run_line:
        raise SystemExit("run 列表里没有 %s" % args.cve)
    cve, run = run_line.split("/", 1)
    r2dir = ROOT / "reports/analysis" / cve / run / "second_pass" / "consolidated"
    evidence = None
    for f in sorted(r2dir.glob("*_r2.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        for key in ("retrieval_evidence", "gap_retrieval_evidence"):
            for ev in (j.get(key) or []):
                for c in (ev.get("candidates") or []):
                    if not isinstance(c, dict):
                        continue
                    if str(c.get("channel") or c.get("primary_channel")) == "curated_issue" \
                            and int(c.get("sqlite_id") or -1) == int(curated_id):
                        evidence = c
                        break
                if evidence:
                    break
            if evidence:
                break
        if evidence:
            break
    if not evidence:
        raise SystemExit("在产物里没找到该 curated 候选的证据（检查 run 列表）")
    print("取自真实证据：channel=%s sqlite_id=%s matched_fields=%s"
          % (evidence.get("channel"), evidence.get("sqlite_id"), evidence.get("matched_fields")))
    print("              structured=%.3f semantic=%.3f anchor=%.3f context=%.3f"
          % (evidence.get("structured_score", 0), evidence.get("semantic_score", 0),
             evidence.get("anchor_score", 0), evidence.get("context_score", 0)))

    def build(solution: str) -> dict:
        return {
            "channel": "curated_issue",
            "sqlite_id": curated_id,
            "error_type": "",           # 占位：curated 通道不靠它
            "vector_layer": evidence.get("vector_layer"),
            "structured_score": evidence.get("structured_score", 0.0),
            "semantic_score": evidence.get("semantic_score", 0.0),
            "context_score": evidence.get("context_score", 0.0),
            "anchor_score": evidence.get("anchor_score", 0.0),
            "matched_fields": list(evidence.get("matched_fields") or []),
            "solution": solution,
            "_current_code": current_code,
            "_analysis_file": str(analyzed),
            "file_pattern": str(analyzed),
        }

    print("\n" + "=" * 96)
    print("针层面（生产实现提取的针 vs 被分析文件）")
    print("=" * 96)
    for label, sol in (("旧 curated solution", old_solution), ("新 curated solution", new_solution)):
        frags = agent._extract_error_code_fragments(sol or "")
        hits = [agent._is_contiguous_subseq(t, current_toks) for t in frags]
        print("  %-24s 针数=%d  命中=%s  针长=%s"
              % (label, len(frags), hits, [len(t) for t in frags]))

    print("\n" + "=" * 96)
    print("门控层面（按**生产顺序**：先补 clone 证据，再 _gate_candidate）")
    print("=" * 96)
    results = {}
    for label, sol in (("旧 solution", old_solution), ("新 solution", new_solution)):
        cand = build(sol or "")
        # **关键**：候选进门前还有一步计算不能漏 —— "错误代码克隆命中"（权重 0.5，
        # 会把结构化分数抬 0.5）。真实流水线里，curated 通道是在 `_match_curated_issue()`
        # 内部算它的；我第一版只调了 `_gate_candidate()`，于是"新 solution"虽然针命中了，
        # 分数却没变（unified_s 仍是 0.2）—— 这就是"离线重建漏算生产流程里的中间量"。
        # 这里单独调 `_apply_error_code_clone_evidence()` 与之**数值等价**
        # （同一 haystack、同一 token 匹配函数），已由 `predict_t2_gain.py` 用 143 条真实证据
        # 逐条重放一致（含 matched_fields 与 structured_score）验证过。
        agent._apply_error_code_clone_evidence(
            cand, str(analyzed), {"file": str(analyzed), "code_snippet": current_code[:2000]})
        agent._gate_candidate(cand)
        results[label] = cand
        print("  %-14s → 决策=%-16s 拒因=%-24s matched=%s" %
              (label, cand.get("gating_decision"), cand.get("rejection_reason") or "-",
               cand.get("matched_fields")))
        print("                   structured=%.3f  unified_s=%s  total=%s"
              % (cand.get("structured_score", 0),
                 cand.get("unified_structured_score",
                          agent._unified_structured_score(cand["matched_fields"])),
                 cand.get("total_score")))

    print("\n" + "=" * 96)
    print("判定（先写死，再看结果）")
    print("=" * 96)
    old_ok = results["旧 solution"].get("rejection_reason") == "code_already_fixed"
    new_dec = results["新 solution"].get("gating_decision")
    new_ok = new_dec in ("formal_hit", "explanatory_hit")
    print("  [%s] 对照组：旧 solution 必须复现运行时的拒因 code_already_fixed"
          % ("OK" if old_ok else "NG"))
    print("  [%s] 预测：新 solution 应被放行（formal_hit / explanatory_hit）→ 实际 %s"
          % ("OK" if new_ok else "NG", new_dec))
    print("\n  结论：%s" % ("预测成立：curated 表的 payload 就是那个卡点"
                            if old_ok and new_ok else "预测不成立 —— 需要继续往下量别的卡点"))


if __name__ == "__main__":
    main()
