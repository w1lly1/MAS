# -*- coding: utf-8 -*-
"""汇总导出：把 4 个评测脚本的输出整理成一张 CSV。

前置（二选一）：
    1) 已分别跑完 evaluate_400 / delta_recall / bm25_endtoend / hard_negative；
    2) 或直接 `--run-evals` 让本脚本先依次跑完这 4 个再汇总。

运行（MAS 根目录）：
    python utils/experiments/export_summary.py             # 仅汇总（评测已跑完）
    python utils/experiments/export_summary.py --run-evals # 先跑评测再汇总

产出：reports/experiment_summary_400.csv
  - 第一段「汇总指标」：召回 / 误报 / ΔRecall / BM25 / 硬负样本 等关键数字
  - 第二段「每样本明细」：每个 CVE 的逐条结果（合并 delta_recall 的首轮/二次命中）
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"
EVAL_DIR = ROOT / "utils/experiments"

EVAL_SCRIPTS = [
    "evaluate_400.py",
    "delta_recall.py",
    "bm25_endtoend.py",
    "hard_negative.py",
]


def run_evals() -> None:
    """依次运行 4 个评测脚本（每个脚本自带输出文件落盘）。"""
    py = sys.executable
    for s in EVAL_SCRIPTS:
        print(f"\n{'=' * 60}\n运行 {s}\n{'=' * 60}")
        subprocess.run([py, str(EVAL_DIR / s)], cwd=str(ROOT), check=False)


def load_json(name: str):
    p = REPORTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_eval_rows():
    p = REPORTS / "eval_400.csv"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("true", "1", "yes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-evals", action="store_true", help="先依次运行 4 个评测脚本")
    ap.add_argument("--out", default=str(REPORTS / "experiment_summary_400.csv"))
    args = ap.parse_args()

    if args.run_evals:
        run_evals()

    eval_rows = load_eval_rows()
    dr = load_json("delta_recall.json")
    bm = load_json("bm25_endtoend.json")
    hn = load_json("hard_negative.json")

    if not eval_rows:
        print("⚠️ 未找到 reports/eval_400.csv，请先运行 evaluate_400.py（或加 --run-evals）。")
        sys.exit(2)

    # ---- 汇总指标 ----
    kb_rows = [r for r in eval_rows if r["role"] == "kb"]
    held_rows = [r for r in eval_rows if r["role"] == "held"]
    kb_ok = [r for r in kb_rows if r["status"] == "ok"]
    held_ok = [r for r in held_rows if r["status"] == "ok"]
    kb_missing = [r for r in kb_rows if r["status"] == "missing"]
    held_missing = [r for r in held_rows if r["status"] == "missing"]

    captured = sum(1 for r in kb_ok if _truthy(r.get("captured")))
    fp = sum(1 for r in held_ok if _truthy(r.get("fp")))
    avg_fp = (sum(int(r["n_findings"]) for r in held_ok) / len(held_ok)) if held_ok else 0.0

    summary: list[list] = []

    def add(metric, value, rate="", note=""):
        summary.append([metric, value, rate, note])

    add("召回(kb正确捕捉)", captured,
        f"{captured / len(kb_ok):.1%}" if kb_ok else "-",
        f"答案在库 {len(kb_ok)} 个有效样本中二次校验正确召回自身条目")
    add("误报(held)", fp,
        f"{fp / len(held_ok):.1%}" if held_ok else "-",
        f"答案不在库 {len(held_ok)} 个有效样本中产生误报")
    add("平均误报条数", f"{avg_fp:.2f}", "",
        "held 有效样本平均 new_findings 条数")

    if dr:
        n = int(dr.get("n") or 0)
        add("首轮LLM召回", int(dr.get("first_pass_recall") or 0),
            f"{dr.get('first_pass_recall', 0) / n:.1%}" if n else "-",
            "首轮 pureLLM consolidated 已发现该 CVE 历史缺陷")
        add("二次校验召回", int(dr.get("second_pass_recall") or 0),
            f"{dr.get('second_pass_recall', 0) / n:.1%}" if n else "-",
            "二次校验命中自身条目")
        add("ΔRecall(独立净增)", int(dr.get("delta_recall") or 0),
            f"{dr.get('delta_rate', 0):.1%}",
            "二次校验命中且首轮漏检 = 二次校验相对首轮LLM的独立净增")
        add("两者都命中(重叠)", int(dr.get("both") or 0), "", "首轮与二次校验都命中")

    if bm:
        add("BM25召回", int(bm.get("kb_recall") or 0),
            f"{bm.get('kb_rate', 0):.1%}", "BM25 替换语义通道后的 kb 召回")
        add("BM25误报", int(bm.get("held_fp") or 0),
            f"{bm.get('held_fp_rate', 0):.1%}", "BM25 替换语义通道后的 held 误报")

    if hn:
        add("硬负样本残留率", len(hn.get("residue", [])),
            f"{hn.get('residue_rate', 0):.1%}",
            "已修复 after 代码中历史错误代码仍残留的比例")

    if kb_missing:
        add("召回池缺失样本数", len(kb_missing), "", ", ".join(r["cve"] for r in kb_missing[:10]))
    if held_missing:
        add("误报池缺失样本数", len(held_missing), "", ", ".join(r["cve"] for r in held_missing[:10]))

    # ---- 每样本明细 ----
    dr_detail = (dr or {}).get("detail", {})
    detail_header = ["cve", "role", "status", "n_findings", "captured",
                     "fp", "first_pass_hit", "delta_net"]
    detail_rows = []
    for r in eval_rows:
        cve = r["cve"]
        first_hit = ""
        delta_net = ""
        if cve in dr_detail:
            d = dr_detail[cve]
            first_hit = "1" if d.get("first_pass") else "0"
            delta_net = "1" if (d.get("second_pass") and not d.get("first_pass")) else "0"
        detail_rows.append([
            cve, r["role"], r["status"], r["n_findings"],
            r.get("captured", ""), r.get("fp", ""), first_hit, delta_net,
        ])

    # ---- 写 CSV ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["=== 汇总指标 ===", "", "", ""])
        w.writerow(["指标", "数值", "比率", "说明"])
        for row in summary:
            w.writerow(row)
        w.writerow([])
        w.writerow(["=== 每样本明细 ==="] + [""] * (len(detail_header) - 1))
        w.writerow(detail_header)
        for row in detail_rows:
            w.writerow(row)

    # ---- 控制台 ----
    print("\n" + "=" * 62)
    print("汇总指标")
    print("=" * 62)
    for m, v, rate, note in summary:
        print(f"  {m:<18} {v:<6} {rate:<8} {note}")
    print(f"\n明细 {len(detail_rows)} 行已写入: {out}")


if __name__ == "__main__":
    main()
