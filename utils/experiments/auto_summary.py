# -*- coding: utf-8 -*-
"""批量分析结束后的自动汇总：生成 CSV 到 reports/。

由 api/main.py 的 batch 流程在「批量分析结束」后自动调用：
  - 总是生成逐项汇总 reports/batch_summary.csv
    （含 security agent 是否真正用上 LLM：hybrid_fusion / 检测漏洞数 / 威胁文本长度）
  - 若是 400 实验批（文件名含 400，或条目≥100 且含 held），再自动跑 4 个评测
    并生成 reports/experiment_summary_400.csv

这样冒烟测试（smoke1.json）跑完也能立刻看到 batch_summary.csv，验证自动汇总机制。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"


def _is_400_batch(config_path: str, items: list) -> bool:
    name = Path(config_path).name.lower()
    if "400" in name:
        return True
    if len(items) >= 100 and any(str(i.get("role", "")).lower() == "held" for i in items):
        return True
    return False


def _read_security_summary(cve: str, run_id: str) -> dict:
    """读取某 run 的 security agent 报告，抽取关键字段（判断是否空壳）。"""
    d = REPORTS / "analysis" / cve / run_id / "agents" / "security"
    if not d.exists():
        return {}
    strategies: set[str] = set()
    vulns = 0
    raw_len = 0
    n = 0
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        n += 1
        a = j.get("security_result", {}).get("ai_security_analysis", {})
        fusion = (a.get("overall_security_rating") or {}).get("fusion") or {}
        strategies.add(str(fusion.get("strategy", "")))
        vulns += len(a.get("vulnerabilities_detected") or [])
        raw_len = max(raw_len, int((a.get("threat_model") or {}).get("raw_text_length") or 0))
    return {"n_reports": n, "strategy": ",".join(sorted(strategies)),
            "vulns": vulns, "raw_len": raw_len}


def generate_batch_summary(config_path: str, items: list, results: list) -> Path:
    """逐项汇总：每个批条目一行，突出 security agent 是否真正跑起 LLM。"""
    out = REPORTS / "batch_summary.csv"
    header = ["cve", "role", "status", "run_id", "security_strategy",
              "llm_active", "vulns_detected", "raw_text_length"]
    rows = []
    for item, res in zip(items, results):
        cve = item.get("cve") or item.get("output_dir") or ""
        role = item.get("role", "")
        status = res.get("status", "")
        run_id = res.get("run_id", "")
        s = _read_security_summary(cve, run_id) if run_id else {}
        strat = s.get("strategy", "")
        llm_active = "1" if "hybrid_fusion" in strat else ("0" if strat else "")
        rows.append([cve, role, status, run_id, strat, llm_active,
                     s.get("vulns", ""), s.get("raw_len", "")])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return out


def run_auto_summary(config_path: str, items: list, results: list) -> list:
    """批结束后自动汇总。返回生成的 CSV 路径列表。"""
    produced = [generate_batch_summary(config_path, items, results)]
    print(f"\n📄 逐项汇总已写: {produced[0]}")

    if _is_400_batch(config_path, items):
        from utils.experiments import export_summary
        print("\n🧪 检测到 400 实验批，自动运行评测并汇总...")
        export_summary.run_evals()
        produced.append(export_summary.generate_summary())

    return produced
