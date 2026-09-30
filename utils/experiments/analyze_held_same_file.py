# -*- coding: utf-8 -*-
"""回答一个问题：混在 held 组里的"同文件但不同 CVE"样本，到底是什么情况？

对每一个这样的样本，找出它在知识库里的"同文件兄弟"（同一个文件、另一个 CVE），
逐项对比：

  1. 同一个文件？      —— 构造上必然成立
  2. 同一个问题类型？  —— 三重依据：CWE 编号 / 漏洞分类文本 / 本项目自己的 error_type 分类
  3. 同一个修复方案？  —— 直接比修复文本（KB 的 solution vs 该样本自己按同一条流水线生成的 solution）

输出：逐条对照表 + 汇总（多少对是"同文件+同类型+不同修复"）。
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import load_kb_index, normalize_key  # noqa: E402

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
META, BEFORE, AFTER = DS / "metadata", DS / "before", DS / "after"
DB = ROOT / "infrastructure/database/mas.db"
AUDIT = ROOT / "reports/role_audit_400_error.json"


def meta_of(cve: str) -> dict:
    p = META / cve / "cve_metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tokens(text: str):
    return [t for t in re.findall(r"[A-Za-z_]\w*|\d+", str(text or "").lower()) if len(t) > 1]


def jaccard(a, b) -> float:
    A, B = set(tokens(a)), set(tokens(b))
    return (len(A & B) / len(A | B)) if (A or B) else 0.0


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    items = audit["items"]
    conf = [r for r in items if r["role_auto"] == "held" and r.get("file_in_kb")]
    print("审计批次: %s" % audit["batch"])
    print("判定为 held 但库中存在同路径文件的样本: %d 条\n" % len(conf))

    kb = load_kb_index(DB)
    c = sqlite3.connect(str(DB))
    rows = c.execute("select id, title, error_type, severity, framework, language, "
                     "file_pattern, class_pattern, problematic_pattern, solution, "
                     "error_description from issue_patterns").fetchall()
    c.close()
    recs = []
    for r in rows:
        recs.append({"id": r[0], "cve": (r[1] or "").strip().upper(), "error_type": r[2],
                     "severity": r[3], "framework": r[4], "language": r[5],
                     "file_pattern": r[6], "class_pattern": r[7],
                     "problematic_pattern": r[8], "solution": r[9], "desc": r[10]})

    # 该样本自己按同一条流水线生成的记录（用于比较"修复方案是否相同"）
    print("正在为该批样本生成自身的结构化记录（同一条 ingest 流水线）...")
    try:
        from utils.bigvul_ingest.build_structured_ingest import BuildConfig
        from utils.bigvul_ingest.build_kb40_tasks import _build_payload_for_cves
        cfg = BuildConfig(metadata_root=META, before_root=BEFORE, after_root=AFTER,
                          output_dir=ROOT / "reports", output_name="_tmp_self.json",
                          start=0, count=len(conf), max_snippet_chars=2000,
                          session_id="audit-self", ingest_mode="strict")
        payload = _build_payload_for_cves([r["cve"] for r in conf], cfg)
        selfmap = {}
        # payload["data"] 是 [{"instances":[...], "pattern":{...}}, ...]
        for item in (payload.get("data") or []):
            if not isinstance(item, dict):
                continue
            rec = item.get("pattern") if isinstance(item.get("pattern"), dict) else item
            cve = str(rec.get("title") or rec.get("cve") or rec.get("cve_id") or "").strip().upper()
            if cve:
                selfmap[cve] = rec
        if selfmap:
            sample = next(iter(selfmap.values()))
            print("  自身记录 %d 条；字段样例: %s" % (len(selfmap), sorted(sample.keys())[:14]))
        else:
            print("  [警告] 未能定位自身记录，类型/修复对比将只用元数据")
    except Exception as e:  # noqa: BLE001
        print("  [警告] 生成自身记录失败: %s" % e)
        selfmap = {}

    print("\n%-16s %-26s %-16s %-9s %-9s %-9s %s" % (
        "样本CVE", "文件（末两级）", "库内兄弟CVE", "CWE同?", "分类同?", "类型同?", "修复文本相似度"))
    stat = {"pairs": 0, "same_cwe": 0, "same_class": 0, "same_type": 0, "same_solution": 0}
    detail = []
    for r in conf:
        cve = str(r["cve"])
        keys = {normalize_key(f, 2) for f in (r.get("hit_relpaths") or [])}
        sibs = [x for x in recs if normalize_key(x["file_pattern"], 2) in keys]
        if not sibs:
            continue
        me = meta_of(cve)
        mine = selfmap.get(cve, {})
        my_cwe = str(me.get("cwe_id") or "").strip().upper()
        my_cls = str(me.get("vulnerability_classification") or "").strip().lower()
        my_type = str(mine.get("error_type") or "").strip().lower()
        my_sol = str(mine.get("solution") or "")
        for s in sibs:
            sm = meta_of(s["cve"])
            s_cwe = str(sm.get("cwe_id") or "").strip().upper()
            s_cls = str(sm.get("vulnerability_classification") or "").strip().lower()
            s_type = str(s["error_type"] or "").strip().lower()
            same_cwe = bool(my_cwe and s_cwe and my_cwe == s_cwe)
            same_cls = bool(my_cls and s_cls and my_cls == s_cls)
            same_type = bool(my_type and s_type and my_type == s_type)
            sim = jaccard(my_sol, s["solution"])
            stat["pairs"] += 1
            stat["same_cwe"] += int(same_cwe)
            stat["same_class"] += int(same_cls)
            stat["same_type"] += int(same_type)
            stat["same_solution"] += int(sim > 0.95)
            print("%-16s %-26s %-16s %-9s %-9s %-9s %.3f" % (
                cve, sorted(keys)[0][:26], s["cve"], "是" if same_cwe else "否",
                "是" if same_cls else "否", "是" if same_type else "否", sim))
            detail.append({
                "held_cve": cve, "file_key": sorted(keys)[0],
                "kb_sibling_cve": s["cve"],
                "held": {"cwe": my_cwe, "classification": my_cls, "error_type": my_type,
                         "solution": my_sol[:400]},
                "sibling": {"cwe": s_cwe, "classification": s_cls, "error_type": s_type,
                            "solution": str(s["solution"])[:400],
                            "problematic_pattern": str(s["problematic_pattern"])[:200]},
                "same_cwe": same_cwe, "same_classification": same_cls,
                "same_error_type": same_type, "solution_similarity": round(sim, 4),
            })

    print("\n=== 汇总（%d 对样本↔库内兄弟）===" % stat["pairs"])
    for k, label in (("same_cwe", "同一 CWE 编号"), ("same_class", "同一漏洞分类文本"),
                     ("same_type", "同一 error_type 分类"), ("same_solution", "修复文本几乎相同")):
        n = stat[k]
        print("  %-22s %3d / %-3d  = %.1f%%" % (label, n, stat["pairs"], 100 * n / max(1, stat["pairs"])))
    both = sum(1 for d in detail
               if (d["same_cwe"] or d["same_classification"] or d["same_error_type"])
               and d["solution_similarity"] <= 0.95)
    print("  %-22s %3d / %-3d  = %.1f%%   ← 用户问的『同文件+同类型+不同修复』" % (
        "同类型但修复不同", both, stat["pairs"], 100 * both / max(1, stat["pairs"])))

    out = ROOT / "reports/held_same_file_cross_cve_analysis.json"
    out.write_text(json.dumps({"summary": stat, "same_type_diff_fix": both,
                               "pairs": detail}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写出:", out)


if __name__ == "__main__":
    main()
