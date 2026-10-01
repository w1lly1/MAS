# -*- coding: utf-8 -*-
"""重算整库的 `error_type` 并与库里存的值对比 —— 量化"规则修好了多少"。

## 为什么

`error_type` 不只是个标签：索引侧 `semantic` 层写着 `[error_type] <值>`，
`code_pattern` 层那句"模式描述"（`problematic_pattern`）也是**按它选的**。
所以分类一错，那一层的文本也跟着变弱。

实测踩到的 bug：原规则在原文里找 `"out of bounds"`（空格），而摘要里是
`"out-of-bounds"`（连字符）→ 匹配失败 → 一条明确的越界读被归到 `general`。

## 这个脚本给出三件事

1. **全库重算 vs 库里存值**：多少条会变、变成什么（体现"规则修正"的影响面）；
2. **变更明细**：逐条给出 摘要 → 旧分类 → 新分类，便于人工抽查是否改对了；
3. **与"大模型自由判断"的一致性**（给了 `--families` 的话）：
   模型没被告知答案时选的家族，是这里唯一可用的外部参照。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.bigvul_ingest.rules import derive_error_type, normalize_for_match  # noqa: E402

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--dataset-root", type=Path, default=DS)
    ap.add_argument("--families", type=Path, default=None,
                    help="模型自由判断的家族（{CVE: family}），用于算一致率")
    ap.add_argument("--show", type=int, default=15, help="打印多少条变更明细")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/error_type_recompute.json")
    args = ap.parse_args()

    con = sqlite3.connect(str(args.db))
    rows = [(int(i), (t or "").strip().upper(), (et or ""), (ed or ""))
            for i, t, et, ed in con.execute(
                "select id, title, error_type, error_description from issue_patterns")]
    con.close()

    def meta(cve: str) -> dict:
        p = args.dataset_root / "metadata" / cve / "cve_metadata.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    changed, same, no_meta = [], 0, 0
    new_dist, old_dist = Counter(), Counter()
    for _id, cve, old, desc in rows:
        m = meta(cve)
        if not m:
            no_meta += 1
            new = old
        else:
            new = derive_error_type(str(m.get("cwe_id") or ""),
                                    str(m.get("vulnerability_classification") or ""),
                                    str(m.get("summary") or desc))
        old_dist[old] += 1
        new_dist[new] += 1
        if new == old:
            same += 1
        else:
            changed.append({"cve": cve, "id": _id, "old": old, "new": new,
                            "cwe": str(m.get("cwe_id") or ""),
                            "summary": desc[:160],
                            "matched": [w for w in ("out of bounds", "buffer overflow",
                                                    "oversight", "denial of service",
                                                    "validation", "bypass", "race",
                                                    "off by one", "leak")
                                        if w in normalize_for_match(desc)]})

    print("=" * 104)
    print("整库 error_type 重算（用修好的规则；输入取数据集 metadata 的 cwe/分类/摘要）")
    print("=" * 104)
    print("  条目 %d 条；**分类会改变 %d 条 = %.1f%%**；不变 %d；缺 metadata 无法重算 %d"
          % (len(rows), len(changed), 100 * len(changed) / max(1, len(rows)), same, no_meta))

    print("\n  旧分布 vs 新分布:")
    keys = sorted(set(old_dist) | set(new_dist))
    print("    %-22s %6s %6s %s" % ("家族", "旧", "新", "变化"))
    for k in sorted(keys, key=lambda x: -max(old_dist[x], new_dist[x])):
        d = new_dist[k] - old_dist[k]
        print("    %-22s %6d %6d %+d" % (k, old_dist[k], new_dist[k], d))

    print("\n  变更明细（前 %d 条）:" % args.show)
    print("    %-16s %-20s → %-20s %-10s %s" % ("CVE", "旧", "新", "CWE", "命中词"))
    for c in changed[: args.show]:
        print("    %-16s %-20s → %-20s %-10s %s" % (
            c["cve"], c["old"], c["new"], c["cwe"] or "-", ",".join(c["matched"]) or "-"))

    if args.families and args.families.exists():
        raw_fam = json.loads(args.families.read_text(encoding="utf-8"))
        # 查询侧的家族文件键是 `CVE@分片序号`（同一个文件的多个分片各有一条）。
        # 这里按 CVE 归并、取**众数**当该样本的模型判断，这样两种键形式都能吃。
        per_cve = {}
        for k, v in raw_fam.items():
            if not v:
                continue
            per_cve.setdefault(k.split("@")[0].split("#")[0].upper(), []).append(v)
        fam = {c: Counter(vs).most_common(1)[0][0] for c, vs in per_cve.items()}

        by_cve = {}
        for _i, cve, old, desc in rows:
            m = meta(cve)
            new = derive_error_type(str(m.get("cwe_id") or ""),
                                    str(m.get("vulnerability_classification") or ""),
                                    str(m.get("summary") or desc)) if m else old
            by_cve[cve] = (old, new)
        cmp_rows = [(c, fam[c], by_cve[c][0], by_cve[c][1]) for c in fam if c in by_cve]
        if cmp_rows:
            n = len(cmp_rows)
            agree_old = sum(1 for _c, f, o, _n in cmp_rows if f == o)
            agree_new = sum(1 for _c, f, _o, nw in cmp_rows if f == nw)
            print("\n  与『大模型自由判断』的一致性（%d 个样本；模型**没被告知答案**）:" % n)
            print("    旧规则 %d/%d = %.1f%%   新规则 %d/%d = %.1f%%"
                  % (agree_old, n, 100 * agree_old / n, agree_new, n, 100 * agree_new / n))
            for c, f, o, nw in sorted(cmp_rows, key=lambda r: (r[1] != r[3], r[0])):
                mark = "← 新规则改对了" if (f == nw and f != o) else (
                    "← 新规则改错了" if (f == o and f != nw) else "")
                print("      %-16s 模型=%-20s 旧=%-18s 新=%-18s %s" % (c, f, o, nw, mark))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"changed": len(changed), "total": len(rows),
         "old_dist": dict(old_dist), "new_dist": dict(new_dist),
         "detail": changed}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写出: %s" % args.out)
    print("\n  怎么读：'变成 memory_overflow/input_validation' 的多，说明原规则因为"
          "连字符/词形问题漏判了不少；\n"
          "          与模型一致率若明显上升，说明改的是真错，不只是换了个标签。")


if __name__ == "__main__":
    main()
