#!/usr/bin/env bash
# 把那批"同文件但没被指派层"的候选逐条打出来：分布在几个样本、s(x) 多少、缺什么字段。
# 用途：估算"若给它们补上语义分，最多能多放行几个样本"（决定值不值得再改代码 + 重跑）。
set -u
cd /root/autodl-tmp/MAS || exit 1
RUNS="${1:?用法: srv_list_samefile_nolayer.sh <runs 列表>}"

venv/bin/python - "$RUNS" <<'PY'
import json, sys, glob, os
from collections import Counter, defaultdict

SAME = {"file_basename_anchor", "basename_match"}
runs = [ln.strip() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip()]
per_cve = defaultdict(list)
for rel in runs:
    cve = rel.split("/")[0]
    for f in glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*_r2.json"),
                       recursive=True):
        try:
            j = json.loads(open(f, encoding="utf-8").read())
        except Exception:
            continue
        for key in ("retrieval_evidence", "gap_retrieval_evidence"):
            for b in (j.get(key) or []):
                for c in (b.get("candidates") or []):
                    if not isinstance(c, dict) or "fusion_score" not in c:
                        continue
                    if float(c.get("unified_structured_score") or 0) >= 0.65:
                        continue
                    mf = set(c.get("matched_fields") or [])
                    if not (mf & SAME):
                        continue
                    if c.get("vector_layer"):
                        continue
                    per_cve[cve].append((c.get("sqlite_id"), float(c.get("unified_structured_score") or 0),
                                         sorted(mf), str(c.get("channel"))))

print("『同文件但没层』的候选共 %d 条，分布在 %d 个样本上" %
      (sum(len(v) for v in per_cve.values()), len(per_cve)))
print()
print("%-16s %4s  %-42s %s" % ("样本", "条数", "s(x) 取值", "命中字段（去掉描述类弱证据）"))
for cve, rows in sorted(per_cve.items(), key=lambda kv: -len(kv[1])):
    sx = Counter(round(r[1], 2) for r in rows)
    weak = {"phenomenon_in_description", "root_cause_in_description", "error_type_in_description",
            "error_description_prefix", "problematic_pattern_prefix", "file_basename_in_description",
            "location_in_description", "pattern_in_snippet"}
    strong = Counter(tuple(sorted(set(r[2]) - weak)) for r in rows)
    print("%-16s %4d  %-42s %s" % (cve, len(rows), dict(sx),
                                   [ (",".join(k) or "(全是弱证据)", v) for k, v in strong.items() ][:3]))
print()
print("估算：要让 s(x)=0.2 的候选越过 θ=0.7，需要语义项 ≥ %.3f（z ≥ %.2f）" % ((0.7-0.2)/1.5, (0.7-0.2)/1.5*4))
print("      要让 s(x)=0.45 的候选越过 θ=0.7，需要语义项 ≥ %.3f（z ≥ %.2f）" % ((0.7-0.45)/1.5, (0.7-0.45)/1.5*4))
PY
