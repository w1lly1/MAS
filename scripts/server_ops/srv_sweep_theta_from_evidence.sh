#!/usr/bin/env bash
# 事后扫 θ：用**同一次跑批**的产物算"换个阈值会多放行几条、是不是同文件"。
#
# 为什么能这么做：每条走到 DNF 的候选都把 `fusion_score` 与 `unified_structured_score`
# 落盘了 ⇒ θ 只是一个比较动作，可以在离线重算，**不必为每个 θ 各跑一臂**。
# 这既省机器时间，也避免"为看结果而反复调参"。
#
# 用法: bash srv_sweep_theta_from_evidence.sh <runs 列表文件>
set -u
cd /root/autodl-tmp/MAS || exit 1
RUNS="${1:?用法: srv_sweep_theta_from_evidence.sh <runs 列表文件>}"

venv/bin/python - "$RUNS" <<'PY'
import json, sys, glob, os
from collections import Counter

THETA_S = 0.65                     # 与生产 gate_structured_threshold 一致
THETAS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]
runs = [ln.strip() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip()]

sub = []          # 未达 θ_s（即"老规则不放行"）但有融合分的候选
n_all = 0
for rel in runs:
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
                    n_all += 1
                    us = float(c.get("unified_structured_score") or 0.0)
                    if us >= THETA_S:
                        continue
                    mf = set(c.get("matched_fields") or [])
                    same = bool(mf & {"file_basename_anchor", "basename_match"})
                    sub.append({"cve": rel.split("/")[0], "sid": c.get("sqlite_id"),
                                "us": us, "fs": float(c.get("fusion_score") or 0.0),
                                "term": float(c.get("fusion_semantic_term") or 0.0),
                                "same": same, "ch": c.get("channel"),
                                "dec": c.get("gating_decision")})

print("走到 DNF 的候选: %d 条；其中**未达 θ_s（老规则不放行）**的: %d 条" % (n_all, len(sub)))
print()
print("%-8s %10s %12s %12s %12s" % ("θ", "会新放行", "其中同文件", "其中跨文件", "跨文件样本数"))
for t in THETAS:
    hit = [c for c in sub if c["fs"] >= t]
    same = sum(1 for c in hit if c["same"])
    cross = len(hit) - same
    print("%-8.2f %10d %12d %12d %12d" % (t, len(hit), same, cross,
                                          len({c["cve"] for c in hit if not c["same"]})))
print()
terms = sorted(c["term"] for c in sub if c["term"] > 0)
if terms:
    print("这些候选的语义项分布（仅 >0 的 %d 条）: 最小 %.4f / 中位 %.4f / 最大 %.4f"
          % (len(terms), terms[0], terms[len(terms) // 2], terms[-1]))
print("语义项 >0 的候选数: %d / %d" % (sum(1 for c in sub if c["term"] > 0), len(sub)))
print()
print("说明：'会新放行' = 在当前 θ 下 `fusion_score >= θ` 且 `s(x) < 0.65`（即老规则本来不放行）。")
print("      '同文件' 用 matched_fields 里的 file_basename_anchor/basename_match 判定，")
print("      跨文件的条数就是**新增误报面**的代价。")
PY
