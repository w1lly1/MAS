#!/bin/bash
# 根因诊断：**哪些通道的候选**走到了融合分支、它们的语义项为什么是 0。
#
# 假设（待验证，不许猜）：curated_issue / sqlite 通道的候选 `semantic_score` 恒为 0
# （它们不是向量检索来的），于是 z = (0 - μ)/σ 极负 ⇒ 语义项 0 ⇒ fusion_score 永远等于 s(x)。
set -u
cd /root/autodl-tmp/MAS || exit 1
RUNS=${1:-reports/kbself_on_runs.txt}

venv/bin/python - "$RUNS" <<'PY'
import json, sys, glob, os
from collections import Counter, defaultdict

runs = [ln.strip() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip()]
by_chan = defaultdict(lambda: {"n": 0, "sem": [], "fields": Counter(), "term_nonzero": 0,
                               "score_nonzero": 0, "stats_ok": 0})
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
                    ch = str(c.get("channel") or "?")
                    d = by_chan[ch]
                    d["n"] += 1
                    d["sem"].append(float(c.get("semantic_score") or 0.0))
                    for mf in (c.get("matched_fields") or []):
                        d["fields"][mf] += 1
                    if float(c.get("fusion_semantic_term") or 0.0) > 0:
                        d["term_nonzero"] += 1
                    if float(c.get("fusion_score") or 0.0) > 0:
                        d["score_nonzero"] += 1
                    if int(c.get("fusion_stats_n") or 0) > 0:
                        d["stats_ok"] += 1

print("%-16s %6s %10s %10s %10s %10s   %s" %
      ("通道", "条数", "语义项>0", "分>0", "拿到分布", "语义分均值", "命中字段（前 4）"))
for ch, d in sorted(by_chan.items(), key=lambda kv: -kv[1]["n"]):
    sem = d["sem"]
    print("%-16s %6d %10d %10d %10d %10.4f   %s" %
          (ch, d["n"], d["term_nonzero"], d["score_nonzero"], d["stats_ok"],
           sum(sem) / max(1, len(sem)), dict(d["fields"].most_common(4))))
print()
print("说明：'拿到分布' = 该候选所属层的整层相似度分布我取到了（fusion_stats_n>0）；")
print("      若某通道 '拿到分布' 与 '语义项>0' 都接近 0，说明它本就拿不到语义分，")
print("      融合对这条通道**结构性无效**（不是阈值问题，也不是否决问题）。")
PY
