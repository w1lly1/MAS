#!/usr/bin/env bash
# 打印某个样本"未达阈值"的候选明细：s(x) / 语义项 / 知识条目 id / 是不是 own / 命中字段。
# 用途：核对"为什么某些候选没被 (λ,θ) 扫描算成新增放行"。
set -u
cd /root/autodl-tmp/MAS || exit 1
RUN="${1:?用法: srv_dump_subthreshold_cands.sh <run 相对路径>}"
DB=infrastructure/database/mas.db

venv/bin/python - "$RUN" "$DB" <<'PY'
import json, sys, glob, os, sqlite3
rel, db = sys.argv[1], sys.argv[2]
con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
id_by_title = {(t or "").strip().upper(): int(i) for i, t in
               con.execute("select id, title from issue_patterns")}
title_by_id = {v: k for k, v in id_by_title.items()}
con.close()
cve = rel.split("/")[0]
own = id_by_title.get(cve.upper())
print("样本 %s  own 条目 id = %s" % (cve, own))
seen = set()
for f in glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*_r2.json"), recursive=True):
    try:
        j = json.loads(open(f, encoding="utf-8").read())
    except Exception:
        continue
    for key in ("retrieval_evidence", "gap_retrieval_evidence"):
        for b in (j.get(key) or []):
            for c in (b.get("candidates") or []):
                if not isinstance(c, dict) or "fusion_score" not in c:
                    continue
                s = float(c.get("unified_structured_score") or 0.0)
                if s >= 0.65:
                    continue
                mf = set(c.get("matched_fields") or [])
                if not (mf & {"file_basename_anchor", "basename_match"}):
                    continue
                key2 = (c.get("channel"), c.get("sqlite_id"), c.get("kb_pattern_id"),
                        round(s, 2), round(float(c.get("fusion_semantic_term") or 0.0), 4))
                if key2 in seen:
                    continue
                seen.add(key2)
                print("  ch=%-14s sqlite_id=%-6s kb_pattern_id=%-6s (=%s)  s=%.2f term=%.4f 分数(λ=4)=%.3f  %s"
                      % (c.get("channel"), c.get("sqlite_id"), c.get("kb_pattern_id"),
                         title_by_id.get(c.get("kb_pattern_id"), "?"), s,
                         float(c.get("fusion_semantic_term") or 0.0),
                         s + 4.0 * float(c.get("fusion_semantic_term") or 0.0),
                         sorted(mf - {"phenomenon_in_description", "root_cause_in_description"})))
PY
