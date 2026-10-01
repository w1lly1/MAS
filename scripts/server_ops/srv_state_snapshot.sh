#!/bin/bash
# 跑臂之前的状态快照（避免在 ssh 里内联 python -c 被本地 shell 拆坏）
set -u
cd /root/autodl-tmp/MAS || exit 1
./venv/bin/python - <<'PY'
import json, sqlite3
d = json.load(open("utils/experiments/smoke_kb30.json", encoding="utf-8"))
items = d["items"]
roles = {}
for it in items:
    roles[it.get("role")] = roles.get(it.get("role"), 0) + 1
print("30 样本批次：items=%d  roles=%s" % (len(items), roles))
print("前 3 个样本：", [it["cve"] for it in items[:3]])

cfg = json.load(open("infrastructure/config/ai_agent_config.json", encoding="utf-8"))
sp = cfg.get("second_pass_analysis_agent", {})
print("gap_chunk_semantic_lookup =", sp.get("gap_chunk_semantic_lookup"))
print("gap_query_source =", sp.get("gap_query_source"))
print("error_code_clone_min_tokens =", sp.get("error_code_clone_min_tokens"))

con = sqlite3.connect("infrastructure/database/mas.db")
n_sem = con.execute("select count(*) from issue_patterns where trim(coalesce(llm_semantic,''))<>''").fetchone()[0]
n_all = con.execute("select count(*) from issue_patterns").fetchone()[0]
print("DB：%d 条，llm_semantic 非空 %d" % (n_all, n_sem))
con.close()
PY
echo "--- git ---"
git log --oneline -1
echo "dirty=$(git status --short | wc -l)"
echo "--- 磁盘 ---"
df -h /root/autodl-tmp | tail -1
