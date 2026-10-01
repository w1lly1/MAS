#!/bin/bash
# KB 切换（**可逆**）：把"线上库 + 索引"整体切到 重建前 或 重建后。
#
# 为什么连 `solution` 属性一起切：门控的 clone 尺子读的是候选的 `solution`
# —— 若只切向量不切属性，baseline 臂会**偷偷用上针修复**，对比就不干净了。
#
# 用法: bash _srv_kb_switch.sh old|new
set -u
MODE="${1:?用法: _srv_kb_switch.sh old|new}"
cd /root/autodl-tmp/MAS || exit 1

PREREBUILD_DB=$(ls -t /root/autodl-tmp/mas.db.bak_prerebuild_* 2>/dev/null | head -1)
case "$MODE" in
  old)
    DUMP=/root/autodl-tmp/weaviate_kb_dump_prerebuild.jsonl
    DB="$PREREBUILD_DB"
    SEM_MODE=clear
    ;;
  new)
    DUMP=/root/autodl-tmp/weaviate_kb_dump_after.jsonl
    DB=/root/autodl-tmp/mas_rebuild_candidate.db
    SEM_MODE=fill
    ;;
  *) echo "未知模式: $MODE"; exit 1 ;;
esac
[ -f "$DUMP" ] || { echo "*** 缺少快照 $DUMP"; exit 1; }
[ -f "$DB" ] || { echo "*** 缺少数据库 $DB"; exit 1; }

echo "切换到 $MODE：DB=$DB  DUMP=$DUMP  语义属性=$SEM_MODE"
cp -f "$DB" infrastructure/database/mas.db

./venv/bin/python - "$DUMP" "$SEM_MODE" <<'PY'
import json, sqlite3, sys
import weaviate

dump_path, sem_mode = sys.argv[1], sys.argv[2]
con = sqlite3.connect("infrastructure/database/mas.db")
cols = [r[1] for r in con.execute("pragma table_info(issue_patterns)")]
has_sem = "llm_semantic" in cols
sem = {}
if has_sem:
    sem = {int(i): (s or "").strip() for i, s in
           con.execute("select id, llm_semantic from issue_patterns")}
con.close()
if not has_sem:
    print("注意：这份库**没有** llm_semantic 列（= 重建前的结构）—— 语义属性将全部置空")

rows = []
with open(dump_path, encoding="utf-8") as fh:
    for line in fh:
        rows.append(json.loads(line))
print("快照对象数:", len(rows), " DB 里 llm_semantic 非空:", sum(1 for v in sem.values() if v))

c = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
try:
    col = c.collections.get("KnowledgeItem")
    have_sem = "llm_semantic" in [p.name for p in col.config.get().properties]
    idx = {}
    for o in col.query.fetch_objects(limit=1000, include_vector=True).objects:
        p = o.properties
        idx[(int(p.get("sqlite_id")), str(p.get("vector_layer")))] = o.uuid
    ok = miss = 0
    for r in rows:
        key = (int(r["sqlite_id"]), str(r.get("vector_layer") or ""))
        u = idx.get(key)
        if not u:
            miss += 1
            continue
        vec = r.get("_vector") or []
        props = {"layer_text": r.get("layer_text") or "", "solution": r.get("solution") or ""}
        if have_sem:
            props["llm_semantic"] = sem.get(key[0], "") if sem_mode == "fill" else ""
        try:
            col.data.update(uuid=u, properties=props, vector=[float(x) for x in vec])
            ok += 1
        except Exception:
            col.data.update(uuid=u, properties=props, vector={"default": [float(x) for x in vec]})
            ok += 1
    print("已更新对象: %d（缺失 %d）" % (ok, miss))

    # 验证
    n_sem = n_text = 0
    check = {(int(r["sqlite_id"]), str(r.get("vector_layer") or "")): (r.get("layer_text") or "", r.get("solution") or "")
             for r in rows}
    for o in col.query.fetch_objects(limit=1000, include_vector=False).objects:
        p = o.properties
        key = (int(p.get("sqlite_id")), str(p.get("vector_layer")))
        if str(p.get("llm_semantic") or "").strip():
            n_sem += 1
        if key in check and (str(p.get("layer_text") or "") == check[key][0]):
            n_text += 1
    print("验证：layer_text 与快照一致的对象 %d / %d" % (n_text, len(rows)))
    print("验证：llm_semantic 非空的对象 %d（old 模式应为 0，new 模式应约 386）" % n_sem)
finally:
    c.close()
PY

echo "DB 侧：$(./venv/bin/python -c "
import sqlite3
c = sqlite3.connect('infrastructure/database/mas.db')
cols = [r[1] for r in c.execute('pragma table_info(issue_patterns)')]
if 'llm_semantic' in cols:
    print('llm_semantic 非空 =', c.execute(\"select count(*) from issue_patterns where trim(coalesce(llm_semantic,''))<>''\").fetchone()[0])
else:
    print('无 llm_semantic 列（重建前的结构）')
")"
