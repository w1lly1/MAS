#!/bin/bash
# Step 0d：把缺失的 `llm_semantic` 属性补上（走**生产代码**路径），并回填 193 条值。
#
# 说明：`WeaviateVectorService.connect()` 在"类已存在"时会调用 `_ensure_additive_properties()`
# 给旧类补新属性。同步代码之前，服务器跑的是旧代码（没有这段逻辑），同步之后还没有任何
# 流程真正连过 Weaviate —— 所以这个属性一直没被补上（不是代码坏了，是没被跑到）。
set -u
cd /root/autodl-tmp/MAS || exit 1

echo "===== 1) 走生产路径连接（触发补属性） ====="
./venv/bin/python - <<'PY'
import sys
sys.path.insert(0, ".")
from infrastructure.database.weaviate.service import WeaviateVectorService
svc = WeaviateVectorService()
ok = svc.connect(auto_create_schema=True)
print("connect ->", ok)
col = svc._get_collection()
props = [p.name for p in col.config.get().properties]
print("补完之后 schema properties:", props)
print("llm_semantic 在不在:", "llm_semantic" in props)
try:
    svc.client.close()
except Exception:
    pass
PY

echo
echo "===== 2) 回填 193 条 llm_semantic 属性 ====="
./venv/bin/python - <<'PY'
import sqlite3
import weaviate

con = sqlite3.connect("infrastructure/database/mas.db")
sem = {int(i): (s or "").strip() for i, s in
       con.execute("select id, llm_semantic from issue_patterns")}
con.close()
targets = {k for k, v in sem.items() if v}
print("库里 llm_semantic 非空的条目:", len(targets))

c = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
try:
    col = c.collections.get("KnowledgeItem")
    props = [p.name for p in col.config.get().properties]
    if "llm_semantic" not in props:
        print("*** 属性仍不存在，回填中止")
        raise SystemExit(1)
    idx = {}
    for o in col.query.fetch_objects(limit=1000, include_vector=False).objects:
        p = o.properties
        idx.setdefault(int(p.get("sqlite_id")), {})[str(p.get("vector_layer"))] = o.uuid
    up = 0
    for sid in sorted(targets):
        for L in ("semantic", "full"):
            u = idx.get(sid, {}).get(L)
            if not u:
                continue
            col.data.update(uuid=u, properties={"llm_semantic": sem[sid]})
            up += 1
    print("回填对象数:", up, "（期望 = 193 条 × 2 层 = 386）")
    # 验证
    one = next(iter(sorted(targets)))
    o = col.query.fetch_object_by_id(idx[one]["semantic"])
    print("抽查 id=%d semantic 对象的 llm_semantic 前 80 字: %r"
          % (one, str(o.properties.get("llm_semantic") or "")[:80]))
finally:
    c.close()
PY

echo
echo "===== 3) 最终确认 ====="
./venv/bin/python - <<'PY'
import weaviate
c = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
try:
    col = c.collections.get("KnowledgeItem")
    print("schema properties:", [p.name for p in col.config.get().properties])
    n = 0
    for o in col.query.fetch_objects(limit=1000, include_vector=False).objects:
        if str(o.properties.get("llm_semantic") or "").strip():
            n += 1
    print("带 llm_semantic 值的对象数:", n)
    print("对象总数:", col.aggregate.over_all(total_count=True).total_count)
finally:
    c.close()
PY
