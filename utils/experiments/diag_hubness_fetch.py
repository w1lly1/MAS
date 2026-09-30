#!/usr/bin/env python
"""阶段0 诊断 · 第1步：从 Weaviate 导出全部 (layer_text, vector, 元数据) 供离线分析。

只读，不改动向量库。产物：/root/autodl-tmp/kb_vectors.json
"""
import json
import os
import urllib.request
from collections import Counter

BASE = "http://127.0.0.1:8080"
OUT = "/root/autodl-tmp/kb_vectors.json"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.load(r)


print("=== 1) schema 属性 ===")
schema = get("/v1/schema")
cls = [c for c in schema.get("classes", []) if c.get("class") == "KnowledgeItem"]
if not cls:
    raise SystemExit("KnowledgeItem 类不存在")
props = [p["name"] for p in cls[0].get("properties", [])]
print("  KnowledgeItem 属性:", props)

print("\n=== 2) 分页拉取对象（含 vector）===")
objs = []
offset = 0
while True:
    batch = get(
        f"/v1/objects?class=KnowledgeItem&limit=100&offset={offset}&include=vector"
    )
    items = batch.get("objects", [])
    if not items:
        break
    objs.extend(items)
    offset += len(items)
    if len(items) < 100 or offset > 5000:
        break
print("  拉取对象数:", len(objs))

novec = sum(1 for o in objs if not o.get("vector"))
print("  其中无 vector 的对象:", novec)

print("\n=== 3) 样例对象结构 ===")
if objs:
    o = objs[0]
    print("  顶层键:", sorted(o.keys()))
    print("  properties 键:", sorted((o.get("properties") or {}).keys()))
    v = o.get("vector") or []
    print("  vector 维度:", len(v))
    nz = sum(1 for x in v if abs(x) > 1e-12)
    print("  非零分量数:", nz, "（说明白化后零填充的 k）")
    print("  properties 样例:",
          json.dumps({k: (str(v2)[:60]) for k, v2 in (o.get("properties") or {}).items()},
                     ensure_ascii=False)[:700])

print("\n=== 4) 分布 ===")
lay = Counter((o.get("properties") or {}).get("vector_layer") for o in objs)
print("  vector_layer 分布:", dict(lay))
sid = {(o.get("properties") or {}).get("sqlite_id") for o in objs}
print("  不同 sqlite_id 数:", len(sid))
has_lt = sum(1 for o in objs if (o.get("properties") or {}).get("layer_text"))
print("  带 layer_text 的对象数:", has_lt)

mode = 0o644
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(objs, f, ensure_ascii=False)
print("\n  已写出:", OUT, os.path.getsize(OUT) // 1024, "KB")
