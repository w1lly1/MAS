# -*- coding: utf-8 -*-
"""把 Weaviate KnowledgeItem 全部对象（含向量）导出为 JSONL（用 weaviate-client v4）。"""
import json
import weaviate

client = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
col = client.collections.get("KnowledgeItem")

OUT = "/root/autodl-tmp/weaviate_kb_dump.jsonl"
out = open(OUT, "w", encoding="utf-8")
n = 0
after = None
while True:
    kwargs = dict(limit=1000, include_vector=True)
    if after:
        kwargs["after"] = after
    res = col.query.fetch_objects(**kwargs)
    objs = res.objects
    for obj in objs:
        props = dict(obj.properties)
        vec = obj.vector
        if isinstance(vec, dict):
            vec = vec.get("default") or next(iter(vec.values()), None)
        rec = dict(props)
        rec["_vector"] = list(vec) if vec is not None else None
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    if not objs or len(objs) < 1000:
        break
    after = objs[-1].uuid

out.close()
print(f"dumped {n} objects -> {OUT}")
