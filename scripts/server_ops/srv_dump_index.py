#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出一份"线上索引现状"留档（在服务器上跑，只读）。

为什么要它：手里已有的 dump 是**切库前**和**切库当刻**那两份。
以后任何"索引是不是被人动过""线上到底长什么样"的判断，都要有**切库之后**的一份基准；
否则又会踩"拿过期的 dump 当线上现状"那个坑（《03》坑 28）。

导出内容与既有 dump 同构：每行一个对象，含全部属性 + `layer_text` + `_vector`。

用法: venv/bin/python -u srv_dump_index.py /root/autodl-tmp/weaviate_kb_dump_postswitch.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import weaviate

out_path = Path(sys.argv[1] if len(sys.argv) > 1
                else "/root/autodl-tmp/weaviate_kb_dump_postswitch.jsonl")

c = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
try:
    col = c.collections.get("KnowledgeItem")
    props = [p.name for p in col.config.get().properties]
    print("属性列: %s" % props)
    objs = col.query.fetch_objects(limit=10000, include_vector=True).objects
    print("对象数: %d" % len(objs))
    per_layer = Counter()
    n_sem = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for o in objs:
            p = dict(o.properties)
            vec = o.vector
            if isinstance(vec, dict):          # 命名向量：取 default
                vec = vec.get("default") or next(iter(vec.values()))
            rec = dict(p)
            rec["_vector"] = [float(x) for x in (vec or [])]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            per_layer[str(p.get("vector_layer"))] += 1
            if str(p.get("llm_semantic") or "").strip():
                n_sem += 1
    print("按层: %s" % dict(per_layer))
    print("llm_semantic 非空的对象: %d" % n_sem)
    print("已写出: %s（%.1f MB）" % (out_path, out_path.stat().st_size / 1e6))
finally:
    c.close()
