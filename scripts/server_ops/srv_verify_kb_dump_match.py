#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""证明"离线 KB dump"与"线上 Weaviate 索引"逐条一致（在服务器上跑，只读）。

为什么需要：A5b 的 LLM 判定提示词是从**离线 dump** 里取历史记录文本的。
如果 dump 与线上索引不一致，那批判定就是在"另一套知识库"上做的，结论不可用。
本脚本把 dump 的 800 行与线上 800 个对象按 (sqlite_id, vector_layer) 对齐，
逐条比对 layer_text 的 sha256，任何一条不同都会列出来。

用法：
    venv/bin/python -u scripts/server_ops/srv_verify_kb_dump_match.py \
        [--dump reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path,
                    default=REPO / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--class", dest="cls", default="KnowledgeItem")
    args = ap.parse_args()

    if not args.dump.is_file():
        print("缺 dump 文件：%s" % args.dump)
        return 1

    offline = {}
    for line in args.dump.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        offline[(int(r["sqlite_id"]), r.get("vector_layer") or "")] = sha(r.get("layer_text") or "")
    print("离线 dump：%d 条 (sqlite_id, layer) 组合，文件 %s" % (len(offline), args.dump.name))

    import weaviate
    client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)
    try:
        coll = client.collections.get(args.cls)
        agg = coll.aggregate.over_all(total_count=True)
        print("线上对象总数：%s" % agg.total_count)
        online, n = {}, 0
        for obj in coll.iterator(include_vector=False):
            p = obj.properties
            key = (int(p.get("sqlite_id") or 0), p.get("vector_layer") or "")
            online[key] = sha(p.get("layer_text") or "")
            n += 1
        print("线上取回：%d 个对象" % n)
    finally:
        client.close()

    only_off = sorted(set(offline) - set(online))
    only_on = sorted(set(online) - set(offline))
    diff = sorted(k for k in set(offline) & set(online) if offline[k] != online[k])
    print("\n只在 dump 里：%d %s" % (len(only_off), only_off[:5]))
    print("只在线上：  %d %s" % (len(only_on), only_on[:5]))
    print("两边都有但 layer_text 不同：%d %s" % (len(diff), diff[:5]))

    print()
    if not only_off and not only_on and not diff:
        print("✅ 完全一致（%d 条逐条 sha256 相同）" % len(offline))
        return 0
    print("❌ 不一致 —— 不能用这份 dump 的文本做判定")
    return 2


if __name__ == "__main__":
    sys.exit(main())
