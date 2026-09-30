#!/usr/bin/env python
"""S0-7：把 SQLite 的 file_pattern/class_pattern 回填到现有 800 个 Weaviate 对象属性。

背景与预期：
  · 分析发现 `_backfill_weaviate_candidate_solution()` 会在门控前用 sqlite_id 从 SQLite
    回填这两个字段 → 门控**已**看到真实值（落盘候选核验 3,659/3,659 = 100%）。
  · 因此本回填**预期不改变门控行为**，其价值在于：① 存储属性与 SQLite 一致（数据卫生）；
    ② 任何直接读属性的检索/导出路径不再拿到空值。
  · 本脚本会实测确认「向量与 layer_text 保持不变」，这是"纯属性回填"的硬要求。

做法：Weaviate REST PATCH 逐对象更新 properties（vectorizer=none，不会触发重向量化）。
只读 SQLite；不改 layer_text；删改前先记录抽样向量指纹用于比对。
"""
import hashlib
import json
import os
import sqlite3
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

BASE = "http://127.0.0.1:8080"
KB = "/root/autodl-tmp/kb_vectors.json"
FINGER_BEFORE = "/root/autodl-tmp/kb_finger_before.json"
FINGER_AFTER = "/root/autodl-tmp/kb_finger_after.json"
CLASS = "KnowledgeItem"


def req(method, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=60) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def fingerprint(objs):
    """对每个对象取 (id, vector_sha1前16, layer_text_sha1前16) 指纹。"""
    fp = {}
    for o in objs:
        p = o.get("properties") or {}
        vec = (o.get("vectors") or {}).get("default") or o.get("vector") or []
        vh = hashlib.sha1(json.dumps([round(float(x), 6) for x in vec]).encode()).hexdigest()[:16]
        th = hashlib.sha1(str(p.get("layer_text") or "").encode()).hexdigest()[:16]
        fp[o["id"]] = {"vec": vh, "layer_text": th,
                       "sqlite_id": p.get("sqlite_id"), "layer": p.get("vector_layer")}
    return fp


def fetch_all():
    objs, offset = [], 0
    while True:
        b = req("GET", f"/v1/objects?class={CLASS}&limit=100&offset={offset}&include=vector")
        items = b.get("objects", [])
        if not items:
            break
        objs.extend(items)
        offset += len(items)
        if len(items) < 100 or offset > 5000:
            break
    return objs


def main():
    objs = json.load(open(KB, encoding="utf-8"))
    print("载入 %s：%d 个对象" % (KB, len(objs)))
    if not os.path.exists(FINGER_BEFORE):
        live = fetch_all()
        json.dump(fingerprint(live), open(FINGER_BEFORE, "w"))
        print("已记录回填前指纹: %d 个对象 -> %s" % (len(live), FINGER_BEFORE))

    sq = sqlite3.connect(os.path.join(ROOT, "infrastructure", "database", "mas.db"))
    meta = {str(r[0]): {"file_pattern": r[1] or "", "class_pattern": r[2] or ""}
            for r in sq.execute("select id, file_pattern, class_pattern from issue_patterns").fetchall()}
    print("SQLite: %d 行；其中 file_pattern 非空 %d，class_pattern 非空 %d" % (
        len(meta), sum(1 for v in meta.values() if v["file_pattern"]),
        sum(1 for v in meta.values() if v["class_pattern"])))

    ok = skip = fail = 0
    errs = []
    for i, o in enumerate(objs, 1):
        p = o.get("properties") or {}
        sid = str(p.get("sqlite_id"))
        m = meta.get(sid)
        if not m:
            skip += 1
            continue
        try:
            req("PATCH", f"/v1/objects/{CLASS}/{o['id']}", {
                "class": CLASS,
                "properties": {"file_pattern": m["file_pattern"],
                               "class_pattern": m["class_pattern"]},
            })
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            if len(errs) < 3:
                errs.append((o["id"], sid, str(e)[:120]))
        if i % 200 == 0:
            print("  已处理 %d / %d" % (i, len(objs)))
    print("PATCH 结果: 成功 %d  跳过 %d  失败 %d" % (ok, skip, fail))
    for e in errs:
        print("   失败样例:", e)

    print("\n=== 回填后核验 ===")
    live = fetch_all()
    json.dump(fingerprint(live), open(FINGER_AFTER, "w"))
    before = json.load(open(FINGER_BEFORE))
    after = json.load(open(FINGER_AFTER))
    vec_same = sum(1 for k in after if k in before and after[k]["vec"] == before[k]["vec"])
    lt_same = sum(1 for k in after if k in before and after[k]["layer_text"] == before[k]["layer_text"])
    print("  对象数 before=%d after=%d" % (len(before), len(after)))
    print("  向量指纹一致: %d / %d" % (vec_same, len(after)))
    print("  layer_text 指纹一致: %d / %d" % (lt_same, len(after)))
    nfp = sum(1 for o in live if str((o.get("properties") or {}).get("file_pattern") or "").strip())
    ncp = sum(1 for o in live if str((o.get("properties") or {}).get("class_pattern") or "").strip())
    print("  属性 file_pattern 非空: %d / %d" % (nfp, len(live)))
    print("  属性 class_pattern 非空: %d / %d" % (ncp, len(live)))
    bad = [o["id"] for o in live
           if str(meta.get(str((o.get("properties") or {}).get("sqlite_id")), {}).get("file_pattern") or "")
           and (o.get("properties") or {}).get("file_pattern") !=
           meta[str((o.get("properties") or {}).get("sqlite_id"))]["file_pattern"]]
    print("  与 SQLite 不一致的对象数: %d" % len(bad))


if __name__ == "__main__":
    sys.exit(main())
