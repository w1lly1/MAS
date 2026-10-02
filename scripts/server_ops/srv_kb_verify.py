#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对线上"知识库 + 向量索引"的实际状态（在服务器上跑，只读）。

为什么要有它：真机切库/写向量是不可逆操作，**每一步之后都要能证明"现在线上到底是什么"**。
本脚本一次打印：

  1. Weaviate：对象总数、按层分布、`llm_semantic` 非空数、几个抽样对象的 layer_text 指纹；
  2. SQLite：`issue_patterns` / `curated_issues` 行数、`llm_semantic` 非空数、
     `solution` 指纹（用来判断"针重排"到底有没有落库）、
     以及若干指定 entry 的 solution 摘要（默认看 127 / curated 163 这两个关键条目）；
  3. 一致性：库里 llm_semantic 与索引里 llm_semantic 是否一致（按 sqlite_id 抽 20 个比对）。

用法：
    venv/bin/python -u scripts/server_ops/srv_kb_verify.py
    venv/bin/python -u scripts/server_ops/srv_kb_verify.py --show 127 --show-curated 163
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "infrastructure/database/mas.db"


def fp(text: str, n: int = 12) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, action="append", default=[],
                    help="打印这些 issue_patterns.id 的 solution 摘要")
    ap.add_argument("--show-curated", type=int, action="append", default=[],
                    help="打印这些 curated_issues.id 的 solution 摘要")
    args = ap.parse_args()

    print("=" * 100)
    print("SQLite：%s" % DB)
    print("=" * 100)
    con = sqlite3.connect(DB)
    ip_cols = [r[1] for r in con.execute("pragma table_info(issue_patterns)")]
    n_ip = con.execute("select count(*) from issue_patterns").fetchone()[0]
    n_ci = con.execute("select count(*) from curated_issues").fetchone()[0]
    print("  issue_patterns 行数=%d   curated_issues 行数=%d" % (n_ip, n_ci))
    if "llm_semantic" in ip_cols:
        n_sem = con.execute("select count(*) from issue_patterns "
                            "where trim(coalesce(llm_semantic,''))<>''").fetchone()[0]
        print("  llm_semantic 列存在：非空 %d / %d" % (n_sem, n_ip))
    else:
        n_sem = 0
        print("  **没有 llm_semantic 列**（= 重建前的结构）")
    # 针（payload）指纹：把所有 solution 串起来取哈希，用来判断"重排"是否落库
    all_sol = "\n".join(str(r[0] or "") for r in con.execute("select solution from issue_patterns order by id"))
    print("  issue_patterns.solution 总指纹 = %s（长度 %d）" % (fp(all_sol), len(all_sol)))
    n_needle_sep = con.execute(
        "select count(*) from issue_patterns where solution like '%;;%'").fetchone()[0]
    print("  含 ';;'（重排后的切分标记）的条目 = %d" % n_needle_sep)
    all_csol = "\n".join(str(r[0] or "") for r in con.execute("select solution from curated_issues order by id"))
    print("  curated_issues.solution 总指纹 = %s（长度 %d）"
          % (fp(all_csol), len(all_csol)))
    n_csep = con.execute(
        "select count(*) from curated_issues where solution like '%;;%'").fetchone()[0]
    print("  curated 里含 ';;' 的条目 = %d" % n_csep)

    db_sem = {}
    if n_sem:
        db_sem = {int(i): (s or "").strip() for i, s in
                  con.execute("select id, llm_semantic from issue_patterns")}

    for rid in args.show:
        row = con.execute("select solution, file_pattern, title from issue_patterns where id=?",
                          (rid,)).fetchone()
        if row:
            print("\n  [issue_patterns id=%d] title=%s file=%s\n    solution=%s"
                  % (rid, row[2], row[1], (row[0] or "")[:220]))
    for cid in args.show_curated:
        row = con.execute("select solution, pattern_id from curated_issues where id=?",
                          (cid,)).fetchone()
        if row:
            print("\n  [curated_issues id=%d] pattern_id=%s\n    solution=%s"
                  % (cid, row[1], (row[0] or "")[:220]))
    con.close()

    print()
    print("=" * 100)
    print("Weaviate：KnowledgeItem")
    print("=" * 100)
    try:
        import weaviate
    except Exception as exc:  # noqa: BLE001
        print("  连不上（缺 weaviate 包）：%s" % exc)
        return 1
    c = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
    try:
        col = c.collections.get("KnowledgeItem")
        props = [p.name for p in col.config.get().properties]
        objs = col.query.fetch_objects(limit=1000, include_vector=False).objects
        per_layer = {}
        sem_objs = {}
        text_fp = {}
        for o in objs:
            p = o.properties
            layer = str(p.get("vector_layer") or "")
            per_layer[layer] = per_layer.get(layer, 0) + 1
            sid = int(p.get("sqlite_id"))
            if str(p.get("llm_semantic") or "").strip():
                sem_objs[sid] = True
            text_fp[(sid, layer)] = fp(str(p.get("layer_text") or ""))
        print("  对象总数=%d  按层=%s" % (len(objs), per_layer))
        print("  属性列含 llm_semantic: %s" % ("llm_semantic" in props))
        print("  llm_semantic 非空的对象数=%d（期望：重建前 0，重建后约 386=193×2：semantic+full）"
              % len(sem_objs))
        if db_sem:
            ids = sorted(db_sem)[:20]
            same = sum(1 for i in ids if (i in sem_objs) == bool(db_sem[i]))
            print("  一致性抽查（前 20 个条目，库 vs 索引的 llm_semantic 非空与否）: %d/20 一致" % same)
        print("  抽样 layer_text 指纹: %s" % list(text_fp.items())[:4])
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
