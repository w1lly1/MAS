# -*- coding: utf-8 -*-
"""BM25 无门控独立检测基线（seed=2024）。

口径：与本文相同的代码分片作为查询，对知识库 200 条条目的完整文本视图做 BM25 检索，
不加任何门控（检索到什么就算什么）：
  库内组：top-k 是否包含样本自身的知识库条目 → 检索级召回率
  库外组：top-k 必然返回 k 条其他 CVE 条目 → 若直接采信，错配样本率为 100%
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
DB = ROOT / "infrastructure/database/mas.db"
KB = ROOT / "reports/weaviate_kb_seed2024.jsonl"
TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|[+\-*/%=<>!&|^~]")
K = 3


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        df = collections.Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [collections.Counter(d) for d in docs]

    def topk(self, query, k=K):
        scores = []
        for i in range(self.N):
            dl, tf, sc = self.doc_len[i], self.tf[i], 0.0
            for t in query:
                f = tf.get(t)
                if not f:
                    continue
                sc += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((sc, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]


def tok(s):
    return [t.lower() for t in TOKEN_RE.findall(s or "")]


con = sqlite3.connect(str(DB))
id_by_title = {t: i for i, t in con.execute("SELECT id, title FROM issue_patterns")}
title_by_id = {i: t for t, i in id_by_title.items()}
con.close()

docs, ids = [], []
for line in KB.open(encoding="utf-8"):
    o = json.loads(line)
    if o.get("vector_layer") != "full":
        continue
    docs.append(tok(o.get("layer_text") or ""))
    ids.append(id_by_title.get(o["title"]))
bm = BM25(docs)
print(f"知识库 {len(docs)} 条（完整文本视图）")

hit = collections.Counter()
n_kb = n_held = 0
held_cand = 0
for sid in range(4):
    with gzip.open(ROOT / f"reports/wsens_dump.jsonl_shard{sid}.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            role, cve = rec.get("role"), rec.get("cve")
            own = id_by_title.get(cve)
            texts = []
            for fe in rec.get("files") or []:
                for e in fe.get("gap") or []:
                    cc = e.get("code_chunk") or {}
                    if cc.get("text"):
                        texts.append(cc["text"])
            if not texts:
                continue
            q = tok(" ".join(texts)[:4000])
            top = [ids[i] for i in bm.topk(q, K)]
            if role == "kb":
                n_kb += 1
                for k in (1, 2, 3):
                    if own in top[:k]:
                        hit[k] += 1
            else:
                n_held += 1
                held_cand += len(top)

print(f"库内组 {n_kb} 个样本：BM25 检索级召回率 recall@1={hit[1]/n_kb*100:.1f}%、"
      f"recall@3={hit[3]/n_kb*100:.1f}%")
print(f"库外组 {n_held} 个样本：每个样本均返回 {K} 条候选，平均 {held_cand/max(n_held,1):.1f} 条/样本"
      f" → 不加门控直接采信时错配样本率为 100.0%")
