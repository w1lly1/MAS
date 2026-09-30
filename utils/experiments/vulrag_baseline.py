# -*- coding: utf-8 -*-
"""Vul-RAG 风格知识检索基线（生成式，需 GPU + LLM）。

评审 #2 要求的强基线：复现 Vul-RAG 核心思路（不做全系统复现）——
  「历史漏洞多维知识（功能语义/成因/修复方案）按相似度检索 → 拼入 LLM 做生成式判定」。

与本文「非生成式二次校验」的差异即本基线要对比的对象：
  - 本文：检索 → 置信门控 → 非生成式写入（禁止自由生成）。
  - 本基线：检索 → 拼入 Qwen1.5-7B-Chat 提示 → 生成式漏洞判定（无门控）。

协议（与 #4 一致）：
  - kb 200：答案在库，正确输出自身 CVE 即召回。
  - held 200：答案不在库，输出任意 CVE 即误报（错配）。
  - 检索：BM25 对 200 条 KB 条目（title+error_description+solution+class_pattern）top-k。
  - 生成：Qwen1.5-7B-Chat（fp16, GPU），贪婪解码（do_sample=False）。

运行（远程 GPU 机，MAS 根目录）：
    export HF_HOME=/root/autodl-tmp/hf-cache
    python -u utils/experiments/vulrag_baseline.py --top-k 3 --limit 400
产出：reports/vulrag_baseline.jsonl + 汇总打印
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

BATCH_CANDIDATES = [
    ROOT / "utils" / "experiments" / "test_400_error_batch.json",
    ROOT / "utils" / "experiments" / "test_400_error_batch_seed2025.json",
    ROOT / "论文" / "test_400_error_batch.json",
]
DB_DEFAULT = ROOT / "infrastructure" / "database" / "mas.db"
TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|[+\-*/%=<>!&|^~]")
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def topk(self, query, k=5):
        scores = []
        for i in range(self.N):
            tf = self.tf[i]
            dl = self.doc_len[i]
            sc = 0.0
            for t in query:
                if t not in tf:
                    continue
                f = tf[t]
                sc += self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9)))
            scores.append((sc, i))
        scores.sort(key=lambda x: -x[0])
        return scores[:k]


def _find_batch(arg_batch: str) -> Path:
    if arg_batch:
        p = Path(arg_batch)
    else:
        p = next((c for c in BATCH_CANDIDATES if c.exists()), BATCH_CANDIDATES[0])
    return p


def _remap_target_dir(target_dir: str) -> str:
    """batch 的 Linux 路径在 Windows 上跑时重映射到本地 ROOT。"""
    if not target_dir:
        return target_dir
    if target_dir.startswith("/root/autodl-tmp/MAS/"):
        return str(ROOT / target_dir[len("/root/autodl-tmp/MAS/"):])
    if "E:/" in target_dir or "E:\\" in target_dir:
        idx = target_dir.find("MAS")
        if idx >= 0:
            return str(ROOT) + target_dir[idx + 3:]
    return target_dir


def _load_kb(db_path):
    import sqlite3
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT id, title, error_type, error_description, problematic_pattern, solution, file_pattern, class_pattern FROM issue_patterns")
    rows = cur.fetchall()
    con.close()
    entries = []
    for r in rows:
        entries.append({
            "id": r[0],
            "title": (r[1] or "").strip(),
            "error_type": r[2] or "",
            "error_description": r[3] or "",
            "problematic_pattern": r[4] or "",
            "solution": r[5] or "",
            "file_pattern": r[6] or "",
            "class_pattern": r[7] or "",
            "text": " ".join([r[1] or "", r[2] or "", r[3] or "", r[4] or "", r[5] or "", r[6] or "", r[7] or ""]),
        })
    return entries


def _build_prompt(entries, code, top_k):
    lines = ["You are a vulnerability detector. Given a list of known historical vulnerabilities and a code snippet, determine which of these known vulnerabilities are present in the code.",
             "", "Historical vulnerabilities:"]
    for i, e in enumerate(entries[:top_k], 1):
        lines.append(f"[{i}] {e['title'] or 'unknown'}")
        lines.append(f"    Type: {e['error_type']}")
        lines.append(f"    Root cause: {e['error_description']}")
        lines.append(f"    Fix: {e['solution'][:600]}")
    lines.append("")
    lines.append("Code to analyze:")
    lines.append(code[:12000])
    lines.append("")
    lines.append("Answer with the CVE ID(s) present in the code, separated by commas. If none apply, output NONE.")
    return "\n".join(lines)


def _local_model_path() -> str:
    """解析本地缓存的 Qwen 快照路径，避免 AutoTokenizer 对 model id 触发 model_info 联网。"""
    import os
    hf = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache/huggingface"))
    base = hf / "hub" / "models--Qwen--Qwen1.5-7B-Chat" / "snapshots"
    if base.exists():
        snaps = [d for d in base.iterdir() if d.is_dir()]
        if snaps:
            return str(snaps[0])
    return "Qwen/Qwen1.5-7B-Chat"


def _load_qwen(device: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    name = _local_model_path()
    if device == "gpu":
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    else:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, device_map=None, local_files_only=True)
    model.eval()
    return tok, model


def _generate(tok, model, prompt, max_new_tokens=220):
    import torch
    # Qwen 对话模板（单轮 system 风格合并进 user）
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[0][enc.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=str, default="")
    ap.add_argument("--db", type=str, default=str(DB_DEFAULT))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个样本（0=全部），用于冒烟")
    ap.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--out", type=str, default=str(ROOT / "reports" / "vulrag_baseline.jsonl"))
    args = ap.parse_args()

    batch_path = _find_batch(args.batch)
    if not batch_path.exists():
        print(f"[fatal] batch 不存在: {batch_path}")
        sys.exit(1)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    items = batch["items"]
    if args.limit > 0:
        items = items[: args.limit]

    entries = _load_kb(args.db)
    docs = [tokenize(e["text"]) for e in entries]
    bm25 = BM25(docs)
    print(f"KB 条目 {len(entries)}，BM25 索引就绪", flush=True)

    tok, model = _load_qwen(args.device)
    print(f"LLM 就绪（{args.device}）", flush=True)

    out_path = Path(args.out)
    out_f = open(out_path, "w", encoding="utf-8")
    kb_recall = held_fp = kb_n = held_n = 0
    for idx, it in enumerate(items, 1):
        cve = it.get("cve")
        role = it.get("role")
        files = []
        td = _remap_target_dir(it.get("target_dir"))
        if td and Path(td).exists():
            files = [p for p in Path(td).rglob("*") if p.is_file() and p.suffix.lower() in {".c", ".h", ".cpp", ".cc", ".cxx", ".py", ".java"}]
        if not files:
            print(f"[{idx}/{len(items)}] {cve} ({role}): 无源文件，跳过", flush=True)
            continue
        code = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files[:3])[:40000]

        q = tokenize(code)
        ranked = bm25.topk(q, k=args.top_k)
        top_entries = [entries[i] for _, i in ranked if entries[i]["title"]]

        prompt = _build_prompt(top_entries, code, args.top_k)
        try:
            raw = _generate(tok, model, prompt)
        except Exception as e:
            print(f"[{idx}/{len(items)}] {cve} ({role}): 生成异常 {e}", flush=True)
            raw = ""
        preds = sorted(set(CVE_RE.findall(raw or "")))
        hit = False
        if role == "kb":
            kb_n += 1
            hit = cve in preds
            if hit:
                kb_recall += 1
        else:
            held_n += 1
            hit = bool(preds)  # 答案不在库，任何 CVE 输出都是误报
            if hit:
                held_fp += 1
        out_f.write(json.dumps({"cve": cve, "role": role, "pred": preds, "hit": hit, "raw": raw[:300]}, ensure_ascii=False) + "\n")
        out_f.flush()
        tag = "hit" if hit else "miss"
        print(f"[{idx}/{len(items)}] {cve} ({role}) {tag} pred={preds}  (kb {kb_recall}/{kb_n}, held {held_fp}/{held_n})", flush=True)

    out_f.close()
    print("\n==== Vul-RAG 风格基线汇总 ====")
    print(f"kb 召回: {kb_recall}/{kb_n} = {kb_recall/kb_n:.1%}" if kb_n else "kb: 0")
    print(f"held 误报: {held_fp}/{held_n} = {held_fp/held_n:.1%}" if held_n else "held: 0")


if __name__ == "__main__":
    main()
