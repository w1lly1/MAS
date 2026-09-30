# -*- coding: utf-8 -*-
import sqlite3, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
sys.path.insert(0, str(ROOT))
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from infrastructure.embeddings.codebert_embedder import get_embedder

def build_layer_texts(p):
    return {
        "semantic": "\n".join([f"[error_type] {p.get('error_type') or ''}",
                               f"[severity] {p.get('severity') or ''}",
                               f"[language] {p.get('language') or ''}",
                               f"[framework] {p.get('framework') or ''}",
                               f"[description] {p.get('error_description') or ''}"]),
        "code_pattern": "\n".join([f"[problematic_pattern] {p.get('problematic_pattern') or ''}",
                                   f"[file_pattern] {p.get('file_pattern') or ''}",
                                   f"[class_pattern] {p.get('class_pattern') or ''}",
                                   f"[language] {p.get('language') or ''}"]),
        "solution": "\n".join([f"[solution] {p.get('solution') or ''}",
                               f"[error_description] {p.get('error_description') or ''}",
                               f"[severity] {p.get('severity') or ''}"]),
        "full": "\n".join([f"[error_type] {p.get('error_type') or ''}",
                           f"[severity] {p.get('severity') or ''}",
                           f"[language] {p.get('language') or ''}",
                           f"[framework] {p.get('framework') or ''}",
                           f"[description] {p.get('error_description') or ''}",
                           f"[pattern] {p.get('problematic_pattern') or ''}",
                           f"[solution] {p.get('solution') or ''}",
                           f"[file_pattern] {p.get('file_pattern') or ''}",
                           f"[class_pattern] {p.get('class_pattern') or ''}"]),
    }

db = ROOT / "reports" / "mas_seed2025.db"
con = sqlite3.connect(str(db))
con.row_factory = sqlite3.Row
pats = [dict(r) for r in con.execute(
    "SELECT id, title, error_type, severity, language, framework, error_description, "
    "problematic_pattern, solution, file_pattern, class_pattern FROM issue_patterns")]
snips = {pid: (s or "") for pid, s in con.execute(
    "SELECT pattern_id, MAX(code_snippet) FROM curated_issues WHERE code_snippet IS NOT NULL GROUP BY pattern_id")}
con.close()
pats = [p for p in pats if (snips.get(p["id"]) or "").strip()]
print("n=", len(pats))

# find empty layer texts
for L in ["semantic", "code_pattern", "solution", "full"]:
    empties = [p["id"] for p in pats if not build_layer_texts(p)[L].strip()]
    print(f"{L}: {len(empties)} 个空文本, ids={empties[:20]}")

empty_q = [p["id"] for p in pats if not snips[p["id"]].strip()]
print("空查询:", empty_q[:20])

# check NaN vectors
emb = get_embedder()
import torch
for L in ["semantic", "code_pattern", "solution", "full"]:
    nan_cnt = 0
    first_nan = None
    for p in pats:
        t = build_layer_texts(p)[L]
        v = emb._forward_raw(t)
        if v is None or not np.isfinite(v).all():
            nan_cnt += 1
            if first_nan is None:
                first_nan = p["id"]
    print(f"{L}: NaN/None 向量 {nan_cnt}, 首个={first_nan}")
