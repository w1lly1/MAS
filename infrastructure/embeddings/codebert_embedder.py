# -*- coding: utf-8 -*-
"""分层文本向量编码器（单例）——按 layer 选模型。

- code_pattern 层：codebert-base（code→code，错误代码精确匹配）
- semantic / solution / full 层：distilbert-base-uncased（text→text，各向异性更轻）

历史：本文件原名 codebert_embedder，最初只接 codebert；后改为 distilbert；现按层分流。
文件名保留以兼容三处 import。

mean pooling + L2 归一化，均 768 维；本地加载（local_files_only=True，CPU）；
加载失败回退到 768 维平凡向量（保持维度一致，near_vector 不报错）。

用法：
    from infrastructure.embeddings.codebert_embedder import embed_text
    vec = embed_text("some text", layer="code_pattern")   # codebert
    vec = embed_text("some text", layer="semantic")        # distilbert
"""
from __future__ import annotations

import json
import os
import threading
from typing import List, Optional

EMBED_DIM = 768
CODEBERT = "microsoft/codebert-base"
DISTILBERT = "distilbert-base-uncased"

# PCA 白化变换文件（whiten_prepare.py 生成）。存在则对向量做白化去各向异性。
WHITENING_PATH = os.path.join(os.path.dirname(__file__), "whitening_transform.json")

_embedder: Optional["TextEmbedder"] = None
_lock = threading.Lock()
_whitening: Optional[dict] = None


def _load_whitening() -> dict:
    """懒加载白化变换 {layer: {"mean": [...], "W": [[...]...]}}。文件缺失则返回空（不白化）。"""
    global _whitening
    if _whitening is not None:
        return _whitening
    _whitening = {}
    if os.path.exists(WHITENING_PATH):
        try:
            with open(WHITENING_PATH, encoding="utf-8") as f:
                _whitening = json.load(f)
        except Exception:
            _whitening = {}
    return _whitening


def _apply_whitening(vec: List[float], layer: Optional[str]) -> List[float]:
    """在 L2 归一化后的向量上做白化：(v - mean) @ W，再 L2 归一，并零填充回 768 维。

    零填充不改余弦相似度（两向量零填充后点积不变），但保证 Weaviate 维度一致。
    无变换则原样返回。
    """
    entry = _load_whitening().get(layer or "")
    if not entry:
        return vec
    try:
        import numpy as np

        mean = np.array(entry.get("mean") or [], dtype=np.float64)
        W = np.array(entry.get("W") or [], dtype=np.float64)
        if mean.size == 0 or W.size == 0:
            return vec
        v = np.array(vec, dtype=np.float64)
        wv = (v - mean) @ W
        n = float(np.linalg.norm(wv))
        if not n:
            return vec
        wv = wv / n
        padded = np.zeros(EMBED_DIM, dtype=np.float64)
        padded[: wv.shape[0]] = wv
        return padded.tolist()
    except Exception:
        return vec


class TextEmbedder:
    def __init__(self) -> None:
        self._cb_tok = None
        self._cb_model = None
        self._cb_attempted = False
        self._db_tok = None
        self._db_model = None
        self._db_attempted = False

    def _ensure_codebert(self) -> None:
        if self._cb_attempted:
            return
        self._cb_attempted = True
        try:
            from transformers import AutoModel, AutoTokenizer

            self._cb_tok = AutoTokenizer.from_pretrained(CODEBERT, local_files_only=True)
            self._cb_model = AutoModel.from_pretrained(CODEBERT, local_files_only=True)
            self._cb_model.eval()
        except Exception as e:  # noqa: BLE001
            self._cb_tok = None
            self._cb_model = None
            print(f"[embedder] codebert 加载失败: {e}")

    def _ensure_distilbert(self) -> None:
        if self._db_attempted:
            return
        self._db_attempted = True
        try:
            from transformers import AutoModel, AutoTokenizer

            self._db_tok = AutoTokenizer.from_pretrained(DISTILBERT, local_files_only=True)
            self._db_model = AutoModel.from_pretrained(DISTILBERT, local_files_only=True)
            self._db_model.eval()
        except Exception as e:  # noqa: BLE001
            self._db_tok = None
            self._db_model = None
            print(f"[embedder] distilbert 加载失败: {e}")

    def embed(self, text: str, layer: Optional[str] = None) -> List[float]:
        # 回滚：code_pattern 层不再用 codebert（代码精确匹配由 is_subseq 承担），
        # 四层统一走 distilbert 文本向量。
        self._ensure_distilbert()
        tok, model = self._db_tok, self._db_model
        if model is None or tok is None:
            return _fallback_embed(text)
        try:
            import torch  # noqa: F401

            text = text or ""
            inp = tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**inp)
            last = out.last_hidden_state  # (1, L, 768)
            # mean pooling：去掉 CLS token，对剩余 token 求均值
            vec = last[:, 1:, :].mean(dim=1).squeeze(0).tolist()
            return _apply_whitening(_l2(vec), layer)
        except Exception:  # noqa: BLE001
            return _fallback_embed(text)


def _l2(v: List[float]) -> List[float]:
    import math

    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else [0.0] * len(v)


def _fallback_embed(text: str) -> List[float]:
    """768 维平凡向量（保持维度一致），前 3 维与旧 _default_embed 同构，其余补 0。"""
    if text is None:
        text = ""
    total = float(sum(ord(c) for c in text))
    length = float(len(text) or 1)
    v = [length, (total % 991) / 991.0, (total % 313) / 313.0]
    return v + [0.0] * (EMBED_DIM - len(v))


def get_embedder() -> "TextEmbedder":
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = TextEmbedder()
    return _embedder


def embed_text(text: str, layer: Optional[str] = None) -> List[float]:
    return get_embedder().embed(text, layer)
