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

import functools
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
# 已告警过的 layer 键，避免每个向量都刷屏
_warned_layer_keys: set = set()


def _warn_missing_whitening(layer: Optional[str], available) -> None:
    """白化变换已启用、但当前 layer 取不到变换时告警（每个 layer 只告警一次）。

    这是「静默失配」的兜底：layer=None（或层名写错/大小写不符）会直接跳过白化，
    得到 768 维全非零的原始向量；而索引里存的是白化后向量（仅前 k 维非零，其余恒 0）。
    两者不在同一空间，余弦相似度失去意义，而此前**不产生任何日志**。
    """
    key = layer if isinstance(layer, str) and layer else "<None>"
    if key in _warned_layer_keys:
        return
    _warned_layer_keys.add(key)
    print(
        f"[embedder] ⚠️ 白化变换已启用，但 layer={key!r} 取不到对应变换，"
        f"本次向量【未白化】(768 维全非零)。可用层={sorted(available)}。"
        f"若拿它与已白化的索引向量比较相似度，结果无意义——请显式传入 layer。"
    )


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
    transform = _load_whitening()
    entry = transform.get(layer or "")
    if not entry:
        # 变换存在却取不到 → layer 缺失或写错，会导致查询与索引空间不一致，必须显式告警
        if transform:
            _warn_missing_whitening(layer, transform.keys())
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

    def _forward_raw(self, text: str) -> Optional[List[float]]:
        """distilbert 前向 + mean pooling + L2 归一化（不白化）。

        返回 None 表示模型未就绪（调用方走 _fallback_embed）。
        与白化解耦：semantic/solution/full 三层共享同一 query_text 时，
        只需一次前向，白化按层分别应用（白化是廉价矩阵乘）。
        """
        self._ensure_distilbert()
        tok, model = self._db_tok, self._db_model
        if model is None or tok is None:
            return None
        import torch  # noqa: F401

        text = text or ""
        inp = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**inp)
        last = out.last_hidden_state  # (1, L, 768)
        # mean pooling：去掉 CLS token，对剩余 token 求均值
        vec = last[:, 1:, :].mean(dim=1).squeeze(0).tolist()
        return _l2(vec)

    def embed(self, text: str, layer: Optional[str] = None) -> List[float]:
        # 回滚：code_pattern 层不再用 codebert（代码精确匹配由 is_subseq 承担），
        # 四层统一走 distilbert 文本向量；前向按文本缓存，白化按层应用。
        raw = _forward_raw(text)
        if raw is None:
            return _fallback_embed(text)
        try:
            return _apply_whitening(raw, layer)
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


@functools.lru_cache(maxsize=30000)
def _forward_raw(text: str) -> Optional[List[float]]:
    """按文本缓存的原始前向（不白化）。semantic/solution/full 共享 query_text 时只算一次。"""
    return get_embedder()._forward_raw(text)


@functools.lru_cache(maxsize=30000)
def embed_text(text: str, layer: Optional[str] = None) -> List[float]:
    return get_embedder().embed(text, layer)
