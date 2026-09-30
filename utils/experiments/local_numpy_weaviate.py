# -*- coding: utf-8 -*-
"""本地 numpy 复刻 WeaviateVectorService.search_knowledge_items。

Weaviate 默认 cosine 距离：distance = 1 - cos_sim（范围 [0,2]，越小越相似）。
agent 内部把 distance 转成 similarity = 1 - distance/2，本复刻逐字段一致。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class NumpyVectorService:
    def __init__(self, dump_path):
        raw: Dict[str, List[Dict[str, Any]]] = {}
        for line in Path(dump_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("_vector"):
                continue
            raw.setdefault(r.get("vector_layer"), []).append(r)

        self._layers: Dict[str, Dict[str, Any]] = {}
        for layer, recs in raw.items():
            vecs = np.array([r["_vector"] for r in recs], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._layers[layer] = {"vecs": vecs / norms, "recs": recs}

    def is_connected(self) -> bool:
        return True

    def connect(self, auto_create_schema: bool = True) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def search_knowledge_items(
        self,
        query_vector: List[float],
        limit: int = 10,
        layer: str = "full",
        additional_filters: Optional[Dict] = None,
    ) -> List[Dict]:
        entry = self._layers.get(layer)
        if not entry:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        sims = entry["vecs"] @ q  # cosine similarity（两侧均 L2 归一化）
        distance = 1.0 - sims
        k = min(int(limit), len(entry["recs"]))
        idx = np.argsort(distance)[:k]
        items = []
        for i in idx:
            rec = entry["recs"][i]
            d = float(distance[i])
            item = {k2: v2 for k2, v2 in rec.items() if k2 != "_vector"}
            item["_additional"] = {"distance": d}
            items.append(item)
        return items

    def get_knowledge_items(self, sqlite_id: Optional[int] = None, limit: int = 20) -> List[Dict]:
        out = []
        for entry in self._layers.values():
            for rec in entry["recs"]:
                if sqlite_id is None or rec.get("sqlite_id") == sqlite_id:
                    out.append({k: v for k, v in rec.items() if k != "_vector"})
        return out[:limit]
