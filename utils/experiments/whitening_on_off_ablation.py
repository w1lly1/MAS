# -*- coding: utf-8 -*-
"""白化开 vs 白化关 对照表（四视图 recall@1 + 相似度分离度）。

协议（离线 CPU，不联网）：
  - 库侧：reports/_from_gpu/weaviate_kb_dump.jsonl，200 条目 × 4 视图，取每行 layer_text；
  - 查询侧：reports/mas_seed2025.db 表 curated_issues，取每个 pattern_id 的一条非空
    code_snippet（MAX(code_snippet)，与仓库 view_recall1_whitening_ablation.py 同口径）；
  - 只保留"库与查询都存在的同一批条目"，输出实际条目数 n；
  - 白化关 = get_embedder()._forward_raw(text)（mean pooling 去 CLS + L2，未白化）；
  - 白化开 = embed_text(text, layer)（_forward_raw 后 (v-mean)@W 再 L2，零填充回 768，
    白化按层应用，库与查询同层一致）；
  - 穷举余弦（向量已 L2 归一，点积即余弦），recall@1 = 正确条目排第 1 的比例；
  - 分离度 = 正确命中相似度中位数 − 错误匹配相似度中位数。

运行（MAS 根目录，离线 CPU）：
    python utils/experiments/whitening_on_off_ablation.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
from pathlib import Path

# 离线约束：必须在导入 transformers 之前设置（模型已本地缓存）。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from infrastructure.embeddings.codebert_embedder import embed_text, get_embedder  # noqa: E402

LAYERS = ["semantic", "code_pattern", "solution", "full"]
DUMP = ROOT / "reports" / "_from_gpu" / "weaviate_kb_dump.jsonl"
DB = ROOT / "reports" / "mas_seed2025.db"
OUT = Path(__file__).resolve().with_name("whitening_on_off_results.json")


def load_library() -> dict:
    """读 weaviate_kb_dump.jsonl -> {layer: {sqlite_id: layer_text}}。"""
    rows = [json.loads(l) for l in DUMP.read_text(encoding="utf-8").splitlines() if l.strip()]
    lib: dict = {}
    for r in rows:
        lib.setdefault(r["vector_layer"], {})[r["sqlite_id"]] = r["layer_text"]
    return lib


def load_queries() -> dict:
    """读 curated_issues -> {pattern_id: code_snippet}（每 pattern 一条非空，MAX）。"""
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    snips = {
        r["pattern_id"]: r["code_snippet"]
        for r in con.execute(
            "SELECT pattern_id, MAX(code_snippet) AS code_snippet FROM curated_issues "
            "WHERE code_snippet IS NOT NULL AND TRIM(code_snippet) <> '' GROUP BY pattern_id"
        )
    }
    con.close()
    return snips


def l2(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def evaluate(lib_vecs: np.ndarray, q_vecs: np.ndarray) -> tuple:
    """lib_vecs/q_vecs 均为 (n, 768) 已归一。返回 (recall, correct_sims, wrong_sims)。"""
    n = lib_vecs.shape[0]
    S = q_vecs @ lib_vecs.T  # (n, n) 余弦相似度
    order = np.argsort(-S, axis=1)
    recall = int((order[:, 0] == np.arange(n)).sum())
    correct = S.diagonal()
    wrong = S[~np.eye(n, dtype=bool)]
    return recall, correct, wrong


def main() -> None:
    import torch

    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("MKL_NUM_THREADS", "16")
    torch.set_num_threads(16)

    lib = load_library()
    snips = load_queries()

    # 同一批条目：库四视图齐全 且 查询有非空 code_snippet。
    common = sorted(i for i in snips if all(i in lib[L] for L in LAYERS))
    n = len(common)
    print(f"库四视图条目数: {len(lib['semantic'])}（semantic 视图计）")
    print(f"查询有非空 code_snippet 的 pattern 数: {len(snips)}")
    print(f"两口径交集（实际参与计算条目数 n）: {n}")
    print(f"四视图每视图库行数: { {L: len(lib[L]) for L in LAYERS} }\n", flush=True)

    emb = get_embedder()
    raw_cache: dict = {}

    def get_raw(text: str):
        if text not in raw_cache:
            v = emb._forward_raw(text)
            if v is None:
                raise RuntimeError("distilbert 未就绪（本地模型加载失败），无法前向")
            raw_cache[text] = v
        return raw_cache[text]

    # 一致性校验：embed_text(t, layer) 是否 ≡ _forward_raw(t) 后白化（用第一个查询样本、full 层）。
    probe = snips[common[0]]
    from infrastructure.embeddings.codebert_embedder import _apply_whitening

    assert np.allclose(
        np.array(embed_text(probe, "full")),
        np.array(_apply_whitening(get_raw(probe), "full")),
        atol=1e-9,
    ), "embed_text 与 _forward_raw+_apply_whitening 不一致"
    print("已确认：embed_text(t, layer) ≡ _forward_raw(t) 后 _apply_whitening(·, layer)\n", flush=True)

    results: dict = {}
    for L in LAYERS:
        lib_texts = [lib[L][i] for i in common]
        q_texts = [snips[i] for i in common]

        # 白化关：库/查询都用 _forward_raw
        lib_off = np.array([l2(get_raw(t)) for t in lib_texts])
        q_off = np.array([l2(get_raw(t)) for t in q_texts])

        # 白化开：库/查询都用 embed_text(t, L)（同层一致）
        lib_on = np.array([l2(embed_text(t, L)) for t in lib_texts])
        q_on = np.array([l2(embed_text(t, L)) for t in q_texts])

        r_off, cor_off, wrg_off = evaluate(lib_off, q_off)
        r_on, cor_on, wrg_on = evaluate(lib_on, q_on)

        results[L] = {
            "off": {
                "recall": r_off,
                "correct_median": statistics.median(float(x) for x in cor_off),
                "wrong_median": statistics.median(float(x) for x in wrg_off),
                "separation": statistics.median(float(x) for x in cor_off)
                - statistics.median(float(x) for x in wrg_off),
            },
            "on": {
                "recall": r_on,
                "correct_median": statistics.median(float(x) for x in cor_on),
                "wrong_median": statistics.median(float(x) for x in wrg_on),
                "separation": statistics.median(float(x) for x in cor_on)
                - statistics.median(float(x) for x in wrg_on),
            },
        }
        print(f"  [{L}] 完成：off recall={r_off}/{n}，on recall={r_on}/{n}", flush=True)

    # ---- 表 1：recall@1 ----
    print("\n" + "=" * 64)
    print(f"表1  逐视图 recall@1（n={n}）")
    print("=" * 64)
    print(f"{'视图':<14}{'白化关':>20}{'白化开':>20}")
    for L in LAYERS:
        ro = results[L]["off"]["recall"]
        rn = results[L]["on"]["recall"]
        print(f"{L:<14}{f'{ro}/{n} = {ro/n:.1%}':>20}{f'{rn}/{n} = {rn/n:.1%}':>20}")

    # ---- 表 2：分离度 ----
    print("\n" + "=" * 84)
    print(f"表2  正确命中中位 / 错误匹配中位 / 分离度（n={n}，余弦相似度）")
    print("=" * 84)
    print(f"{'视图':<14}{'白化关 正确/错误/分离':>30}{'白化开 正确/错误/分离':>30}")
    for L in LAYERS:
        o = results[L]["off"]
        nw = results[L]["on"]
        s_off = f"{o['correct_median']:.4f} / {o['wrong_median']:.4f} / {o['separation']:+.4f}"
        s_on = f"{nw['correct_median']:.4f} / {nw['wrong_median']:.4f} / {nw['separation']:+.4f}"
        print(f"{L:<14}{s_off:>30}{s_on:>30}")

    OUT.write_text(
        json.dumps({"n": n, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写 {OUT}")


if __name__ == "__main__":
    main()
