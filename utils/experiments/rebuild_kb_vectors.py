# -*- coding: utf-8 -*-
"""③b 为重建成候选库算出**全部 800 条层向量**，落成一个可以直接同步的文件。

为什么单独一步：
* 重建改的是**索引文本**（`semantic`/`full` 装入 `llm_semantic`，`solution` 重排 needle 片段），
  文本一变、向量就失效，必须重嵌；
* 嵌向量是**纯 CPU** 的（distilbert 本地、白化按层），**不需要 GPU** ——
  趁实例被占用时把这一步做完，等它空出来只剩"起 Weaviate + 同步 800 条对象"。

## 忠实性自检（这一条必须做对，否则整批向量不可信）

线上 dump 里每个 (条目, 层) 都带**旧层文本**和**旧向量**，所以能精确分成两类：

* **旧文本 == 新文本** → 这次重嵌**必须**复现线上向量（余弦 ≈ 1）。
  对不上就说明重嵌口径与线上不一致（比如白化没生效、层名传错），这批向量**不能用**。
* **旧文本 != 新文本** → 向量本来就会不同，那是重建的目的，不是错误。

（第一版把两类混在一起取平均，得到的数字既不能证明一致、也不能定位问题 —— 已改掉。）

## 用法

    python utils/experiments/rebuild_kb_vectors.py --db reports/mas_rebuild_candidate.db
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

LAYERS = ("semantic", "code_pattern", "solution", "full")


def load_live(path: Path) -> dict:
    """{layer: {sqlite_id: {"text":…, "vec":[…]}}}（线上现状，用于自检）。"""
    out = {L: {} for L in LAYERS}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            L = str(r.get("vector_layer") or "")
            if L in out and isinstance(r.get("_vector"), list) and r["_vector"]:
                out[L][int(r["sqlite_id"])] = {
                    "text": str(r.get("layer_text") or ""),
                    "vec": [float(x) for x in r["_vector"]],
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_rebuild_candidate.db")
    ap.add_argument("--live-dump", type=Path, default=ROOT / "reports/weaviate_kb_dump_today.jsonl",
                    help="线上现状（今天的 dump），既用于自检也用于判断哪些 (条目,层) 真的变了")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/kb_rebuild_vectors.jsonl")
    ap.add_argument("--report", type=Path, default=ROOT / "reports/kb_rebuild_vectors_report.json")
    ap.add_argument("--narrow", action="store_true",
                    help="窄口径：构造层文本时把 file_pattern/class_pattern 剔掉，"
                         "从而**逐字节复现线上文本**（只让 llm_semantic 与 needle 重排生效）")
    ap.add_argument("--emit", choices=("changed", "all"), default="changed",
                    help="只写「真的变了」的 (条目,层)，还是全部 800 条")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    from infrastructure.database.weaviate.service import WeaviateVectorService

    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    svc = WeaviateVectorService()

    con = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    cols = [r[1] for r in con.execute("pragma table_info(issue_patterns)")]
    rows = [dict(zip(cols, r)) for r in con.execute("select * from issue_patterns")]
    con.close()
    live = load_live(args.live_dump)
    print("候选库 %s：%d 条；线上 dump 可对比 %d 个 (条目, 层) 对"
          % (args.db.name, len(rows), sum(len(v) for v in live.values())))

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    same_layer_checks = defaultdict(list)     # 文本未变 → 应当复现
    changed_pairs = defaultdict(int)          # 文本变了 → 预期不同
    written = 0
    skipped_unchanged = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            sid = int(row["id"])
            props = dict(row, sqlite_id=sid, status=row.get("status") or "active")
            if args.narrow:
                # 窄口径：让 code_pattern/full 两层**逐字节复现线上文本**。
                # 线上那两个字段是空的（索引建于回填之前），而库里已经有值 —— 这是
                # 《01》问题 3 描述的漂移。本轮只想让 llm_semantic 与 needle 重排生效，
                # 所以把这两个字段临场剔掉，问题 3 留给它自己的实验。
                props = dict(props, file_pattern="", class_pattern="")
            for L in LAYERS:
                text = svc._build_enhanced_issue_pattern_text(props, L)
                old = live.get(L, {}).get(sid)
                if old is not None and old["text"] == text:
                    # 文本没变 → 向量也不该变；顺便做忠实性自检
                    vec = agent._default_embed(text, L)
                    same_layer_checks[L].append(dot(vec, old["vec"]))
                    if args.emit == "changed":
                        skipped_unchanged += 1
                        continue
                else:
                    changed_pairs[L] += 1
                    vec = agent._default_embed(text, L)
                fh.write(json.dumps({"sqlite_id": sid, "vector_layer": L,
                                     "layer_text": text, "_vector": vec}, ensure_ascii=False) + "\n")
                written += 1

    print("\n" + "=" * 96)
    print("忠实性自检：**层文本没变**的 (条目, 层) 对，重嵌向量必须与线上一致")
    print("=" * 96)
    bad = 0
    for L in LAYERS:
        vals = same_layer_checks.get(L) or []
        if not vals:
            print("  %-13s 文本未变的对比对：0" % L)
            continue
        near = sum(1 for v in vals if v > 0.999)
        worst = min(vals)
        if near != len(vals):
            bad += len(vals) - near
        print("  %-13s n=%3d  余弦>0.999 的 %3d 条  最低 %.4f  %s"
              % (L, len(vals), near, worst, "OK" if near == len(vals) else "*** 有不一致 ***"))
    print("\n  层文本**变了**的对（预期不同，不是错误）："
          + "  ".join("%s=%d" % (L, changed_pairs[L]) for L in LAYERS))
    print("  写出 %d 条向量；因文本未变而跳过 %d 条（--emit %s）" % (written, skipped_unchanged, args.emit))
    print("  （窄口径 %s）" % ("开" if args.narrow else "关"))

    args.report.write_text(json.dumps({
        "db": str(args.db), "vectors_written": written,
        "fidelity_failures": bad,
        "unchanged_pairs_checked": {L: len(same_layer_checks.get(L) or []) for L in LAYERS},
        "changed_pairs": {L: changed_pairs[L] for L in LAYERS},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n向量文件 -> %s\n报告 -> %s" % (args.out, args.report))
    print("\n下一步（仍未做）：把候选库写回线上库 + 把这份向量同步进 Weaviate（起 Weaviate 后几分钟）")


if __name__ == "__main__":
    main()
