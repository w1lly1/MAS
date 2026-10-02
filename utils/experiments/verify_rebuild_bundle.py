# -*- coding: utf-8 -*-
"""上机器前的一致性核对：确认"重排后的库"与"要写进向量库的那 535 条"是配套的。

**为什么需要它**：真机那一步（切库 + 写 Weaviate 向量）有代价，一旦把**过期的向量**写进线上，
向量库与 SQLite 就会不一致——而且**不会报错**（《03》坑 29 的同款病）。
本次的具体风险：向量包 `kb_rebuild_upsert.jsonl` 是按候选库 **v1**（只改 `issue_patterns`）生成的，
之后又做了 **v2**（追加 `curated_issues` 重排）。必须证明两版**在 `issue_patterns` 上语义一致**、
且向量包里的层文本**逐条等于用 v2 重算出来的**，那份向量包才仍然配套。

核对三项：
  A. 两个候选库的 `issue_patterns` 除 `updated_at` 时间戳外是否逐行相同；
  B. 向量包的 (条目, 层) 集合是否 == "用 v2 + 线上 dump 重算后确实会变"的集合，
     且每条的 `layer_text` 是否逐字节等于重算值（口径与 `rebuild_kb_vectors.py --narrow` 一致）；
  C. `curated_issues` 的重排只影响 SQLite（向量包里不该有它）。

用法：python -X utf8 utils/experiments/verify_rebuild_bundle.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

OLD_DB = ROOT / "reports/mas_live.db"
V1_DB = ROOT / "reports/mas_rebuild_candidate.db"
V2_DB = ROOT / "reports/mas_rebuild_candidate_v2.db"
UPSERT = ROOT / "reports/kb_rebuild_upsert.jsonl"
DUMP = ROOT / "reports/weaviate_kb_dump_today.jsonl"
LAYERS = ("semantic", "code_pattern", "solution", "full")
IGNORE_COLS = {"updated_at"}          # 时间戳不参与语义比较


def table_rows(path: Path, table: str):
    con = sqlite3.connect("file:%s?mode=ro" % path.as_posix(), uri=True)
    cols = [r[1] for r in con.execute("pragma table_info(%s)" % table)]
    rows = {r[0]: dict(zip(cols, r)) for r in con.execute("select * from %s" % table)}
    con.close()
    return cols, rows


def load_dump(path: Path):
    """{层: {条目 id: 层文本}}（线上现状）。"""
    out = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        layer = str(row.get("vector_layer") or "")
        if layer in LAYERS:
            out[layer][int(row["sqlite_id"])] = str(row.get("layer_text") or "")
    return out


def main() -> None:
    ok_all = True

    print("=" * 88)
    print("A. 两个候选库的 issue_patterns 是否语义一致（> 决定向量包是否仍然配套）")
    print("=" * 88)
    cols1, r1 = table_rows(V1_DB, "issue_patterns")
    cols2, r2 = table_rows(V2_DB, "issue_patterns")
    print("  列一致: %s（%d 列）；行数 v1=%d v2=%d" % (cols1 == cols2, len(cols1), len(r1), len(r2)))
    diffs, ts_only = [], []
    for rid in sorted(set(r1) | set(r2)):
        a, b = r1.get(rid, {}), r2.get(rid, {})
        diff_cols = [c for c in cols1 if a.get(c) != b.get(c)]
        if diff_cols:
            diffs.append((rid, diff_cols))
            if set(diff_cols) <= IGNORE_COLS:
                ts_only.append(rid)
    semantic_diffs = [d for d in diffs if not set(d[1]) <= IGNORE_COLS]
    print("  有差异的行: %d，其中**只差时间戳**的: %d" % (len(diffs), len(ts_only)))
    if semantic_diffs:
        print("  ⚠ 语义不一致的行: %s" % semantic_diffs[:5])
    a_ok = cols1 == cols2 and set(r1) == set(r2) and not semantic_diffs
    ok_all &= a_ok
    print("  [%s] %s" % ("OK" if a_ok else "NG",
                        "两版在 issue_patterns 上语义一致 → 那 535 条向量仍然配套"
                        if a_ok else "两版不一致 → 向量包必须按 v2 重新生成！"))

    print()
    print("=" * 88)
    print("B. 向量包是否 == 用 v2 + 线上 dump 重算出的「会变的 (条目, 层)」")
    print("=" * 88)
    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent  # noqa: E402
    from infrastructure.database.weaviate.service import WeaviateVectorService  # noqa: E402

    svc = WeaviateVectorService()
    dump = load_dump(DUMP)
    _, v2_rows = table_rows(V2_DB, "issue_patterns")
    expected_changed, expected_text = set(), {}
    for rid, row in v2_rows.items():
        props = dict(row, sqlite_id=rid, status=row.get("status") or "active")
        props = dict(props, file_pattern="", class_pattern="")   # 与 --narrow 口径一致
        for layer in LAYERS:
            text = svc._build_enhanced_issue_pattern_text(props, layer)
            expected_text[(rid, layer)] = text
            if dump.get(layer, {}).get(rid) != text:
                expected_changed.add((rid, layer))

    upsert_rows = [json.loads(line) for line in UPSERT.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    got = {(int(r["sqlite_id"]), str(r["vector_layer"])): str(r.get("layer_text") or "")
           for r in upsert_rows}
    by_layer = defaultdict(int)
    for _, layer in got:
        by_layer[layer] += 1
    print("  向量包 %d 行，按层: %s" % (len(got), json.dumps(dict(by_layer), ensure_ascii=False)))
    print("  重算出的应变化集合 %d 个，按层: %s"
          % (len(expected_changed), json.dumps(
              {L: sum(1 for _, l in expected_changed if l == L) for L in LAYERS}, ensure_ascii=False)))
    missing = expected_changed - set(got)
    extra = set(got) - expected_changed
    text_mismatch = [(k, got[k], expected_text.get(k)) for k in got
                     if k in expected_text and got[k] != expected_text[k]]
    print("  集合差异: 缺失 %d 个 / 多余 %d 个" % (len(missing), len(extra)))
    if missing:
        print("     缺失示例: %s" % sorted(missing)[:5])
    if extra:
        print("     多余示例: %s" % sorted(extra)[:5])
    print("  layer_text 与 v2 重算值不一致的行: %d" % len(text_mismatch))
    if text_mismatch:
        k, got_t, exp_t = text_mismatch[0]
        print("     首个不一致 %s\n      包内: %r\n      重算: %r" % (k, got_t[:120], (exp_t or "")[:120]))
    b_ok = not missing and not extra and not text_mismatch
    ok_all &= b_ok
    print("  [%s] %s" % ("OK" if b_ok else "NG",
                        "向量包与 v2 逐条对得上，可以直接用来写线上"
                        if b_ok else "向量包与 v2 对不上 → 别把它写进线上"))

    print()
    print("=" * 88)
    print("C. curated_issues 的重排只影响 SQLite（向量包里不该有它）")
    print("=" * 88)
    _, c1 = table_rows(V1_DB, "curated_issues")
    _, c2 = table_rows(V2_DB, "curated_issues")
    changed = [rid for rid in c1 if {k: v for k, v in c1[rid].items() if k not in IGNORE_COLS}
               != {k: v for k, v in (c2.get(rid) or {}).items() if k not in IGNORE_COLS}]
    print("  curated_issues 行数 v1=%d v2=%d；语义有变化的行 = %d" % (len(c1), len(c2), len(changed)))
    print("  向量包条目 id 全部来自 issue_patterns（%s）"
          % ("是" if max(r["sqlite_id"] for r in upsert_rows) <= max(v2_rows) else "需人工确认"))

    print()
    print("=" * 88)
    print("总判定：%s" % ("可以上机器（库与向量包配套）" if ok_all else "**先别上机器**（上面有 NG）"))
    print("=" * 88)
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
