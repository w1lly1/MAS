# -*- coding: utf-8 -*-
"""KB 白化一致性校验 —— 确认 Weaviate 索引向量与查询侧向量处于同一（白化）空间。

【为什么必须查】
    embed_text(text, layer) 在白化文件存在且 layer 命中时，会把向量投影到 k 维白化子空间
    再零填充回 768 维 → 前 k 维非零、其余恒为 0.0；
    若 layer=None 或层名写错，则跳过白化 → 768 维全非零。
    两者不在同一空间，余弦相似度失去意义。

    危险场景：知识库入库时刻**早于** whitening_transform.json 生成时刻，则库里存的是
    未白化向量，而检索时查询向量是白化的。这种失配**不报错、无日志**，只会让跨文件
    语义命中静默归零（并被误读为"模型没有区分度/需微调"）。

    已知时间线（本仓库 mas.db）：
        issue_patterns.created_at  = 2026-09-02 15:08–15:09   ← KB 入库
        whitening_transform.json   = 2026-09-02 21:55         ← 白化文件生成（晚 6.5h）
    因此该库**除非之后做过全量重同步**，否则索引为原始向量。

【判定方法（无需人工标注）】
    对每层统计每条向量的非零维数：
        全部 == 该层 k（来自 whitening_transform.json）  → WHITENED_OK
        全部 == 768（无零元素）                          → RAW_UNWHITENED
        其它（混合/部分）                                → MISMATCH
        记录无 _vector                                   → NO_VECTOR

【用法】（MAS 根目录；--from-weaviate 需要 weaviate-client，请用项目 venv）
    # 1) 自检：本地 embedder 是否为各层产出 k 维非零（不依赖 Weaviate，随时可跑）
    venv\\Scripts\\python.exe utils\\experiments\\check_kb_whitening.py --self-test

    # 2) 检查已有 dump（JSONL，含 _vector 字段）
    venv\\Scripts\\python.exe utils\\experiments\\check_kb_whitening.py --dump reports\\weaviate_kb_seed2024.jsonl

    # 3) 从 Weaviate 现场拉取 → 落盘 → 立即检查
    venv\\Scripts\\python.exe utils\\experiments\\check_kb_whitening.py --from-weaviate

    # 4) 无参数：自动扫描 reports/ 与 utils/experiments/ 下的 *weaviate_kb*.jsonl
    venv\\Scripts\\python.exe utils\\experiments\\check_kb_whitening.py

【退出码】
    0 = 全部通过（索引已白化，与查询侧同空间）
    1 = 存在未通过项（RAW_UNWHITENED / MISMATCH / 缺向量 / 缺层）→ 可用于批处理前置闸门
    2 = 无法验证（如 Weaviate 不可达而只能 skip）→ 不得当作"已通过"
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WHITENING_PATH = ROOT / "infrastructure" / "embeddings" / "whitening_transform.json"

# 判定标签
OK = "WHITENED_OK"
RAW = "RAW_UNWHITENED"
MISMATCH = "MISMATCH"
NO_VEC = "NO_VECTOR"


# --------------------------------------------------------------------------- #
# 期望值：各层 k
# --------------------------------------------------------------------------- #
def load_expected_k() -> dict:
    """读取 whitening_transform.json → {layer: k}。文件缺失/损坏返回 {}（表示未启用白化）。"""
    if not WHITENING_PATH.exists():
        return {}
    try:
        data = json.loads(WHITENING_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 白化文件解析失败({e})，视为未启用白化：{WHITENING_PATH}")
        return {}
    out = {}
    for layer, entry in (data or {}).items():
        W = (entry or {}).get("W") or []
        k = (entry or {}).get("k")
        if not isinstance(k, int) or k <= 0:
            k = len(W[0]) if W and W[0] else 0
        out[str(layer)] = int(k)
    return out


# --------------------------------------------------------------------------- #
# 读 dump
# --------------------------------------------------------------------------- #
def iter_dump(path: Path):
    """逐行产出 (record, vector)。vector 不存在时为 None。"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            vec = rec.get("_vector")
            yield rec, (list(vec) if vec else None)


def classify(vectors, k: int, dim_expected: int = 768) -> dict:
    """按非零维数判定该层是否已白化。"""
    if not vectors:
        return {"verdict": NO_VEC, "n": 0, "detail": "该层无记录"}
    dims = {len(v) for v in vectors}
    if dims != {dim_expected}:
        return {
            "verdict": MISMATCH,
            "n": len(vectors),
            "detail": f"向量维度不统一/异常: {sorted(dims)}（期望 {dim_expected}）",
        }
    nz = [sum(1 for x in v if x != 0.0) for v in vectors]
    uniq = sorted(set(nz))
    if uniq == [k]:
        return {"verdict": OK, "n": len(vectors), "nonzero": uniq,
                "detail": f"全部为非零维数 {k}（= 该层 k），零填充正确"}
    if uniq == [dim_expected]:
        return {"verdict": RAW, "n": len(vectors), "nonzero": uniq,
                "detail": f"全部 768 维非零，无零填充 → 该层为【未白化】原始向量"}
    return {
        "verdict": MISMATCH,
        "n": len(vectors),
        "nonzero": uniq,
        "detail": f"非零维数不统一 = {uniq}（期望全部 {k}；768 表示未白化）",
    }


def check_dump(path: Path, expected_k: dict, strict_layers: bool = True) -> bool:
    """检查单个 dump。返回 True 表示该 dump 通过。"""
    print("=" * 78)
    print(f"检查 dump: {path}")

    by_layer = collections.defaultdict(list)
    n_total = 0
    n_no_vec = 0
    for rec, vec in iter_dump(path):
        n_total += 1
        if vec is None:
            n_no_vec += 1
            continue
        by_layer[str(rec.get("vector_layer") or "<missing>")].append(vec)

    size = path.stat().st_size if path.exists() else 0
    print(f"  记录 {n_total} 条，含向量 {n_total - n_no_vec} 条，无向量 {n_no_vec} 条，文件 {size/1024/1024:.1f} MB")
    if not expected_k:
        print("  ⚠️ 未找到 whitening_transform.json → 无法判定期望 k；仅报告非零维数分布。")
    else:
        print(f"  期望 k（来自 whitening_transform.json）: {expected_k}")

    if n_no_vec and not by_layer:
        print(f"  ❌ {NO_VEC}：dump 不含 _vector 字段。请用 include_vector=True 重新导出。")
        print("     （可执行：--from-weaviate 重新拉取）")
        return False

    passed = True
    seen_layers = set()
    for layer in sorted(by_layer):
        seen_layers.add(layer)
        k = expected_k.get(layer)
        vecs = by_layer[layer]
        if k is None:
            nz = sorted({sum(1 for x in v if x != 0.0) for v in vecs})
            print(f"  · {layer:14s} n={len(vecs):5d}  非零维数={nz}  → ⚠️ 无对应白化层，跳过判定")
            continue
        res = classify(vecs, k)
        mark = "✅" if res["verdict"] == OK else "❌"
        print(f"  {mark} {layer:14s} n={len(vecs):5d}  {res['verdict']:16s} {res['detail']}")
        if res["verdict"] != OK:
            passed = False

    # 层完整性：每个实体应有 transform 中的全部层
    if expected_k and strict_layers:
        missing = set(expected_k) - seen_layers
        if missing:
            print(f"  ❌ 缺失层: {sorted(missing)}（transform 声明 {sorted(expected_k)}）→ 同步不完整")
            passed = False

    print(f"  结论: {'✅ 通过（索引已白化，与查询侧同空间）' if passed else '❌ 未通过 —— 索引与查询可能不在同一空间'}")
    return passed


# --------------------------------------------------------------------------- #
# 现场拉取
# --------------------------------------------------------------------------- #
def dump_from_weaviate(out_path: Path, batch: int = 1000) -> bool:
    """用项目自身的 WeaviateVectorService 拉取全部对象（含向量）并落盘。"""
    try:
        from infrastructure.database.weaviate import WeaviateVectorService
    except Exception as e:  # noqa: BLE001
        print(f"❌ 无法导入 WeaviateVectorService（{e}）。--from-weaviate 需项目 venv。")
        return False

    svc = WeaviateVectorService()
    if not svc.connect(auto_create_schema=False):
        print("❌ Weaviate 未连接（检查 WEAVIATE_URL / 容器是否 Up）。")
        return False
    try:
        col = svc._get_collection()  # noqa: SLF001 —— 与 dump_weaviate_kb.py 同做法
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        cursor = None
        with out_path.open("w", encoding="utf-8") as f:
            while True:
                kwargs = dict(limit=batch, include_vector=True)
                if cursor:
                    kwargs["after"] = cursor
                res = col.query.fetch_objects(**kwargs)
                objs = list(res.objects)
                if not objs:
                    break
                for obj in objs:
                    props = dict(obj.properties)
                    vec = obj.vector
                    if isinstance(vec, dict):  # named vectors
                        vec = vec.get("default") or next(iter(vec.values()), None)
                    props["_vector"] = list(vec) if vec is not None else None
                    f.write(json.dumps(props, ensure_ascii=False) + "\n")
                    n += 1
                if len(objs) < batch:
                    break
                cursor = objs[-1].uuid
        print(f"已导出 {n} 个对象 → {out_path}")
        return n > 0
    finally:
        try:
            svc.disconnect()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 自检（不依赖 Weaviate）
# --------------------------------------------------------------------------- #
def self_test() -> bool:
    """确认本地 embedder：显式 layer → 恰好 k 维非零；layer=None → 未白化（768 维非零）。"""
    print("=" * 78)
    print("自检：本地 embedder 白化行为")
    expected_k = load_expected_k()
    if not expected_k:
        print("⚠️ 未找到 whitening_transform.json → 白化未启用，无法自检。")
        return True

    from infrastructure.embeddings.codebert_embedder import embed_text

    probe = "int main(void){ char buf[8]; memcpy(buf, src, len); return 0; }"
    ok = True
    print(f"  探针文本: {probe!r}\n")
    for layer, k in sorted(expected_k.items()):
        v = embed_text(probe, layer)
        nz = sum(1 for x in v if x != 0.0)
        good = (len(v) == 768 and nz == k)
        ok = ok and good
        print(f"  {'✅' if good else '❌'} layer={layer:14s} dim={len(v)} 非零={nz:4d} 期望={k}")

    v0 = embed_text(probe, None)
    nz0 = sum(1 for x in v0 if x != 0.0)
    print(f"\n  layer=None  → dim={len(v0)} 非零={nz0}（应为 768，即未白化）")
    print("  ↑ 上面若出现 [embedder] ⚠️ 告警，即为新加的静默失配兜底在生效（预期行为）。")
    print(f"\n  自检结论: {'✅ 通过' if ok else '❌ 未通过'}")
    return ok


# --------------------------------------------------------------------------- #
def discover_dumps() -> list:
    pats = [
        str(ROOT / "reports" / "**" / "*weaviate_kb*.jsonl"),
        str(ROOT / "utils" / "experiments" / "*weaviate_kb*.jsonl"),
    ]
    found = []
    for p in pats:
        found.extend(glob.glob(p, recursive=True))
    return sorted({Path(f) for f in found})


def main() -> int:
    ap = argparse.ArgumentParser(
        description="KB 白化一致性校验：索引向量与查询向量是否同空间",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dump", action="append", default=[],
                    help="要检查的 JSONL dump（含 _vector）；可重复")
    ap.add_argument("--from-weaviate", action="store_true",
                    help="从 Weaviate 现场拉取（并自动检查）")
    ap.add_argument("--out", default=str(ROOT / "reports" / "weaviate_kb_dump_check.jsonl"),
                    help="--from-weaviate 的落盘路径")
    ap.add_argument("--self-test", action="store_true",
                    help="仅自检本地 embedder 的白化行为（不依赖 Weaviate）")
    ap.add_argument("--no-strict-layers", action="store_true",
                    help="不要求 dump 覆盖 transform 声明的全部层")
    args = ap.parse_args()

    expected_k = load_expected_k()
    print(f"白化变换: {WHITENING_PATH}")
    print(f"  {'存在，各层 k = ' + str(expected_k) if expected_k else '不存在或为空 → 白化未启用'}")

    results = []  # (label, status) status ∈ {"ok", "fail", "skip"}

    if args.self_test:
        results.append(("self-test（本地 embedder 白化行为）", "ok" if self_test() else "fail"))

    targets = [Path(p) for p in args.dump]
    if args.from_weaviate:
        out = Path(args.out)
        if dump_from_weaviate(out):
            targets.append(out)
        else:
            results.append((f"现场拉取 {out.name}（Weaviate 不可达，未能验证）", "skip"))
    if not args.self_test and not args.dump and not args.from_weaviate:
        targets = discover_dumps()
        if not targets:
            print("\n未发现可检查的 dump。请用 --dump <path> 或 --from-weaviate，或 --self-test。")
            return 1
        print(f"\n自动发现 {len(targets)} 个 dump：")
        for t in targets:
            print(f"  - {t.relative_to(ROOT)}")

    for t in targets:
        if not t.exists():
            print(f"\n❌ 文件不存在: {t}")
            results.append((str(t), "fail"))
            continue
        ok = check_dump(t, expected_k, strict_layers=not args.no_strict_layers)
        results.append((str(t), "ok" if ok else "fail"))

    print("=" * 78)
    print("汇总")
    n_ok = sum(1 for _, s in results if s == "ok")
    n_fail = sum(1 for _, s in results if s == "fail")
    n_skip = sum(1 for _, s in results if s == "skip")
    for name, status in results:
        mark = {"ok": "✅", "fail": "❌", "skip": "⚠️"}[status]
        label = name if len(name) < 60 else "…" + name[-57:]
        print(f"  {mark} {label}")

    if n_fail:
        print(f"\n❌ {n_fail} 项未通过。若某层判定为 RAW_UNWHITENED，说明该库写入时白化尚未启用，")
        print("   需对该层做全量重同步（例：utils/experiments/resync_weaviate.py）后再评测。")
        return 1
    if n_skip:
        print(f"\n⚠️ 无法验证（{n_skip} 项跳过；已通过 {n_ok} 项）。"
              f"请确认 Weaviate 在 line 或改用 --dump 提供证据，不要当成『已通过』。")
        return 2
    print(f"\n✅ 全部通过（{n_ok} 项）：索引向量与查询向量处于同一白化空间。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
