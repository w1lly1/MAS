# -*- coding: utf-8 -*-
"""跨批次 / 跨随机种子的 CVE 使用登记表（避免两个知识库出现同一批 CVE）

## 为什么需要它

原抽样实现是：

    rng = random.Random(seed)
    return rng.sample(cves, n)          # 每个种子都从**全池**独立抽

不同种子各自从全池独立抽样，**必然互相重叠**。于是"用另一个种子独立复验"
这句话在数据上并不成立：两次划分会共用一批 CVE，知识库里也会出现同一批 CVE。
实测：两个都用 seed=2024 的批次之间，仅"建库组"就重叠 27 条。

## 它怎么解决

维护一份登记表，记录**历史上所有批次用过哪些 CVE**。
新批次抽样时先把这些排除掉，抽完再登记回去。
这样任意两次抽样的 CVE 集合天然不相交，且**可复现**（同样的登记表 + 同样的种子 → 同样的结果）。

## 登记表长什么样

    {
      "_comment": "...",
      "batches": {
        "negative_exp_400@seed2024": {"n": 400, "roles": {"kb": 200, "held": 200},
                                      "cves": ["CVE-...", ...]},
        ...
      },
      "used": ["CVE-...", ...]        # 上面所有批次 cves 的并集（冗余存一份，便于人工查看）
    }
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "reports" / "cve_usage_registry.json"

# 已知的历史批次/清单（用于首次建立登记表）
KNOWN_SOURCES = [
    ROOT / "reports/negative_exp_manifest_400.json",
    ROOT / "reports/negative_exp_manifest_400_error.json",
    ROOT / "reports/negative_exp_manifest.json",
    ROOT / "reports/manifest_400_missing.json",
    ROOT / "reports/five_fold_manifest.json",
    ROOT / "reports/held_out_manifest.json",
    ROOT / "论文/test_400_batch.json",
    ROOT / "论文/test_400_error_batch.json",
    ROOT / "论文/test_400_error_batch_remaining.json",
    ROOT / "论文/test_200_batch.json",
    ROOT / "论文/test_50_codebert_batch.json",
    ROOT / "论文/negative_test_batch.json",
]


# --------------------------------------------------------------------------- #
# 读取任意批次/清单文件，抽出 CVE 与分组
# --------------------------------------------------------------------------- #
def extract_cves(path: Path) -> Dict[str, object]:
    """返回 {"cves": set, "roles": {role: set}, "seed": ...}。兼容 dict / list。"""
    if not path.exists():
        return {"cves": set(), "roles": {}, "seed": None}
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list):
        d = {"rows": d}
    cves: Set[str] = set()
    roles: Dict[str, Set[str]] = {}
    for r in (d.get("rows") or d.get("items") or []):
        if not isinstance(r, dict):
            continue
        c = str(r.get("cve") or "").strip()
        if not c:
            continue
        cves.add(c)
        role = str(r.get("role") or "").strip() or "unlabeled"
        roles.setdefault(role, set()).add(c)
    if not cves:  # manifest 里的 *_cves 字段
        for c in (d.get("kb_cves") or []):
            cves.add(str(c)); roles.setdefault("kb", set()).add(str(c))
        for c in (d.get("held_cves") or []):
            cves.add(str(c)); roles.setdefault("held", set()).add(str(c))
    return {"cves": cves, "roles": roles, "seed": d.get("seed")}


# --------------------------------------------------------------------------- #
# 登记表读写
# --------------------------------------------------------------------------- #
def load_registry(path: Path = DEFAULT_REGISTRY) -> dict:
    if path.exists():
        reg = json.loads(path.read_text(encoding="utf-8"))
    else:
        reg = {"_comment": "跨批次/跨种子的 CVE 使用登记表；新批次抽样会排除 used 里的 CVE",
               "batches": {}, "used": []}
    reg.setdefault("batches", {})
    reg.setdefault("used", [])
    return reg


def save_registry(reg: dict, path: Path = DEFAULT_REGISTRY) -> None:
    used: Set[str] = set()
    for b in reg["batches"].values():
        used.update(b.get("cves") or [])
    reg["used"] = sorted(used)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")


def used_cves(reg: dict) -> Set[str]:
    s: Set[str] = set(reg.get("used") or [])
    for b in (reg.get("batches") or {}).values():
        s.update(b.get("cves") or [])
    return s


def add_batch(reg: dict, label: str, cves: Iterable[str], extra: Optional[dict] = None) -> None:
    entry = {"cves": sorted(set(cves))}
    if extra:
        entry.update(extra)
    reg["batches"][label] = entry


def bootstrap(registry_path: Path = DEFAULT_REGISTRY,
              sources: Sequence[Path] = tuple(KNOWN_SOURCES)) -> dict:
    """把已知历史批次全部登记进去（幂等：重复调用不会重复登记同一来源）。"""
    reg = load_registry(registry_path)
    added = []
    for p in sources:
        info = extract_cves(p)
        if not info["cves"]:
            continue
        label = "%s@seed%s" % (p.stem, info["seed"]) if info["seed"] else p.stem
        if label in reg["batches"]:
            continue
        roles = {k: len(v) for k, v in sorted(info["roles"].items())}
        add_batch(reg, label, info["cves"], {"source": str(p.relative_to(ROOT)),
                                             "n": len(info["cves"]), "roles": roles})
        added.append((label, len(info["cves"])))
    save_registry(reg, registry_path)
    return {"added": added, "used": len(used_cves(reg))}


# --------------------------------------------------------------------------- #
# 抽样（排除已用）
# --------------------------------------------------------------------------- #
def sample_cves(pool: Sequence[dict], n: int, seed: int,
                exclude: Optional[Set[str]] = None,
                require_full: bool = True) -> List[dict]:
    """从 pool 里随机抽 n 个，**排除 exclude 里的 CVE**。

    注意：原实现名为 stratified_sample 但并未做分层，这里保持"纯随机抽样"，
    只增加排除逻辑，以保证与既有结果口径一致（除排除外行为不变）。
    """
    exclude = exclude or set()
    avail = [c for c in pool if str(c.get("cve")) not in exclude]
    if len(avail) < n:
        msg = ("可用 CVE 不足：池 %d 条，已排除 %d 条，剩余 %d 条，需要 %d 条"
               % (len(pool), len(pool) - len(avail), len(avail), n))
        if require_full:
            raise SystemExit("❌ " + msg + "\n   可用 --no-exclude 关闭排除，或减小 --total。")
        print("⚠️ " + msg + "（按 require_full=False 仅取现有）")
        n = len(avail)
    rng = random.Random(seed)
    return rng.sample(avail, n)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="跨批次/跨种子的 CVE 使用登记表")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--bootstrap", action="store_true", help="把已知历史批次登记进去")
    ap.add_argument("--show", action="store_true", help="查看登记表内容")
    ap.add_argument("--check", type=Path, default=None,
                    help="检查某个批次文件与登记表的重叠情况")
    args = ap.parse_args()

    if args.bootstrap:
        r = bootstrap(args.registry)
        print("已新增登记 %d 个批次：" % len(r["added"]))
        for label, n in r["added"]:
            print("   %-52s %4d 条" % (label, n))
        print("登记表累计已用 CVE: %d 条" % r["used"])

    if args.check:
        reg = load_registry(args.registry)
        used = used_cves(reg)
        info = extract_cves(args.check)
        inter = info["cves"] & used
        print("\n检查 %s：" % args.check)
        print("  该批次 CVE %d 条；与登记表重叠 %d 条（%.1f%%）" % (
            len(info["cves"]), len(inter), 100 * len(inter) / max(1, len(info["cves"]))))
        if inter:
            print("  重叠样例:", sorted(inter)[:10])
        else:
            print("  ✅ 完全不重叠")

    if args.show or not (args.bootstrap or args.check):
        reg = load_registry(args.registry)
        print("\n登记表: %s" % args.registry)
        print("  批次数: %d   累计已用 CVE: %d" % (len(reg["batches"]), len(used_cves(reg))))
        for label, b in sorted(reg["batches"].items()):
            print("   %-52s n=%-4s roles=%s" % (label, b.get("n"), b.get("roles")))


if __name__ == "__main__":
    sys.exit(main())
