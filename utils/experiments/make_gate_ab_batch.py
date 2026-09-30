# -*- coding: utf-8 -*-
"""造一个**平衡**的门控 A/B 批次：同等数量地取"库内(真召回样本)"与"库外真误报样本"。

## 为什么不能用现成的 smoke8

`smoke8` 是 1 个库内 + 7 个库外，召回侧只有 1 个样本 —— 门控改动里最值得看的
"召回有没有上去"根本没法定量。而重跑整个 400 样本代价太大（每个样本的产物约 90 MB，
400 个样本两轮就是几十 GB）。

折中：从**已经用过的 400 样本池**里各取一部分，组成 16 个样本的小批次。
好处是这些样本**就是主实验里的那些**，结论可以直接与主实验的分层数字对话；
而且不消耗新的 CVE（跨种子去重登记表不动）。

## 选样规则（可复现）

* `kb-self`  —— 条目在库、且同文件没有别的条目（干净的召回样本）
* `held-pure` —— 条目不在库、同文件也不在库（**真误报**样本）

刻意**排除** `kb-shared-file` / `held-same-file` 这两层：它们混着"同文件撞车"的第三类现象，
会让"召回↑/误报↑"的方向判断变得含糊。先把干净的两层对齐，再看第三类怎么动。

## 用法

    python utils/experiments/make_gate_ab_batch.py --kb 8 --held 8 \
        --out utils/experiments/gate_ab16.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.eval_strata import G_HELD_PURE, G_KB_SELF, build_strata  # noqa: E402

MANIFEST = ROOT / "reports/negative_exp_manifest_400_error.json"
DB = ROOT / "infrastructure/database/mas.db"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=ROOT / "utils/experiments/gate_ab16.json")
    ap.add_argument("--kb", type=int, default=8)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20240930)
    args = ap.parse_args()

    src = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = src["rows"]
    strata = build_strata([r["cve"] for r in rows], DB)

    by_group = {}
    for r in rows:
        g = strata.get(str(r["cve"]).strip().upper(), {}).get("group", "")
        by_group.setdefault(g, []).append(r)

    rng = random.Random(args.seed)
    picked = []
    for g, n in ((G_KB_SELF, args.kb), (G_HELD_PURE, args.held)):
        pool = sorted(by_group.get(g, []), key=lambda r: r["cve"])
        if len(pool) < n:
            raise SystemExit("池子里 %s 只有 %d 个，取不出 %d 个" % (g, len(pool), n))
        picked += rng.sample(pool, n)
    picked.sort(key=lambda r: (strata[str(r["cve"]).strip().upper()]["group"], r["cve"]))

    items = []
    for r in picked:
        cve = str(r["cve"]).strip().upper()
        g = strata[cve]["group"]
        items.append({
            "role": "kb" if g in (G_KB_SELF,) else "held",
            "cve": cve,
            "project": r.get("project"),
            "target_dir": r.get("before"),
            "output_dir": cve,
            "eval_group": g,
        })

    cfg = {
        "description": "门控 A/B 平衡批次：%d 个干净召回样本(kb-self) + %d 个真误报样本(held-pure)"
                       % (args.kb, args.held),
        "source_manifest": str(args.manifest.relative_to(ROOT)),
        "sampling": {"seed": args.seed, "exclude_groups": ["kb-shared-file", "held-same-file"],
                     "why": "先把干净的两层对齐，避免第三类现象(同文件撞 CVE)把方向搞混"},
        "items": items,
    }
    args.out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("批次已写出: %s" % args.out)
    print("  %-16s %-14s %-10s %s" % ("CVE", "分层", "role", "文件"))
    for it in items:
        st = strata[it["cve"]]
        print("  %-16s %-14s %-10s %s" % (
            it["cve"], it["eval_group"], it["role"],
            (st["file_keys"][0] if st["file_keys"] else "?")[:46]))
    print("\n  预期：kb-self %d 个（测召回）/ held-pure %d 个（测真误报）"
          % (args.kb, args.held))


if __name__ == "__main__":
    main()
