# -*- coding: utf-8 -*-
"""用候选级缓存做门控阈值重放（论文口径：CVE 级、不展开派生流程）。

    admit(x) = F(x) ∧ [ s(x) ≥ θ_s ∨ ( v(x) ≥ τ ∧ a(x) ≥ θ_a ∧ s(x) ≥ θ_w ) ]

用法：
    python utils/experiments/gate_grid_v3.py --formula new|old
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
THETA_A, THETA_W = 0.35, 0.20


def load(formula: str):
    parts = []
    for p in sorted((ROOT / "reports").glob(f"gate_cache_v3_{formula}_shard*.npz")):
        d = np.load(p)
        parts.append({k: d[k] for k in d.files})
    if not parts:
        raise SystemExit(f"未找到 {formula} 的缓存分片")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return out


def metrics(d, ts, tau, use_f=True):
    S, V, A, F = d["S"], d["V"], d["A"], d["F"]
    cve, IS_KB, SELF = d["cve"], d["IS_KB"], d["SELF"]
    lex = S >= ts
    sem = (V >= tau) & (A >= THETA_A) & (S >= THETA_W)
    admit = (F & (lex | sem)) if use_f else (lex | sem)
    kb = np.unique(cve[IS_KB & SELF & admit]).size
    held = np.unique(cve[(~IS_KB) & admit]).size
    return kb, held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formula", type=str, default="new", choices=["old", "new"])
    args = ap.parse_args()
    d = load(args.formula)
    n_kb = np.unique(d["cve"][d["IS_KB"]]).size
    n_held = np.unique(d["cve"][~d["IS_KB"]]).size
    print(f"公式 {args.formula}：候选 {len(d['S'])} 条，库内组 {n_kb} CVE，库外组 {n_held} CVE，"
          f"F(x) 通过候选 {int(d['F'].sum())} 条")
    uniq = sorted(set(np.round(d["S"], 4).tolist()))
    print(f"s(x) 取值集合（{len(uniq)} 个）: {uniq}")

    print("\n[1] 定稿参数点 θ_s=0.65, τ=0.65（论文口径）")
    for use_f in (True, False):
        kb, held = metrics(d, 0.65, 0.65, use_f)
        print(f"  {'F(x) 生效' if use_f else 'F(x) 删除'}: 召回 {kb/n_kb*100:5.1f}% ({kb}/{n_kb})  "
              f"错配 {held/n_held*100:5.1f}% ({held}/{n_held})")

    print("\n[2] θ_s 单维扫描（τ=0.65，F(x) 生效）")
    for ts in [round(0.30 + 0.05 * i, 2) for i in range(15)]:
        kb, held = metrics(d, ts, 0.65, True)
        print(f"  θ_s={ts:.2f}  召回 {kb:>3}/{n_kb} ({kb/n_kb*100:5.1f}%)   错配 {held:>3}/{n_held} ({held/n_held*100:5.1f}%)")

    print("\n[3] θ_s × τ 网格 召回%/错配%（F(x) 生效）")
    taus = [round(0.50 + 0.05 * i, 2) for i in range(8)]
    print("  θ_s\\τ " + "".join(f"{t:>10.2f}" for t in taus))
    for ts in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        line = f"  {ts:>4.2f}  "
        for tau in taus:
            kb, held = metrics(d, ts, tau, True)
            line += f"{kb/n_kb*100:>6.1f}/{held/n_held*100:<3.1f}"
        print(line)


if __name__ == "__main__":
    main()
