# -*- coding: utf-8 -*-
"""k(x) 简化验证：能否把 a(x) ≥ θ_a 从语义通道判据中去掉。

从候选级缓存读 A（锚点完整度）、V、S、F，分别按两种 k(x) 重放：
    原式  k(x) = [v(x) ≥ τ] ∧ [a(x) ≥ θ_a] ∧ [s(x) ≥ θ_w]
    简式  k(x) = [v(x) ≥ τ] ∧ [s(x) ≥ θ_w]
并给出 A 的取值分布，判断 a(x) 条件是否实际起作用。
"""
from __future__ import annotations

import sys
from collections import Counter
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
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


for formula in ("new", "old"):
    d = load(formula)
    A, V, S, F = d["A"], d["V"], d["S"], d["F"]
    cve, IS_KB, SELF = d["cve"], d["IS_KB"], d["SELF"]
    n_kb = np.unique(cve[IS_KB]).size
    n_held = np.unique(cve[~IS_KB]).size
    print(f"\n===== 公式 {formula}：候选 {len(S)} =====")
    dist = Counter(np.round(A, 4).tolist())
    print(f"a(x)（anchor_score）取值分布（{len(dist)} 个不同取值）:")
    for v, n in sorted(dist.items()):
        print(f"   a={v:<6} {n:>9} 条  {n/len(A)*100:5.1f}%")
    below = int((A < THETA_A).sum())
    print(f"   a(x) < θ_a={THETA_A} 的候选: {below} 条（{below/len(A)*100:.2f}%）")

    for ts, tau in ((0.65, 0.65),):
        for label, use_a in (("原式（含 a(x) ≥ θ_a）", True), ("简式（去 a(x) 条件）", False)):
            sem = (V >= tau) & (S >= THETA_W)
            if use_a:
                sem = sem & (A >= THETA_A)
            admit = F & ((S >= ts) | sem)
            kb = np.unique(cve[IS_KB & SELF & admit]).size
            held = np.unique(cve[(~IS_KB) & admit]).size
            print(f"   θ_s={ts} τ={tau}  {label:<20} 召回 {kb:>3}/{n_kb} ({kb/n_kb*100:5.1f}%)  "
                  f"错配 {held:>3}/{n_held} ({held/n_held*100:5.1f}%)")

# 语义支独立准入的候选里，有多少靠 a(x) 才被挡下
d = load("new")
A, V, S, F = d["A"], d["V"], d["S"], d["F"]
sem_no_a = (V >= 0.65) & (S >= THETA_W) & F & (S < 0.65)
sem_with_a = sem_no_a & (A >= THETA_A)
print(f"\n[补充] 仅靠语义支（未达 θ_s）且过 F(x) 的候选: {int(sem_no_a.sum())} 条，"
      f"其中 a(x) ≥ θ_a: {int(sem_with_a.sum())} 条 —— 差额即被 a(x) 挡下的候选数")
