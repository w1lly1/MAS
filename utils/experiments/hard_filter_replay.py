# -*- coding: utf-8 -*-
"""硬前置过滤 F(x) 消融（论文口径 replay，秒级）。

复用 reports/gate_400_cache.npz（400 样本全部候选的 s(x)/v(x)/a(x)/HP/self 标记），
在 θ_s × τ 网格上分别计算 F(x) 生效与 F(x)≡真 两种情况下的
库内组召回率与库外组错配率。

判别式：
    admit(x) = F(x) ∧ [ s(x) ≥ θ_s ∨ ( v(x) ≥ τ ∧ a(x) ≥ θ_a ∧ s(x) ≥ θ_w ) ]

注意口径：本脚本与论文正文一致（CVE 级、不展开派生流程），
其 F(x) 生效结果应与 gate_sweep_400_2d.py 完全一致（144/14 @ θ_s=0.65, τ=0.65）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "reports" / "gate_400_cache.npz"
OUT = ROOT / "reports" / "hard_filter_replay.json"

THETA_A = 0.35
THETA_W = 0.20


def metrics(cve_ids, S, V, A, HP, IS_KB, SELF, theta_s, tau, use_hp, n_kb, n_held):
    lex = S >= theta_s
    sem = (V >= tau) & (A >= THETA_A) & (S >= THETA_W)
    admit = (HP & (lex | sem)) if use_hp else (lex | sem)
    kb_hit = np.unique(cve_ids[IS_KB & SELF & admit]).size
    held_fp = np.unique(cve_ids[(~IS_KB) & admit]).size
    return kb_hit, held_fp, kb_hit / n_kb, held_fp / n_held


def main() -> None:
    d = np.load(CACHE)
    cve_ids, S, V, A = d["cve_ids"], d["S"], d["V"], d["A"]
    HP, IS_KB, SELF = d["HP"], d["IS_KB"], d["SELF"]
    n_kb = np.unique(cve_ids[IS_KB]).size
    n_held = np.unique(cve_ids[~IS_KB]).size
    print(f"候选 {len(S)} 条；库内组 {n_kb} CVE，库外组 {n_held} CVE；F(x) 通过候选 {int(HP.sum())} 条")

    rows = []
    print("\n[1] 定稿参数点 θ_s=0.65, τ=0.65")
    for use_hp in (True, False):
        kh, hf, rec, fp = metrics(cve_ids, S, V, A, HP, IS_KB, SELF, 0.65, 0.65, use_hp, n_kb, n_held)
        tag = "F(x) 生效（定稿）" if use_hp else "F(x) 删除"
        print(f"  {tag:20s} 召回 {rec*100:5.1f}% ({kh}/{n_kb})   错配 {fp*100:5.1f}% ({hf}/{n_held})")
        rows.append({"theta_s": 0.65, "tau": 0.65, "use_hp": use_hp,
                     "kb_hit": int(kh), "held_fp": int(hf),
                     "recall": round(rec, 4), "fp_rate": round(fp, 4)})

    print("\n[2] θ_s × τ 网格（召回% / 错配%），上=F(x)生效 下=F(x)删除")
    ts_vals = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    tau_vals = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    for use_hp in (True, False):
        print(("  " + "θ_s\\τ".ljust(8) + "".join(f"{t:>12.2f}" for t in tau_vals)))
        for ts in ts_vals:
            line = f"  {ts:>5.2f}   "
            for tau in tau_vals:
                kh, hf, rec, fp = metrics(cve_ids, S, V, A, HP, IS_KB, SELF, ts, tau, use_hp, n_kb, n_held)
                rows.append({"theta_s": ts, "tau": tau, "use_hp": use_hp,
                             "kb_hit": int(kh), "held_fp": int(hf),
                             "recall": round(rec, 4), "fp_rate": round(fp, 4)})
                line += f"{rec*100:>7.1f}/{fp*100:<4.1f}"
            print(line)

    # [3] F(x) 究竟拦掉了什么：以 self 候选（库内组自身条目）视角统计
    print("\n[3] F(x) 拦截的候选构成（θ_s=0.65, τ=0.65）")
    lex = S >= 0.65
    sem = (V >= 0.65) & (A >= THETA_A) & (S >= THETA_W)
    would = lex | sem
    blocked = would & ~HP
    self_blocked = blocked & SELF & IS_KB
    print(f"  通过阈值析取但被 F(x) 拦下的候选: {int(blocked.sum())} 条")
    print(f"  其中库内组自身条目(self): {int(self_blocked.sum())} 条，"
          f"涉及 CVE {np.unique(cve_ids[self_blocked]).size} 个")
    print(f"  其中库外组候选: {int((blocked & ~IS_KB).sum())} 条，"
          f"涉及 CVE {np.unique(cve_ids[blocked & ~IS_KB]).size} 个")
    gain_kb = np.unique(cve_ids[IS_KB & SELF & would]).size - np.unique(cve_ids[IS_KB & SELF & HP & would]).size
    gain_held = np.unique(cve_ids[(~IS_KB) & would]).size - np.unique(cve_ids[(~IS_KB) & HP & would]).size
    print(f"  删除 F(x) 后：库内组召回 +{gain_kb} 个 CVE，库外组错配 +{gain_held} 个 CVE")

    OUT.write_text(json.dumps({"n_kb_cve": int(n_kb), "n_held_cve": int(n_held), "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
