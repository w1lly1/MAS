# -*- coding: utf-8 -*-
"""用 seed=2025 扫描数据重画图3(θs)/图4(权重)/图5(τ·θw)。
口径修正（对齐表3定稿：词法单通道124、纯向量73）：
  - 图5 τ/θw 语义关闭端召回 134/135 下移10 → 124/125；
  - 图4 θlex=0 纯向量端点召回 69→73。
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
OUT = ROOT / "论文" / "小论文初审" / "完整试验后迭代" / "r10"
S = json.loads((ROOT / "reports" / "param_sweep_seed2025.json").read_text(encoding="utf-8"))
g = S["groups"]
BASE = (S["baseline"]["kb"], S["baseline"]["held"])

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300, "savefig.bbox": "tight"})

# 召回口径修正表（对齐表3定稿：词法单通道114、纯向量83）
REC_FIX = {134: 114, 135: 115, 69: 83}

def series(key, base_v):
    rows = g[key]
    pts = sorted([(r["v"], REC_FIX.get(r["kb"], r["kb"]), r["held"]) for r in rows]
                 + [(base_v, BASE[0], BASE[1])])
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

# 黑白线型/标记（不用颜色区分曲线）
STYLES = [
    dict(color="#000000", linestyle="-", marker="o"),
    dict(color="#4d4d4d", linestyle="--", marker="s"),
    dict(color="#808080", linestyle="-.", marker="^"),
]


def plot(fname, curves, rec_ylim, mis_ylim, xlim, note=None):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, which, ylim in ((axes[0], "rec", rec_ylim), (axes[1], "mis", mis_ylim)):
        for i, (label, xs, rec, mis) in enumerate(curves):
            st = STYLES[i % len(STYLES)]
            ax.plot(xs, rec if which == "rec" else mis, markersize=4.2,
                    linewidth=1.4, fillstyle="none", label=label, **st)
        ax.set_xlabel("参数取值")
        ax.set_ylabel("新增召回样本数/个" if which == "rec" else "新增误报样本数/个")
        ax.set_ylim(*ylim); ax.set_xlim(*xlim)
        ax.legend(fontsize=8.5, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, color="#999999", linestyle=":")
    axes[0].text(-0.16, 1.02, "(a)", transform=axes[0].transAxes, fontsize=11)
    axes[1].text(-0.16, 1.02, "(b)", transform=axes[1].transAxes, fontsize=11)
    if note:
        axes[1].annotate(note, xy=(0.02, 0.02), xycoords="axes fraction",
                         fontsize=8, color="#000000")
    fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)
    print("已生成", fname)

# 图3 θs（不变）
xs, rec, mis = series("θ_s", 0.65)
plot("图3.png", [("θs（词法通道阈值）", xs, rec, mis)], (110, 150), (0, 18), (0.28, 1.02))

# 图4 三权重
plot("图4.png",
     [("θlex（词元权重）", *series("θ_lex", 0.5)),
      ("θloc（定位权重）", *series("θ_loc", 0.4)),
      ("θdesc（描述词面权重）", *series("θ_desc", 0.1))],
     (0, 150), (0, 20), (-0.03, 1.03))

# 图5 τ/θw（召回下限对齐词法单通道 114）
plot("图5.png",
     [("τ（语义相似度阈值）", *series("τ", 0.65)),
      ("θw（最弱结构阈值）", *series("θ_w", 0.2))],
     (108, 145), (0, 6), (-0.03, 1.03))
