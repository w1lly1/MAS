# -*- coding: utf-8 -*-
"""生成 4.8 节参数敏感性折线图（图2/图3/图4，各含 (a) 召回率 (b) 错配样本率 两个子图）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
OUT = ROOT / "论文" / "小论文初审" / "完整试验后迭代" / "r10"
S = json.loads((ROOT / "reports" / "param_sweep_v5_summary.json").read_text(encoding="utf-8"))
n_kb, n_held = S["n_kb"], S["n_held"]
BASE = (S["baseline"]["kb"] / n_kb * 100, S["baseline"]["held"] / n_held * 100)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300, "savefig.bbox": "tight"})

CURVES = {
    "θ_lex": (S["groups"]["θ_lex"], 0.50, "θlex"),
    "θ_loc": (S["groups"]["θ_loc"], 0.40, "θloc"),
    "θ_desc": (S["groups"]["θ_desc"], 0.10, "θdesc"),
    "θ_w": (S["groups"]["θ_w"], 0.20, "θw"),
    "τ": (S["groups"]["τ"], 0.65, "τ"),
    "θ_s": (S["groups"]["θ_s"], 0.65, "θs"),
}


def series(key, counts=False):
    rows, base_v, _ = CURVES[key]
    if counts:
        pts = sorted([(r["v"], r["kb"], r["held"]) for r in rows]
                     + [(base_v, S["baseline"]["kb"], S["baseline"]["held"])])
    else:
        pts = sorted([(r["v"], r["kb"] / n_kb * 100, r["held"] / n_held * 100) for r in rows]
                     + [(base_v, BASE[0], BASE[1])])
    xs = [p[0] for p in pts]
    y_rec = [p[1] for p in pts]
    y_mis = [p[2] for p in pts]
    return xs, y_rec, y_mis


LABELS = {"θ_lex": "θlex（词元权重）", "θ_loc": "θloc（定位权重）",
          "θ_desc": "θdesc（描述词面权重）", "θ_w": "θw（最弱结构阈值）",
          "τ": "τ（语义相似度阈值）", "θ_s": "θs（词法通道阈值）"}


def panel(ax, keys, which, ylim, title, note=None, xlim=None, counts=False):
    for k in keys:
        xs, y_rec, y_mis = series(k, counts)
        ys = y_rec if which == "rec" else y_mis
        ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.4, label=LABELS[k])
    ax.set_xlabel("参数取值")
    if counts:
        ax.set_ylabel("新增召回样本数/个" if which == "rec" else "新增误报样本数/个")
    else:
        ax.set_ylabel("召回率/%" if which == "rec" else "错配样本率/%")
    ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    if title:
        ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8.5, loc="best", framealpha=0.9)
    if note:
        ax.annotate(note, xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=8, color="#b00")


def make(name, keys, fname, rec_title, mis_title, rec_ylim, mis_ylim,
         xlim=None, note=None, rec_note=None, counts=False):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    panel(axes[0], keys, "rec", rec_ylim, rec_title, rec_note, xlim, counts)
    panel(axes[1], keys, "mis", mis_ylim, mis_title, note, xlim, counts)
    axes[0].text(-0.16, 1.02, "(a)", transform=axes[0].transAxes, fontsize=11)
    axes[1].text(-0.16, 1.02, "(b)", transform=axes[1].transAxes, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    print("已生成", fname)


make("图2", ["θ_lex", "θ_loc", "θ_desc"], "图4.png",
     "结构证据加权分三权重：新增召回样本数", "结构证据加权分三权重：新增误报样本数",
     rec_ylim=(55, 135), mis_ylim=(0, 24), xlim=(-0.03, 1.03), counts=True,
     note="θdesc≥0.65 时误报 200 个样本（超出纵轴范围）",
     rec_note="θloc=0 时新增召回 0 个（低于纵轴范围）")
make("图3", ["θ_w", "τ"], "图5.png",
     "语义通道判据参数：新增召回样本数", "语义通道判据参数：新增误报样本数",
     rec_ylim=(122, 129), mis_ylim=(0, 16), xlim=(0, 1.03), counts=True,
     note="两条误报曲线全程重合于 8 个样本")
make("图4", ["θ_s"], "图3.png",
     "词法通道阈值：新增召回样本数", "词法通道阈值：新增误报样本数",
     rec_ylim=(108, 136), mis_ylim=(4, 19), xlim=(0.25, 1.03), counts=True)
