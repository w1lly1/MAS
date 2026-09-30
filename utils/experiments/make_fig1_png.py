# -*- coding: utf-8 -*-
"""把 图1.svg（总体框架图）按原始坐标重绘为 图1.png，供 docx 直接嵌入。

原始 SVG 为 200×285 用户单位、设计宽度 50 mm；此处按同一坐标系重绘，
字号按"用户单位→点"换算，保证嵌入时与原设计比例一致。
同时把原图中残留的“候选准入规则与确定性过滤”改为“候选准入规则（置信门控）”。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(r"E:\MyOwn\ProgramStudy\MAS\论文\小论文初审\完整试验后迭代\r10\图1.png")

W, H = 200, 285
mm2in = 1 / 25.4
fig_w = 50 * mm2in          # 设计宽度 50 mm
pt_per_unit = fig_w * 72 / W
fig = plt.figure(figsize=(fig_w, 70 * mm2in), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)           # y 轴向下，与 SVG 一致
ax.axis("off")
plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei"]

EDGE = "#444444"
GRAY = "#f7f7f7"

BOXES = [
    (60, 8, 80, 26), (10, 70, 44, 56), (72, 62, 58, 38),
    (134, 62, 64, 38), (35, 126, 130, 28), (42, 176, 116, 26),
    (35, 224, 130, 26),
]
for x, y, w, h in BOXES:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4",
                                linewidth=1.2, edgecolor=EDGE, facecolor=GRAY,
                                mutation_aspect=1))


def txt(x, y, s, size_units):
    ax.text(x, y, s, ha="center", va="center", fontsize=size_units * pt_per_unit,
            color="#111111")


txt(100, 22, "待审代码", 11)
txt(32, 92, "历史缺陷", 9)
txt(32, 106, "知识库", 9)
txt(101, 76, "词法-结构通道", 9)
txt(101, 90, "词法子串主匹配", 9)
txt(166, 76, "线索确认通道", 9)
txt(166, 90, "同文件锚点弱过滤", 9)
txt(100, 141, "候选准入规则（置信门控）", 10)
txt(100, 190, "大模型（最终裁决）", 10)
txt(100, 238, "报告条目（带证据标识）", 10)
txt(100, 117, "召回候选", 8)
txt(112, 166, "通过候选", 8)

SOLID = [((100, 34), (101, 60)), ((100, 34), (166, 60)),
         ((101, 100), (88, 124)), ((166, 100), (112, 124)),
         ((100, 154), (100, 174)), ((100, 202), (100, 222))]
for (x1, y1), (x2, y2) in SOLID:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.2, color=EDGE,
                                 shrinkA=0, shrinkB=0))

DASHED = [((54, 98), (70, 78)), ((54, 98), (132, 78))]
for (x1, y1), (x2, y2) in DASHED:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.0, color="#888888",
                                 linestyle=(0, (4, 3)), shrinkA=0, shrinkB=0))

fig.savefig(OUT, dpi=600, facecolor="white")
print("已生成", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")
