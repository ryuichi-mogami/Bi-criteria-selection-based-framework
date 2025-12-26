#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

# ---------- 表の値 ----------
problems = ["DTLZ1","DTLZ2","DTLZ3","DTLZ4","DTLZ5","DTLZ6","DTLZ7"]

# B-NSGA-II（5万評価）
bnsga2 = {
    2: [0.0011, 0.0007, 0.0032, 0.0946, 0.0020, 0.0544, 0.0020],
    4: [0.0085, 0.0214, 0.0230, 0.0202, 0.0078, 0.0558, 0.0177],
    6: [0.0616, 0.0517, 0.0362, 0.0462, 0.0071, 6.5377, 0.4502],
}

# R-NSGA-II（ROI-C, 5万評価）
rnsga2 = {
    2: [0.0094, 0.0484, 0.0648, 0.0581, 0.0331, 0.1015, 0.0288],
    4: [0.0310, 0.0192, 0.1029, 0.0483, 0.0072, 0.0867, 0.1345],
    6: [0.0611, 0.1560, 0.1298, 0.0994, 0.0745, 0.6687, 1.2346],
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

ms = [2, 4, 6]  # 図を作る目的数

x = np.arange(len(problems))

for m in ms:
    fig, ax = plt.subplots(figsize=(9,6))

    y_b = bnsga2[m]
    y_r = rnsga2[m]

    # 線グラフ
    ax.plot(x, y_b, marker="o", markersize=20, linestyle="", label="B-NSGA-II", color=(255/255, 127/255, 14/255))
    ax.plot(x, y_r, marker="o", markersize=20, linestyle="", label="R-NSGA-II", color=(31/255, 119/255, 180/255))

    # ax.set_title(f"m = {m}", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=45, ha="right", fontsize=30)

    # ---- 縦軸：log スケール ----
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    # ax.set_ylabel("Indicator value", fontsize=14)
    ax.set_yticks([])
    ax.tick_params(axis="both", labelsize=30)
    ax.grid(which="major", linestyle="-", alpha=0.6)
    # ax.legend(fontsize=12)

    plt.tight_layout()
    outname = f"../output/ROI-C_scatter_m{m}.png"
    plt.savefig(outname, dpi=600)
    plt.close(fig)

    print(f"saved: {outname}")
