#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

# ---------- 表の値 ----------
problems = ["DTLZ1","DTLZ2","DTLZ3","DTLZ4","DTLZ5","DTLZ6","DTLZ7"]

# B-NSGA-II（5万評価）
bnsga2 = {
    2: [0.0054, 0.0014, 0.2986, 0.0004, 0.0018, 0.0591, 0.0002],
    4: [51.3542, 0.0524, 156.3671, 0.0256, 0.0022, 1.0080, 0.0037],
    6: [55.2288, 0.1921, 416.9930, 0.1357, 0.0047, 4.4647, 0.2704],
}

# R-NSGA-II（5万評価）
rnsga2 = {
    2: [0.0035, 0.0034, 0.3208, 0.0011, 0.0031, 0.0537, 0.0034],
    4: [39.8545, 0.0804, 165.5543, 0.0255, 0.0062, 4.0537, 0.0204],
    6: [51.4771, 0.4594, 412.0551, 0.4880, 0.4281, 4.5034, 0.2591],
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
    ax.plot(x, y_b, marker="o", markersize=10, linestyle="", label="B-NSGA-II")
    ax.plot(x, y_r, marker="s", markersize=10, linestyle="", label="R-NSGA-II")

    # ax.set_title(f"m = {m}", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=45, ha="right", fontsize=30)

    # ---- 縦軸：log スケール ----
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_yticks([])
    # ax.set_ylabel("Indicator value", fontsize=14)
    ax.tick_params(axis="both", labelsize=30)
    ax.grid(which="major", linestyle="-", alpha=0.6)
    # ax.legend(fontsize=12)

    plt.tight_layout()
    outname = f"../output/ROI-P_scatter_m{m}.png"
    plt.savefig(outname, dpi=600)
    plt.close(fig)

    print(f"saved: {outname}")
