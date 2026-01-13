#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Ellipse
import pandas as pd

# =========================
# ユーザー設定
# =========================
# ここだけ指定すればOK（描画したい解集合CSV）
mult_ref = 1
roi = "roi-c"
# 問題設定（PFとzの読み込みに使用）
alg = "RNSGA2"
m          = 2
prob       = "DTLZ2"
r          = 0.1
SOL_FILE  = f"../output/results_{mult_ref}/{roi}/{alg}/{prob}/m{m}/pop_4th_run_5000fevals.csv"
out_dir    = f'../output/2d_image_single_{mult_ref}/{prob}_m{m}_{roi}_{alg}'
dpi_save   = 600

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'

# ---------- Utility ----------
def make_endpoints_formatter(ideal_val, nadir_val):
    def _fmt(x, pos=None):
        if np.isclose(x, 0.0):
            return rf"${ideal_val:g}$"
        if np.isclose(x, 1.0):
            return rf"${nadir_val:g}$"
        # 中間値も実スケールに戻したいなら以下に変える
        # val = ideal_val + x * (nadir_val - ideal_val)
        # return rf"${val:g}$"
        return rf"${x:g}$"
    return _fmt

def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom

# ---------- 入出力 ----------
def load_pf(prob, m):
    pf_path = f'../ref_point_dataset/{prob}_d{m}_n50000.csv'
    return np.loadtxt(pf_path, delimiter=',')

def load_ref_point(prob, m, t):
    return np.loadtxt(f'../ref_point_data/{roi}/m{m}_{prob}_type{t}.csv', delimiter=',', ndmin=1)

# ---------- 描画 ----------
def plot_2d(PF, Pset, z):
    # ----- 基本量 -----
    true_ideal_x, true_nadir_x = PF[:, 0].min(), PF[:, 0].max()
    true_ideal_y, true_nadir_y = PF[:, 1].min(), PF[:, 1].max()
    true_ideal = PF.min(axis=0)
    true_nadir = PF.max(axis=0)
    I, N = (true_ideal, true_nadir)

    PF_norm  = normalize_points(PF,  I, N)
    P_norm   = normalize_points(Pset, I, N)
    ref_norm = normalize_points(z,    I, N)

    nearest_point = []
    # PF上でzに最も近い点（実スケールで探索 → 正規化座標で描画）
    for i in range(len(z)):
        nearest_point.append(PF_norm[np.argmin(np.linalg.norm(PF - z[i], axis=1))])

    # ----- Figure / Axes -----
    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    # PF
    ax.scatter(PF_norm[:, 0], PF_norm[:, 1],
               color='black', s=3, alpha=0.2, rasterized=True)
    # ref point
    for i in range(len(z)):
        ax.scatter(ref_norm[i][0], ref_norm[i][1],
                color=(44/255, 160/255, 44/255),
                marker='^', s=200,zorder=10)
    # solution set
    ax.scatter(P_norm[:, 0], P_norm[:, 1],
               color=(31/255, 119/255, 180/255), s=100, rasterized=True)

    # nearest point
    if roi != "roi-p":
        for i in range(len(z)):
            ax.scatter(nearest_point[i][0], nearest_point[i][1],
                    color=(255/255, 127/255, 14/255),
                    marker='s', s=200)

    # ROI（正規化半径 r → 元スケールでは楕円, 正規化空間では円）
    # ※「切り出し」は行わない＝可視化のみ

    if roi != "roi-p":
        for i in range(len(z)):
            rx = r / (N[0] - I[0]) * (N[0] - I[0])  # = r
            ry = r / (N[1] - I[1]) * (N[1] - I[1])  # = r
            roi_ellipse = Ellipse(
                xy=(nearest_point[i][0], nearest_point[i][1]),
                width=2*rx, height=2*ry,
                fill=False, edgecolor='black',
                linestyle=(0, (1.9, 1)), linewidth=1.5
            )
            ax.add_patch(roi_ellipse)
    else:
        for i in range(len(z)):
            ax.axvline(ref_norm[i][0],
                linestyle=(0, (1.9, 1)), linewidth=1.5, color='black',zorder=1,  )
            ax.axhline(ref_norm[i][1],
                    linestyle=(0, (1.9, 1)), linewidth=1.5, color='black',zorder=1, )


    # 軸ラベルなど
    ax.set_xlabel(r'$f_1$', fontsize=50)
    ax.set_ylabel(r'$f_2$', fontsize=50)

    # 範囲（正規化）
    ax.set_xlim([0, 1.3])
    ax.set_ylim([0, 1.3])

    # 目盛とラベル
    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.tick_params(axis='both', which='minor', labelsize=45)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.xaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(true_ideal_x, true_nadir_x)))
    ax.yaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(true_ideal_y, true_nadir_y)))

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    ax.tick_params(axis='both', which='both', length=0)

    plt.subplots_adjust(left=0.23, right=0.99, top=0.99, bottom=0.23)
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    # 保存
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(SOL_FILE))[0]
    image_file_pdf = os.path.join(out_dir, f'{base}.pdf')
    image_file_png = os.path.join(out_dir, f'{base}.png')
    plt.savefig(image_file_pdf, dpi=dpi_save)
    plt.savefig(image_file_png, dpi=dpi_save)
    plt.close(fig)
    print(f"2Dプロット画像を作成しました: {image_file_pdf}")
    print(f"2Dプロット画像を作成しました: {image_file_png}")

# ---------- メイン ----------
def main():
    if not os.path.exists(SOL_FILE):
        raise FileNotFoundError(f"sol_file が見つかりません: {SOL_FILE}")

    # PF & z は従来パスから読む
    PF = load_pf(prob, m)
    z = []
    for t in range(1, mult_ref + 1):
        z.append(load_ref_point(prob, m, t))

    print(z)
    # 解集合（任意のファイルをそのまま）
    Pset = np.loadtxt(SOL_FILE, delimiter=',', ndmin=2)
    if Pset.shape[1] < 2:
        raise ValueError("SOL_FILE は少なくとも2列 (f1, f2) を含む必要があります。")

    plot_2d(PF, Pset, z)

if __name__ == "__main__":
    main()
