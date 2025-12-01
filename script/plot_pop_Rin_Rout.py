#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.abspath("../pymoo"))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Ellipse
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

# =========================
# ユーザー設定
# =========================
roi = "roi-p"  # "roi-c" または "roi-p"
SOL_FILE  = f"../output/results_1/{roi}/BNSGA2/DTLZ2/m2/pop_0th_run_300fevals.csv"
m          = 2
prob       = "DTLZ2"
t          = 1
r          = 0.2   # ROI-C の半径

out_dir    = '../output/2d_image_RinRout'
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
        return rf"${x:g}$"
    return _fmt

def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom

# ---------- 入出力 ----------
def load_pf(prob, m, t):
    pf_path = f'../ref_point_dataset/{prob}_d{m}_n50000.csv'
    return np.loadtxt(pf_path, delimiter=',')

def load_ref_point(prob, m, t, roi):
    return np.loadtxt(
        f'../ref_point_data/{roi}/m{m}_{prob}_type{t}.csv',
        delimiter=',', ndmin=1
    )

# ---------- 描画 ----------
def plot_2d(PF, Pset, z, roi):
    # ----- 基本量 -----
    true_ideal_x, true_nadir_x = PF[:, 0].min(), PF[:, 0].max()
    true_ideal_y, true_nadir_y = PF[:, 1].min(), PF[:, 1].max()
    true_ideal = PF.min(axis=0)
    true_nadir = PF.max(axis=0)
    I, N = (true_ideal, true_nadir)

    PF_norm  = normalize_points(PF,  I, N)
    P_norm   = normalize_points(Pset, I, N)
    ref_norm = normalize_points(z,    I, N)

    nondom_idx = find_non_dominated(Pset)
    nondom_F   = Pset[nondom_idx]
    nondom_F_norm = normalize_points(nondom_F, I, N)

    # ref_point (z) に最も近い非劣解を pivot とする
    nearest_idx = np.argmin(np.linalg.norm(nondom_F - z, axis=1))
    nearest_point = nondom_F_norm[nearest_idx]
    pivot_point   = nondom_F[nearest_idx]         # 元スケール（ROI-C 判定用）

    # ----- R^in / R^out の判定（Pset に対して） -----
    if roi == "roi-c":
        diff   = Pset - pivot_point
        val    = np.sum((diff / r)**2, axis=1)
        mask_in = val <= 1.0
    elif roi == "roi-p":
        less_eq    = np.all(Pset <= z, axis=1)
        greater_eq = np.all(Pset >= z, axis=1)
        mask_in    = np.logical_or(less_eq, greater_eq)
    else:
        raise ValueError("roi は 'roi-c' か 'roi-p' を指定してください。")

    P_in_norm  = P_norm[mask_in]
    P_out_norm = P_norm[~mask_in]
    # ----- R^out を R^in と同じ個数に間引き -----
    n_in  = P_in_norm.shape[0]
    n_out = P_out_norm.shape[0]
    if n_in > 10:
        idx_in = np.linspace(0, n_in - 1, num=10, dtype=int)
        P_in_plot = P_in_norm[idx_in]
    else:
        P_in_plot = P_in_norm
    if n_out > 10:
        idx_out = np.linspace(0, n_out - 1, num=10, dtype=int)
        P_out_plot = P_out_norm[idx_out]
    else:
        P_out_plot = P_out_norm
    # if n_in > 0 and n_out > n_in:
    #     # 等間隔にインデックスを取る（再現性のある間引き）
    #     idx = np.linspace(0, n_out - 1, num=n_in, dtype=int)
    #     P_out_plot = P_out_norm[idx]
    # else:
    #     # もともと少ない場合や R^in = 0 の場合はそのまま
    #     P_out_plot = P_out_norm

    # ----- Figure / Axes -----
    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    # PF
    ax.scatter(PF_norm[:, 0], PF_norm[:, 1],
               color='black', s=3, alpha=0.2, rasterized=True)

    # solution set: R^out（バツ印）, R^in（青丸）
    # if P_out_plot.size > 0:
    #     ax.scatter(P_out_plot[:, 0], P_out_plot[:, 1],
    #                color=(31/255, 119/255, 180/255),
    #                marker='x', s=100, linewidths=2, rasterized=True)
    if P_out_plot.size > 0:
        ax.scatter(P_out_plot[:, 0], P_out_plot[:, 1],
                   color=(255/255, 127/255, 14/255),
                   marker='o', s=100, linewidths=2, rasterized=True)
    if P_in_plot.size > 0:
        ax.scatter(P_in_plot[:, 0], P_in_plot[:, 1],
                   color=(31/255, 119/255, 180/255),
                   marker='o', s=100, rasterized=True)

    # pivot（ROI-C のときだけ表示）
    # if roi == "roi-c":
    #     ax.scatter(nearest_point[0], nearest_point[1],
    #                color=(255/255, 127/255, 14/255),
    #                marker='s', s=100)

    # ref point
    ax.scatter(ref_norm[0], ref_norm[1],
               color=(44/255, 160/255, 44/255),
               marker='^', s=230)

    # ROI 可視化
    if roi == "roi-c":
        rx = r
        ry = r
        roi_ellipse = Ellipse(
            xy=(nearest_point[0], nearest_point[1]),
            width=2*rx, height=2*ry,
            fill=False, edgecolor='black',
            linestyle=(0, (1.9, 1)), linewidth=1.5
        )
        ax.add_patch(roi_ellipse)
    elif roi == "roi-p":
        ax.axvline(ref_norm[0],
                   linestyle=(0, (1.9, 1)), linewidth=1.5, color='black')
        ax.axhline(ref_norm[1],
                   linestyle=(0, (1.9, 1)), linewidth=1.5, color='black')

    # 軸ラベルなど（体裁は元コードのまま）
    # ax.set_xlabel(r'$f_1$', fontsize=50)
    # ax.set_ylabel(r'$f_2$', fontsize=50)

    ax.set_xlim([0, 1 + 1])
    ax.set_ylim([0, 1 + 1])

    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.tick_params(axis='both', which='minor', labelsize=45)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.xaxis.set_major_formatter(
        FuncFormatter(make_endpoints_formatter(true_ideal_x, true_nadir_x)))
    ax.yaxis.set_major_formatter(
        FuncFormatter(make_endpoints_formatter(true_ideal_y, true_nadir_y)))

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    ax.tick_params(axis='both', which='both', length=0)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(SOL_FILE))[0]
    image_file_pdf = os.path.join(out_dir, f'{base}_{roi}_RinRout.pdf')
    image_file_png = os.path.join(out_dir, f'{base}_{roi}_RinRout.png')
    plt.savefig(image_file_pdf, dpi=dpi_save)
    plt.savefig(image_file_png, dpi=dpi_save)
    plt.close(fig)
    print(f"{roi} の 2Dプロット画像を作成しました: {image_file_pdf}")
    print(f"{roi} の 2Dプロット画像を作成しました: {image_file_png}")

# ---------- メイン ----------
def main():
    if not os.path.exists(SOL_FILE):
        raise FileNotFoundError(f"sol_file が見つかりません: {SOL_FILE}")

    PF = load_pf(prob, m, t)
    z  = load_ref_point(prob, m, t, roi)

    Pset = np.loadtxt(SOL_FILE, delimiter=',', ndmin=2)
    if Pset.shape[1] < 2:
        raise ValueError("SOL_FILE は少なくとも2列 (f1, f2) を含む必要があります。")

    plot_2d(PF, Pset, z, roi)

if __name__ == "__main__":
    main()
