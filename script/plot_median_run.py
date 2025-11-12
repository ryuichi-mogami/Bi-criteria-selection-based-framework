#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import proj3d
from collections import defaultdict
from matplotlib.patches import Ellipse
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import pandas as pd
# =========================
# ユーザー設定
# =========================
n_obj_list   = [2]
problems     = [
    "DTLZ1","DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"
]
roi_types   = ["roi-p"]
algorithms   = ['BIBEA']#'BNSGA2', 'BIBEA', 'BSMSEMOA','RNSGA2-no'
pop_sel_caption = {
    'POP': 'TRUE'    # 正規化は真の端点
}
pop_selection = list(pop_sel_caption.keys())

n_runs     = 1
t          = 1
r_radius   = 0.1
mu         = 100
out_dir    = '../output/2d_image_median'
dpi_save   = 600
r = 0.1

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'

fevals_list = list(range(100, 50001, 100))
fps        = 15            # フレームレート
bitrate    = 4000          # 出力ビットレート(kbps相当)
fname_fmt = "pop_{nth}th_run_{fevals}fevals.csv"

# ---------- Utility ----------
def make_endpoints_formatter(ideal_val, nadir_val):
    def _fmt(x, pos=None):
        # 0 と 1 のときは実スケールの端点を表示
        if np.isclose(x, 0.0):
            return rf"${0:g}$"
        if np.isclose(x, 1.0):
            return rf"${nadir_val}$"
        # 中間値も実スケールに戻して表示したい場合（不要なら下3行は消してOK）
        # val = ideal_val + x * (nadir_val - ideal_val)
        # return rf"${val:g}$"
    return _fmt


def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom

# ---------- IGD-C+ ----------
# def compute_igd_c_plus(X, PF, PF_norm, z, r, ideal, nadir):
#     # X_norm = normalize_points(X, ideal, nadir)
#     # z_norm = normalize_points(z, ideal, nadir)

#     distance_list = np.zeros(len(PF))
#     for i, p in enumerate(PF):
#         distance_list[i] = np.linalg.norm(p - z)

#     pivot_id = np.argmin(distance_list)
#     pivot_point = PF[pivot_id]
    
#     del_mask = np.full(len(PF), False)
#     for i, p in enumerate(PF):
#         d = np.linalg.norm(p - pivot_point)
#         if d <= r:
#             del_mask[i] = True
#     S_prime = PF[del_mask]

#     if S_prime.shape[0] == 0:
#         return np.nan
#     igd_vals = []
#     for s in S_prime:
#         diff = X - s
#         diff_pos = np.maximum(diff, 0)
#         dists = np.linalg.norm(diff_pos, axis=1)
#         igd_vals.append(dists.min())
#     return float(np.mean(igd_vals))


def compute_igd_c_plus(X, PF, PF_norm, z, r, ideal, nadir, prob, m):
    # X_norm = normalize_points(X, ideal, nadir)
    # z_norm = normalize_points(z, ideal, nadir)

    distance_list = np.zeros(len(PF))
    for i, p in enumerate(PF):
        distance_list[i] = np.linalg.norm(p - z)
    
    pivot_dir = "../output/pivot"
    os.makedirs(pivot_dir, exist_ok=True)
    pivot_file = os.path.join(pivot_dir, f"{prob}_{m}.csv")

    pivot_id = -1
    pivot_point = None
    if os.path.exists(pivot_file):
        pivot_id = int(np.loadtxt(pivot_file))
        pivot_point = PF[pivot_id]
        # print(f"[pivot load] {prob}-m{m}-run{run}")
    else:
        distance_list = np.zeros(len(PF))
        for i, p in enumerate(PF):
            distance_list[i] = np.linalg.norm(p - z)
        pivot_id = np.argmin(distance_list)
        np.savetxt(pivot_file, [pivot_id], fmt='%d')
        pivot_point = PF[pivot_id]
    
    # pivot_id = np.argmin(distance_list)
    # pivot_point = PF[pivot_id]
    
    del_mask = np.full(len(PF), False)
    diff = PF - pivot_point
    val = np.sum((diff/r)**2, axis = 1)
    mask = val <= 1.0
    S_prime = PF[mask]

    if S_prime.shape[0] == 0:
        return np.nan
    igd_vals = []
    for s in S_prime:
        diff = X - s
        diff_pos = np.maximum(diff, 0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))

def asf1(point, ref_point, weight):
    scalar_value = np.max(weight * (point - ref_point))
    return scalar_value   

def compute_igd_a_plus(X, PF, PF_norm, z, r, ideal, nadir, n_obj):
    distance_list = np.zeros(len(PF))
    for i, p in enumerate(PF):
        distance_list[i] = asf1(p, z, np.full(n_obj, 1.0/n_obj))

    pivot_id = np.argmin(distance_list)
    pivot_point = PF[pivot_id]
    
    del_mask = np.full(len(PF), False)
    diff = PF - pivot_point
    val = np.sum((diff/r)**2, axis = 1)
    mask = val <= 1.0
    S_prime = PF[mask]

    if S_prime.shape[0] == 0:
        return np.nan
    igd_vals = []
    for s in S_prime:
        diff = X - s
        diff_pos = np.maximum(diff, 0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))

def compute_igd_p_plus(X, PF, PF_norm, z, r, ideal, nadir):   
    less_eq = np.all(PF <= z, axis=1)   # すべての目的で参照点以下（理想側）
    greater_eq = np.all(PF >= z, axis=1)  # すべての目的で参照点以上（非理想側）
    mask = np.logical_or(less_eq, greater_eq)   # どちらか一方を満たす個体を選択
    S_prime = PF[mask]

    if S_prime.shape[0] == 0:
        return np.nan
    igd_vals = []
    for s in S_prime:
        diff = X - s
        diff_pos = np.maximum(diff, 0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))

# ---------- 入出力（UUA 用に修正） ----------
def sol_path(pop_sel, roi_type, alg, prob, m, run):
    base = f'../output/results/{roi_type}/{alg}/{prob}/m{m}/'
    base_igdC =f'../output/igdC_plus/{roi_type}/{alg}/{prob}/m{m}/'
    os.makedirs(base, exist_ok=True)
    os.makedirs(base_igdC, exist_ok=True)

    csv_path = f'{base}pop_{run}th_run_50000fevals.csv'
    igd_path = f'{base_igdC}pop_{run}th_run_50000fevals.csv'
    return csv_path, igd_path

def load_pf(prob, m, t):
    pf_path = f'../ref_point_dataset/{prob}_d{m}_n50000.csv'
    return np.loadtxt(pf_path, delimiter=',')

def load_ref_point(prob, m, t):
    return np.loadtxt(f'../ref_point_data/roi-c/m{m}_{prob}_type{t}.csv', delimiter=',', ndmin=1)

def select_median_run(vals):
    valid = [(i,v) for i,v in enumerate(vals) if not np.isnan(v)]
    if not valid:
        return None
    sorted_vals = sorted(valid, key=lambda x: x[1])
    n = len(sorted_vals)
    target = sorted_vals[n//2][1] if n % 2 else 0.5*(sorted_vals[n//2-1][1]+sorted_vals[n//2][1])
    best = min(sorted_vals, key=lambda x: (abs(x[1]-target), x[0]))
    return best[0]

def draw_frame(ax, pset_df: pd.DataFrame, fevals: int, PF, z):
    Pset = pset_df.iloc[:, :2].to_numpy(dtype=float)
    # ----- 基本量の計算 -----
    true_ideal_x, true_nadir_x = PF[:, 0].min(), PF[:, 0].max() 
    true_ideal_y, true_nadir_y = PF[:, 1].min(), PF[:, 1].max() 
    true_ideal = PF.min(axis=0) # [ideal_x, ideal_y] 
    true_nadir = PF.max(axis=0) # [nadir_x, nadir_y]
    I, N = (true_ideal, true_nadir)
    PF_norm  = normalize_points(PF,  I, N)
    P_norm   = normalize_points(Pset, I, N)
    ref_norm = normalize_points(z,    I, N)

    # 背景クリア（軸は維持）
    ax.cla()
    #グラフの範囲指定
    ax.set_xlim([0, 1 + 0.3]) 
    ax.set_ylim([0, 1 + 0.3])
    nearest_point = PF_norm[np.argmin(np.linalg.norm(PF - z, axis=1))]

    # PF
    ax.scatter(PF_norm[:, 0], PF_norm[:, 1],
                color='black', s=3, alpha=0.2, rasterized=True)
    # ref point
    ax.scatter(ref_norm[0], ref_norm[1],
                color=(44/255, 160/255, 44/255),
                marker='^', s=230)
    # solution set
    ax.scatter(P_norm[:, 0], P_norm[:, 1],
                color=(31/255, 119/255, 180/255), s=100, rasterized=True)

    # nearest point
    ax.scatter(nearest_point[0], nearest_point[1],
                color=(255/255, 127/255, 14/255),
                marker='s', s=100)
    # ROI（正規化半径 r → 元スケールでは楕円）
    rx = r / (N[0] - I[0])   # x半径
    ry = r / (N[1] - I[1])  # y半径
    roi_ellipse = Ellipse(
        xy=(nearest_point[0], nearest_point[1]),
        width=2*rx, height=2*ry,
        fill=False, edgecolor='black',
        linestyle=(0, (1.9, 1)), linewidth=1.5
    )
    ax.add_patch(roi_ellipse)

    #軸ラベル設定
    ax.set_xlabel(r'$f_1$', fontsize=50)
    ax.set_ylabel(r'$f_2$', fontsize=50)

    #目盛りの数値の大きさ
    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.tick_params(axis='both', which='minor', labelsize=45)
    ax.set_aspect('equal', adjustable='box')
    # 目盛り表示
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # ラベルは実スケールの ideal / nadir
    ax.xaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(true_ideal_x, true_nadir_x)))
    ax.yaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(true_ideal_y, true_nadir_y)))
    ax.set_aspect('equal', adjustable='box')

    plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.25)

    # テロップ（左上）
    # ax.set_title(f"fevals = {fevals}", loc="left", fontsize=28, pad=10)
    ax.text(0.98, 0.98, f"{fevals}fevals", transform=ax.transAxes,
        fontsize=15, va='top', ha='right', color ='black')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    #目盛り線を消す
    ax.tick_params(axis='both', which='both', length=0)

def generate_movie(prob, m, roi_type, alg, pop_sel, run, PF, z):
    file_list = []
    out_mp4    = f"../output/2d_movie_median/{roi_type}/{alg}/{prob}/m{m}/"
    os.makedirs(out_mp4, exist_ok=True)
    out_mp4   += f'pop_median_run.mp4'
    for fe in fevals_list:
        fname = fname_fmt.format(nth=run, fevals=fe)
        path  = os.path.join(f"../output/results/{roi_type}/{alg}/{prob}/m{2}/", fname)
        if os.path.exists(path):
            file_list.append((fe, path))
        else:
            print(f'[warn] missing {path}')
            pass
    
    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    #ffmpeg呼び出し
    writer = FFMpegWriter(fps=fps, bitrate=bitrate, metadata={'artist': 'EC Video'})
    with writer.saving(fig, out_mp4, dpi=150):
        for fe, path in tqdm(file_list, desc="Rendering"):
            try:
                df = pd.read_csv(path, header=None)
            except Exception:
                continue
            draw_frame(ax, df, fe, PF, z)
            writer.grab_frame()
    plt.close(fig)
    print(f"✅ 出力: {out_mp4}")


# ---------- 描画 ----------
def plot_2d(prob, m, roi_type, alg, pop_sel, run, PF, Pset, z, n_obj):
    # ----- 基本量の計算 -----
    true_ideal_x, true_nadir_x = PF[:, 0].min(), PF[:, 0].max() 
    true_ideal_y, true_nadir_y = PF[:, 1].min(), PF[:, 1].max() 
    true_ideal = PF.min(axis=0) # [ideal_x, ideal_y] 
    true_nadir = PF.max(axis=0) # [nadir_x, nadir_y]
    I, N = (true_ideal, true_nadir)
    PF_norm  = normalize_points(PF,  I, N)
    P_norm   = normalize_points(Pset, I, N)
    ref_norm = normalize_points(z,    I, N)

    nearest_point = PF_norm[np.argmin(np.linalg.norm(PF - z, axis=1))]
    if roi_type == 'roi-c':
        nearest_point = PF_norm[np.argmin(np.linalg.norm(PF - z, axis=1))]
    elif roi_type == 'roi-a':
        asf_value = np.zeros(len(PF))
        for i, point in enumerate(PF):
            asf_value[i] = asf1(point, z, np.full(n_obj, 1.0/n_obj))
        pivot_id = np.argmin(asf_value)
        nearest_point = PF_norm[pivot_id]

    # ----- Figure / Axes -----
    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    # PF
    ax.scatter(PF_norm[:, 0], PF_norm[:, 1],
                color='black', s=3, alpha=0.2, rasterized=True)
    # ref point
    ax.scatter(ref_norm[0], ref_norm[1],
                color=(44/255, 160/255, 44/255),
                marker='^', s=230)
    # solution set
    ax.scatter(P_norm[:, 0], P_norm[:, 1],
                color=(31/255, 119/255, 180/255), s=100, rasterized=True)

    # nearest point
    if roi_type == 'roi-c' or roi_type == 'roi-a':
        ax.scatter(nearest_point[0], nearest_point[1],
                    color=(255/255, 127/255, 14/255),
                    marker='s', s=100)
        # ROI（正規化半径 r → 元スケールでは楕円）
        rx = r / (N[0] - I[0]) *true_nadir[0]  # x半径
        ry = r / (N[1] - I[1]) *true_nadir[1]  # y半径
        roi_ellipse = Ellipse(
            xy=(nearest_point[0], nearest_point[1]),
            width=2*rx, height=2*ry,
            fill=False, edgecolor='black',
            linestyle=(0, (1.9, 1)), linewidth=1.5
        )
        ax.add_patch(roi_ellipse)
    # elif roi_type == 'roi-p':
        # ROI-P: 補助線（z の垂直・水平）
        # ax.axvline(ref_norm[0], linestyle=(0, (1.9, 1)), linewidth=1.5, color='black')
        # ax.axhline(ref_norm[1], linestyle=(0, (1.9, 1)), linewidth=1.5, color='black')
    elif roi_type == 'roi-p':
        ax.set_xlim([0, 1 + 0.3]) 
        ax.set_ylim([0, 1 + 0.3])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # ROI-P：z から小さい側（左・下方向）のみ線を引く
        print(ref_norm)
        print(xlim, ylim)
        ax.axvline(
            x=ref_norm[0],
            ymin=0.0,
            ymax=(ref_norm[1] - ylim[0]) / (ylim[1] - ylim[0]),  # zより左側のみ
            linestyle=(0, (1.9, 1)),
            linewidth=1.5,
            color='black'
        )
        ax.axhline(
            y=ref_norm[1],
            xmin=0,
            xmax=(ref_norm[0] - xlim[0]) / (xlim[1] - xlim[0]),
            linestyle=(0, (1.9, 1)),
            linewidth=1.5,
            color='black'
        )

    #軸ラベル設定
    ax.set_xlabel(r'$f_1$', fontsize=50)
    ax.set_ylabel(r'$f_2$', fontsize=50)
    #グラフの範囲指定
    # 範囲（元スケール）
    ax.set_xlim([0, 1 + 0.3]) 
    ax.set_ylim([0, 1 + 0.3])

    #目盛りの数値の大きさ
    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.tick_params(axis='both', which='minor', labelsize=45)
    ax.set_aspect('equal', adjustable='box')
    # 目盛り表示
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # ラベルは実スケールの ideal / nadir
    ax.xaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(true_ideal_x, true_nadir_x)))
    ax.yaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(true_ideal_y, true_nadir_y)))
    ax.set_aspect('equal', adjustable='box')

    plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.25)

    output_dir = f'../output/2d_image_median/{roi_type}/{alg}/{prob}/m{m}/'
    os.makedirs(output_dir, exist_ok=True)
    #画像ファイル名指定
    image_file = os.path.join(output_dir, f'final_pop_median_run.pdf')
    image_file2 = os.path.join(output_dir, f'final_pop_median_run.png')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    #目盛り線を消す
    ax.tick_params(axis='both', which='both', length=0)

    # # 目盛り位置そのものを消す（念のため）
    # ax.set_xticks([])
    # ax.set_yticks([])

    # # 軸ラベルも不要なら
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    #画像保存
    plt.savefig(image_file, dpi=600)
    plt.savefig(image_file2, dpi=600)
    plt.close(fig)
    print(f"2Dプロット画像を作成しました: {image_file}")
    print(f"2Dプロット画像を作成しました: {image_file2}")

def plot_2d_all(prob, m, alg, pop_sel, run, PF, Pset, z):
    # ----- 基本量の計算 -----
    true_ideal_x, true_nadir_x = PF[:, 0].min(), PF[:, 0].max() 
    true_ideal_y, true_nadir_y = PF[:, 1].min(), PF[:, 1].max() 
    true_ideal = PF.min(axis=0) # [ideal_x, ideal_y] 
    true_nadir = PF.max(axis=0) # [nadir_x, nadir_y]
    I, N = (true_ideal, true_nadir)
    PF_norm  = normalize_points(PF,  I, N)
    P_norm   = normalize_points(Pset, I, N)
    ref_norm = normalize_points(z,    I, N)

    nearest_point = PF_norm[np.argmin(np.linalg.norm(PF - z, axis=1))]

    # ----- Figure / Axes -----
    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    # PF
    ax.scatter(PF_norm[:, 0], PF_norm[:, 1],
                color='black', s=3, alpha=0.2, rasterized=True)
    # ref point
    ax.scatter(ref_norm[0], ref_norm[1],
                color=(44/255, 160/255, 44/255),
                marker='^', s=230)
    # solution set
    ax.scatter(P_norm[:, 0], P_norm[:, 1],
                color=(31/255, 119/255, 180/255), s=100, rasterized=True)

    # nearest point
    ax.scatter(nearest_point[0], nearest_point[1],
                color=(255/255, 127/255, 14/255),
                marker='s', s=100)
    # ROI（正規化半径 r → 元スケールでは楕円）
    rx = r / (N[0] - I[0])   # x半径
    ry = r / (N[1] - I[1])   # y半径
    roi_ellipse = Ellipse(
        xy=(nearest_point[0], nearest_point[1]),
        width=2*rx, height=2*ry,
        fill=False, edgecolor='black',
        linestyle=(0, (1.9, 1)), linewidth=1.5
    )
    ax.add_patch(roi_ellipse)


    #軸ラベル設定
    ax.set_xlabel(r'$f_1$', fontsize=50)
    ax.set_ylabel(r'$f_2$', fontsize=50)
    #グラフの範囲指定
    # 範囲（元スケール）
    ax.set_xlim([0, 1 + 0.3]) 
    ax.set_ylim([0, 1 + 0.3])

    #目盛りの数値の大きさ
    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.tick_params(axis='both', which='minor', labelsize=45)
    ax.set_aspect('equal', adjustable='box')
    # 目盛り表示
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # ラベルは実スケールの ideal / nadir
    ax.xaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(true_ideal_x, true_nadir_x)))
    ax.yaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(true_ideal_y, true_nadir_y)))
    ax.set_aspect('equal', adjustable='box')

    plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.25)

    output_dir = f'./output/2d_image/{alg}/{prob}/m{m}/'
    os.makedirs(output_dir, exist_ok=True)
    #画像ファイル名指定
    image_file = os.path.join(output_dir, f'final_pop_{run}th_run.pdf')
    image_file2 = os.path.join(output_dir, f'final_pop_{run}th_run.png')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    #目盛り線を消す
    ax.tick_params(axis='both', which='both', length=0)

    # # 目盛り位置そのものを消す（念のため）
    # ax.set_xticks([])
    # ax.set_yticks([])

    # # 軸ラベルも不要なら
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    #画像保存
    plt.savefig(image_file, dpi=600)
    plt.savefig(image_file2, dpi=600)
    plt.close(fig)
    print(f"2Dプロット画像を作成しました: {image_file}")
    print(f"2Dプロット画像を作成しました: {image_file2}")

# 追加（基準にするpop_sel）
REF_POP = 'POP'
n = [50000, 50086, 102340, 455126, 3162510]
# ---------- メイン ----------
def main():
    os.makedirs(out_dir, exist_ok=True)
    for m in n_obj_list:
        for prob in problems:
            pf_path = f'../ref_point_dataset/{prob}_d{m}_n{n[m - 2]}.csv'
            pf_npy = pf_path.replace('.csv', '.npy')
            if not os.path.exists(pf_npy):
                PF = load_pf(prob, m, t)
                np.save(pf_npy, PF)
            else:
                PF = np.load(pf_npy)
            z  = load_ref_point(prob, m, t)
            ideal, nadir = PF.min(axis=0), PF.max(axis=0)
            PF_norm = normalize_points(PF, ideal, nadir)
            for roi_type in roi_types:
                data = []
                for i in range(m):
                    data.append(nadir[i] *  r_radius)
                r_radius_elipse = np.array(data)
                for alg in algorithms:
                    if alg == "RNSGA2" and roi_type =="roi-p":
                        continue
                    if alg == "gNSGA2" and roi_type =="roi-c":
                        continue
                    # 1) まず基準POPで中央値runを決める
                    ref_vals = []
                    for run in range(n_runs):
                        ref_csv, ref_igd = sol_path(REF_POP, roi_type, alg, prob, m, run)
                        if not os.path.exists(ref_csv):
                            continue
                        if os.path.exists(ref_igd):
                            igd = float(np.loadtxt(ref_igd))
                        else:
                            X = np.loadtxt(ref_csv, delimiter=',', ndmin=2)
                            if roi_type == 'roi-c':
                                igd = compute_igd_c_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir, prob, m)
                                np.savetxt(ref_igd, [igd])
                            elif roi_type == 'roi-a':
                                igd = compute_igd_a_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir, m)
                                np.savetxt(ref_igd, [igd])
                            elif roi_type == 'roi-p':
                                igd = compute_igd_p_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir)
                                np.savetxt(ref_igd, [igd])
                            elif roi_type == 'emo':
                                igd = compute_igd_c_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir, prob, m)
                                np.savetxt(ref_igd, [igd])
                        ref_vals.append((run, igd))

                    if not ref_vals:
                        continue

                    ref_median_run = select_median_run([v for _, v in ref_vals])
                    if ref_median_run is None:
                        continue

                    # 2) 決めたrun番号を全pop_selに使って描画
                    for pop_sel in pop_selection:
                        sol_csv, igd_file = sol_path(pop_sel, roi_type, alg, prob, m, ref_median_run)
                        if not os.path.exists(sol_csv):
                            # 必要なら警告だけ出してスキップ
                            # print(f'[warn] missing {sol_csv}')
                            continue
                        X = np.loadtxt(sol_csv, delimiter=',', ndmin=2)
                        # （IGD-C+は描画に必須ではないが、必要ならキャッシュしておく）
                        # if not os.path.exists(igd_file):
                        #     X = np.loadtxt(sol_csv, delimiter=',', ndmin=2)
                        #     igd = compute_igd_c_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir)
                        #     np.savetxt(igd_file, [igd])
                        Pset = np.loadtxt(sol_csv, delimiter=',', ndmin=2)
                        plot_2d(prob, m, roi_type, alg, pop_sel, ref_median_run, PF, Pset, z, m)
                        # for i in range(31):
                        #     sol_csv, igd_file = sol_path(pop_sel, alg, prob, m, i)
                        #     if not os.path.exists(sol_csv):
                        #         # 必要なら警告だけ出してスキップ
                        #         # print(f'[warn] missing {sol_csv}')
                        #         continue

                        #     Pset = np.loadtxt(sol_csv, delimiter=',', ndmin=2)
                        #     plot_2d_all(prob, m, alg, pop_sel, i, PF, Pset, z)
                        # generate_movie(prob, m, alg, pop_sel, ref_median_run, PF, z)



if __name__ == "__main__":
    main()
