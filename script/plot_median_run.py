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
    "DTLZ7",
]       
mult_ref  = 1
roi_types   = ["roi-c"]
algorithms   = ['BNSGA2', 'BIBEA', 'BSMSEMOA','BSPEA2','BNSGA3','BNSGA2-drs','BSPEA2-drs','RNSGA2','gNSGA2']#'BNSGA2', 'BIBEA', 'BSMSEMOA','RNSGA2-no'
pop_sel_caption = {
    'POP': 'TRUE'    # 正規化は真の端点
}
pop_selection = list(pop_sel_caption.keys())

n_runs     = 31
t          = 1
r_radius   = 0.1
mu         = 100
out_dir    = f'../output/2d_image_median_{mult_ref}'
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
    
    pivot_dir = f"../output/pivot_{mult_ref}"
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
    if alg == "NSGA2":
        base = f'../output/results_{mult_ref}/emo/{alg}/{prob}/m{m}/'
        base_igdC =f'../output/igdC_plus_{mult_ref}/emo-{roi_type}/{alg}/{prob}/m{m}/'
    else:
        base = f'../output/results_{mult_ref}/{roi_type}/{alg}/{prob}/m{m}/'
        base_igdC =f'../output/igdC_plus_{mult_ref}/{roi_type}/{alg}/{prob}/m{m}/'
    os.makedirs(base, exist_ok=True)
    os.makedirs(base_igdC, exist_ok=True)

    if alg == "BNSGA2" and mult_ref == 2:
        csv_path = f'{base}pop_{run}th_run_25000fevals.csv'
        igd_path = f'{base_igdC}pop_{run}th_run_25000fevals.csv'
    else:
        csv_path = f'{base}pop_{run}th_run_50000fevals.csv'
        igd_path = f'{base_igdC}pop_{run}th_run_50000fevals.csv'
    return csv_path, igd_path

def load_pf(prob, m, t):
    pf_path = f'../ref_point_dataset/{prob}_d{m}_n50000.csv'
    return np.loadtxt(pf_path, delimiter=',')

def load_ref_point(prob, m, t):
    if mult_ref == 1:
        return np.loadtxt(f'../ref_point_data/roi-c/m{m}_{prob}_type{t}.csv', delimiter=',', ndmin=1)
    else:   
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

# ---------- 描画 ----------
def plot_2d(prob, m, roi_type, alg, pop_sel, run, PF, Pset, z, n_obj):
    # ----- 基本量の計算 -----
    true_ideal_x, true_nadir_x = PF[:, 0].min(), PF[:, 0].max() 
    true_ideal_y, true_nadir_y = PF[:, 1].min(), PF[:, 1].max() 
    true_ideal = PF.min(axis=0) # [ideal_x, ideal_y] 
    true_nadir = PF.max(axis=0) # [nadir_x, nadir_y]
    print(true_ideal, true_nadir)
    I, N = (true_ideal, true_nadir)
    PF_norm  = normalize_points(PF,  I, N)
    print(PF_norm.max(axis=0), PF_norm.min(axis=0))
    P_norm   = normalize_points(Pset, I, N)
    ref_norm = normalize_points(z,    I, N)
    nearest_point = []
    if roi_type == 'roi-c':
        for i in range(len(z)):
            nearest_point.append(PF_norm[np.argmin(np.linalg.norm(PF - z[i], axis=1))])
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
    if roi_type != "emo" and alg != "NSGA2": 
        # ref point
        for i in range(len(z)):
            ax.scatter(ref_norm[i][0], ref_norm[i][1],
                    color=(44/255, 160/255, 44/255),
                    marker='^', s=200,zorder=10)
    # solution set
    ax.scatter(P_norm[:, 0], P_norm[:, 1],
                color=(31/255, 119/255, 180/255), s=200, rasterized=True)

    # nearest point
    if (roi_type == 'roi-c' or roi_type == 'roi-a') and alg != "NSGA2":
        # for i in range(len(z)):
        #     ax.scatter(nearest_point[i][0], nearest_point[i][1],
        #             color=(255/255, 127/255, 14/255),
        #             marker='s', s=300)
        # ROI（正規化半径 r → 元スケールでは楕円）
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
    # elif roi_type == 'roi-p':
        # ROI-P: 補助線（z の垂直・水平）
        # ax.axvline(ref_norm[0], linestyle=(0, (1.9, 1)), linewidth=1.5, color='black')
        # ax.axhline(ref_norm[1], linestyle=(0, (1.9, 1)), linewidth=1.5, color='black')
    elif roi_type == 'roi-p' and alg != "NSGA2":
        ax.set_xlim([0, 1 + 0.3]) 
        ax.set_ylim([0, 1 + 0.3])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        def is_feasible_wrt_pf_dominance(z, PF, eps=0.0):
            """
            feasible: PF上のどれかの点が z を（弱く）支配する
            ∃p∈PF: p_i <= z_i (∀i)
            """
            z = np.asarray(z, dtype=float).ravel()
            PF = np.asarray(PF, dtype=float)
            return bool(np.any(np.all(PF <= z + eps, axis=1)))
        # ROI-P：z から小さい側（左・下方向）のみ線を引く
        print(ref_norm)
        print(xlim, ylim)
        feasible = is_feasible_wrt_pf_dominance(ref_norm, PF_norm)
        ref_norm = ref_norm.ravel() 

        #ref_norm is fesible
        print("feasible:", feasible)
        if feasible:
            ax.axvline(x = ref_norm[0], ymin = 0, ymax = (ref_norm[1] - ylim[0]) / (ylim[1] - ylim[0]),
                linestyle=(0, (1.9, 1)), linewidth=1.5, color='black',zorder=1, )
            ax.axhline(y = ref_norm[1], xmin = 0, xmax = (ref_norm[0] - xlim[0]) / (xlim[1] - xlim[0]),
                linestyle=(0, (1.9, 1)), linewidth=1.5, color='black',zorder=1, )
        else:
            ax.axvline(x = ref_norm[0], ymin = (ref_norm[1] - ylim[0]) / (ylim[1] - ylim[0]), ymax = 1,
                linestyle=(0, (1.9, 1)), linewidth=1.5, color='black',zorder=1, )
            ax.axhline(y = ref_norm[1], xmin = (ref_norm[0] - xlim[0]) / (xlim[1] - xlim[0]), xmax = 1,
                linestyle=(0, (1.9, 1)), linewidth=1.5, color='black',zorder=1, )
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
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ラベルは実スケールの ideal / nadir
    ax.xaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(0, 1)))
    ax.yaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(0, 1)))
    ax.set_aspect('equal', adjustable='box')

    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.subplots_adjust(left=0.23, right=0.99, top=0.99, bottom=0.23)
    output_dir = f'../output/2d_image_median_{mult_ref}/{roi_type}/{alg}/{prob}/m{m}/'
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
            z = []
            for t in range(1, mult_ref + 1):
                 z.append(load_ref_point(prob, m, t))
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
                    if roi_type == "emo":
                        if alg in ["BNSGA2", "BIBEA", "BSMSEMOA", "RNSGA2", "gNSGA2"]:
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
                                igd = compute_igd_c_plus(X, PF, PF_norm, z[0], r_radius_elipse, ideal, nadir, prob, m)
                                np.savetxt(ref_igd, [igd])
                            elif roi_type == 'roi-a':
                                igd = compute_igd_a_plus(X, PF, PF_norm, z[0], r_radius_elipse, ideal, nadir, m)
                                np.savetxt(ref_igd, [igd])
                            elif roi_type == 'roi-p':
                                igd = compute_igd_p_plus(X, PF, PF_norm, z[0], r_radius_elipse, ideal, nadir)
                                np.savetxt(ref_igd, [igd])
                            elif roi_type == 'emo':
                                igd = compute_igd_c_plus(X, PF, PF_norm, z[0], r_radius_elipse, ideal, nadir, prob, m)
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
