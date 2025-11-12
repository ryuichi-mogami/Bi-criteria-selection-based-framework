#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Ellipse
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import pandas as pd

# =========================
# ユーザー設定
# =========================
n_obj_list   = [2]
problems     = ["DTLZ3"]
roi_types    = ["roi-p"]
algorithms   = ['BNSGA2']  # 必要に応じて変更
pop_sel_caption = {
    'POP': 'TRUE'    # 正規化は真の端点
}
pop_selection = list(pop_sel_caption.keys())

n_runs     = 1
t          = 1
r_radius   = 0.1
mu         = 100
out_dir    = '../output/2d_movie_median/'
dpi_save   = 600
r          = 0.1

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'

fevals_list = list(range(100, 50001, 100))
fps        = 15            # フレームレート
bitrate    = 4000          # 出力ビットレート(kbps相当)
fname_fmt  = "pop_{nth}th_run_{fevals}fevals.csv"

# 追加（基準にするpop_sel）
REF_POP = 'POP'
# PF 用サンプル数（m=2,3,4,... 用）
n = [50000, 50086, 102340, 455126, 3162510]


# ---------- Utility ----------
def make_endpoints_formatter(ideal_val, nadir_val):
    def _fmt(x, pos=None):
        # 0 と 1 のときは実スケールの端点を表示
        if np.isclose(x, 0.0):
            return rf"${0:g}$"
        if np.isclose(x, 1.0):
            return rf"${nadir_val}$"
        # 中間値を実スケールに戻して表示したい場合はここを有効化
        # val = ideal_val + x * (nadir_val - ideal_val)
        # return rf"${val:g}$"
        return ""
    return _fmt


def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom


def sol_path(pop_sel, roi_type, alg, prob, m, run):
    base = f'../output/results/{roi_type}/{alg}/{prob}/m{m}/'
    base_igdC = f'../output/igdC_plus/{roi_type}/{alg}/{prob}/m{m}/'
    os.makedirs(base, exist_ok=True)
    os.makedirs(base_igdC, exist_ok=True)

    csv_path = f'{base}pop_{run}th_run_50000fevals.csv'
    igd_path = f'{base_igdC}pop_{run}th_run_50000fevals.csv'
    return csv_path, igd_path

def sol_path2(pop_sel, roi_type, alg, prob, m, run):
    base = f'../output/results/emo/{alg}/{prob}/m{m}/'
    base_igdC = f'../output/igdC_plus/emo-{roi_type}/{alg}/{prob}/m{m}/'
    os.makedirs(base, exist_ok=True)
    os.makedirs(base_igdC, exist_ok=True)

    csv_path = f'{base}pop_{run}th_run_50000fevals.csv'
    igd_path = f'{base_igdC}pop_{run}th_run_50000fevals.csv'
    return csv_path, igd_path


def load_pf(prob, m, t):
    pf_path = f'../ref_point_dataset/{prob}_d{m}_n{n[m - 2]}.csv'
    return np.loadtxt(pf_path, delimiter=',')


def load_ref_point(prob, m, t):
    return np.loadtxt(f'../ref_point_data/roi-c/m{m}_{prob}_type{t}.csv',
                      delimiter=',', ndmin=1)


def select_median_run(vals):
    """vals: runごとの IGD 値のリスト"""
    valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
    if not valid:
        return None
    sorted_vals = sorted(valid, key=lambda x: x[1])
    n_ = len(sorted_vals)
    if n_ % 2:
        target = sorted_vals[n_ // 2][1]
    else:
        target = 0.5 * (sorted_vals[n_ // 2 - 1][1] + sorted_vals[n_ // 2][1])
    best = min(sorted_vals, key=lambda x: (abs(x[1] - target), x[0]))
    return best[0]


# ---------- 1フレーム描画 ----------
def draw_frame(ax, pset_df: pd.DataFrame, fevals: int, PF, z, roi_type):
    Pset = pset_df.iloc[:, :2].to_numpy(dtype=float)

    # ----- 基本量の計算 -----
    true_ideal_x, true_nadir_x = PF[:, 0].min(), PF[:, 0].max()
    true_ideal_y, true_nadir_y = PF[:, 1].min(), PF[:, 1].max()
    true_ideal = PF.min(axis=0)   # [ideal_x, ideal_y]
    true_nadir = PF.max(axis=0)   # [nadir_x, nadir_y]
    I, N = (true_ideal, true_nadir)
    PF_norm  = normalize_points(PF,  I, N)
    P_norm   = normalize_points(Pset, I, N)
    ref_norm = normalize_points(z,    I, N)

    # 背景クリア（軸は維持）
    ax.cla()
    # グラフの範囲指定
    ax.set_xlim([0, 1 + 50])
    ax.set_ylim([0, 1 + 50])

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
    if roi_type == "roi-c":
        rx = r / (N[0] - I[0]) * true_nadir[0]   # x半径
        ry = r / (N[1] - I[1]) * true_nadir[1]  # y半径
        roi_ellipse = Ellipse(
            xy=(nearest_point[0], nearest_point[1]),
            width=2*rx, height=2*ry,
            fill=False, edgecolor='black',
            linestyle=(0, (1.9, 1)), linewidth=1.5
        )
        ax.add_patch(roi_ellipse)
    elif roi_type == "roi-p":
        ax.axhline(y=ref_norm[1], color='black',
                   linestyle=(0, (1.9, 1)), linewidth=1.5)
        ax.axvline(x=ref_norm[0], color='black',
                   linestyle=(0, (1.9, 1)), linewidth=1.5)

    # 軸ラベル設定
    ax.set_xlabel(r'$f_1$', fontsize=50)
    ax.set_ylabel(r'$f_2$', fontsize=50)

    # 目盛りの数値の大きさ
    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.tick_params(axis='both', which='minor', labelsize=45)
    ax.set_aspect('equal', adjustable='box')

    # 目盛り表示
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # ラベルは実スケールの ideal / nadir
    ax.xaxis.set_major_formatter(
        FuncFormatter(make_endpoints_formatter(true_ideal_x, true_nadir_x)))
    ax.yaxis.set_major_formatter(
        FuncFormatter(make_endpoints_formatter(true_ideal_y, true_nadir_y)))

    plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.25)

    # fevals テキスト（右上）
    ax.text(0.98, 0.98, f"{fevals}fevals", transform=ax.transAxes,
            fontsize=15, va='top', ha='right', color='black')

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    # 目盛り線を消す
    ax.tick_params(axis='both', which='both', length=0)


# ---------- 動画生成 ----------
def generate_movie(prob, m, roi_type, alg, pop_sel, run, PF, z):
    file_list = []
    out_mp4 = f"../output/2d_movie_median/{roi_type}/{alg}/{prob}/m{m}/"
    os.makedirs(out_mp4, exist_ok=True)
    out_mp4 += f'pop_median_run.mp4'

    # 各 fevals の CSV を収集
    for fe in fevals_list:
        fname = fname_fmt.format(nth=run, fevals=fe)
        if alg == "NSGA2":
            path  = os.path.join(f"../output/results/emo/{alg}/{prob}/m{m}/", fname)
        else:
            path  = os.path.join(f"../output/results/{roi_type}/{alg}/{prob}/m{m}/", fname)
        if os.path.exists(path):
            file_list.append((fe, path))
        else:
            print(f'[warn] missing {path}')
            pass

    if not file_list:
        print(f"[warn] no frames found for {roi_type}, {alg}, {prob}, m={m}, run={run}")
        return

    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    # ffmpeg呼び出し
    writer = FFMpegWriter(fps=fps, bitrate=bitrate,
                          metadata={'artist': 'EC Video'})
    with writer.saving(fig, out_mp4, dpi=150):
        for fe, path in tqdm(file_list, desc=f"Rendering {alg}-{prob}-m{m}-run{run}"):
            try:
                df = pd.read_csv(path, header=None)
            except Exception:
                continue
            draw_frame(ax, df, fe, PF, z, roi_type)
            writer.grab_frame()
    plt.close(fig)
    print(f"✅ 出力: {out_mp4}")


# ---------- メイン ----------
def main():
    os.makedirs(out_dir, exist_ok=True)

    for m in n_obj_list:
        for prob in problems:
            # PF の読み込み（なければ CSV から npy を作る）
            pf_path = f'../ref_point_dataset/{prob}_d{m}_n{n[m - 2]}.csv'
            pf_npy = pf_path.replace('.csv', '.npy')
            if not os.path.exists(pf_npy):
                PF = load_pf(prob, m, t)
                np.save(pf_npy, PF)
            else:
                PF = np.load(pf_npy)

            z = load_ref_point(prob, m, t)

            for roi_type in roi_types:
                for alg in algorithms:
                    # RNSGA2/roi-p や gNSGA2/roi-c を弾く処理が必要ならここに追加

                    # 1) まず基準POPで中央値 run を決める（既存 IGD ファイルを使用）
                    ref_vals = []
                    for run in range(n_runs):
                        if alg == "NSGA2":
                            _, ref_igd = sol_path2(REF_POP, roi_type, alg, prob, m, run)
                        else:
                            _, ref_igd = sol_path(REF_POP, roi_type, alg, prob, m, run)
                        if not os.path.exists(ref_igd):
                            continue
                        try:
                            igd = float(np.loadtxt(ref_igd))
                        except Exception:
                            continue
                        ref_vals.append((run, igd))

                    if not ref_vals:
                        print(f"[warn] no IGD files for {roi_type}, {alg}, {prob}, m={m}")
                        continue

                    # run -> igd のリストに変換
                    igd_list = [np.nan] * n_runs
                    for run_id, igd in ref_vals:
                        igd_list[run_id] = igd

                    ref_median_run = select_median_run(igd_list)
                    print(f"Median run for {roi_type}, {alg}, {prob}, m={m} is {ref_median_run}")
                    if ref_median_run is None:
                        print(f"[warn] median run not found for {roi_type}, {alg}, {prob}, m={m}")
                        continue

                    # 2) 決めた run 番号を全 pop_sel に使って動画生成
                    for pop_sel in pop_selection:
                        if alg == "NSGA2":
                            sol_csv, _ = sol_path2(pop_sel, roi_type, alg, prob, m, ref_median_run)
                        else:
                            sol_csv, _ = sol_path(pop_sel, roi_type, alg, prob, m, ref_median_run)
                        if not os.path.exists(sol_csv):
                            print(f'[warn] missing final CSV: {sol_csv}')
                            continue
                        generate_movie(prob, m, roi_type, alg, pop_sel,
                                       ref_median_run, PF, z)


if __name__ == "__main__":
    main()
