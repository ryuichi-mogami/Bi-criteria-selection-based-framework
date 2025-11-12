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
from matplotlib.lines import Line2D
# =========================
# ユーザー設定
# =========================
n_obj_list   = [2,4,6]
problems     = [
    "DTLZ2"
]
algorithms   = ['BNSGA2', 'BIBEA', 'BSMSEMOA']#'BNSGA2', 'BIBEA', 'BSMSEMOA','RNSGA2-no'
pop_sel_caption = {
    'POP': 'TRUE'    # 正規化は真の端点
}
pop_selection = list(pop_sel_caption.keys())

n_runs     = 31
t          = 1
r_radius   = 0.1
mu         = 100
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
    pivot_dir = "../output/pivot"
    os.makedirs(pivot_dir, exist_ok=True)
    pivot_file = os.path.join(pivot_dir, f"{prob}_{m}.csv")

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
    # print(f"[pivot calc] {prob}-m{m}-run{run}: {t1 - t0:.2f}s")

    del_mask = np.full(len(PF), False)
    diff = PF - pivot_point
    val = np.sum((diff/r)**2, axis = 1)
    mask = val <= 1.0
    S_prime = PF[mask]
    # print(f"[mask calc] {prob}-m{m}-run{run}: {t1 - t0:.2f}s")
    print(len(S_prime))
    print(str(m),prob)
    if S_prime.shape[0] == 0:
        return np.nan
    igd_vals = []
    for s in S_prime:
        diff = X - s
        diff_pos = np.maximum(diff, 0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    # print(f"[IGDCplus calc] {prob}-m{m}-run{run}: {t1 - t0:.2f}s")
    return float(np.mean(igd_vals))

def compute_igd_p_plus(X, PF, PF_norm, z, r, ideal, nadir, prob, m):   
    less_eq = np.all(PF <= z, axis=1)   # すべての目的で参照点以下（理想側）
    greater_eq = np.all(PF >= z, axis=1)  # すべての目的で参照点以上（非理想側）
    mask = np.logical_or(less_eq, greater_eq)   # どちらか一方を満たす個体を選択
    print(format(m), prob)
    print("less_true   =", np.sum(less_eq))
    print("greater_true=", np.sum(greater_eq))
    print("union S'    =", np.sum(mask)) 
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
def sol_path(pop_sel, alg, prob, m, run, roi_type):
    if roi_type == "roi-c":
        base = f'../output/results/{roi_type}/{alg}/{prob}/m{m}/'
        base_igdC =f'../output/igdC_plus/{roi_type}/{alg}/{prob}/m{m}/'
    else:
        base = f'../output/results/{roi_type}/{alg}/{prob}/m{m}/'
        base_igdC =f'../output/igdC_plus/{roi_type}/{alg}/{prob}/m{m}/'
    os.makedirs(base, exist_ok=True)
    os.makedirs(base_igdC, exist_ok=True)

    csv_path = f'{base}pop_{run}th_run_50000fevals.csv'
    igd_path = f'{base_igdC}pop_{run}th_run_50000fevals.csv'
    return csv_path, igd_path

def load_pf(prob, m, t, pf_path):
    return np.loadtxt(pf_path, delimiter=',')

def load_ref_point(prob, m, t,roi_type):
    return np.loadtxt(f'../ref_point_data/{roi_type}/m{m}_{prob}_type{t}.csv', delimiter=',', ndmin=1)

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
def plot_function_call(prob, m, alg, function_call, roi_type):
    # ----- Figure / Axes -----
    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    x = np.arange(len(function_call))
    # solution set
    ax.plot(x, function_call,
                color=(31/255, 119/255, 180/255), linewidth=2)

    #軸ラベル設定
    ax.set_xlabel(r'evaluation count', fontsize=50)
    ax.set_ylabel(r'function call', fontsize=50)

    ax.set_xlim([0, 500]) 
    ax.set_ylim([0, 500])
    plt.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.15)

    output_dir = f'../output/function_call_median/{roi_type}/{alg}/{prob}/m{m}/'
    os.makedirs(output_dir, exist_ok=True)
    #画像ファイル名指定
    image_file = os.path.join(output_dir, f'function_call_progress.pdf')
    image_file2 = os.path.join(output_dir, f'function_call_progress.png')
    #画像保存
    plt.savefig(image_file, dpi=600)
    plt.savefig(image_file2, dpi=600)
    plt.close(fig)
    print(f"2Dプロット画像を作成しました: {image_file}")
    print(f"2Dプロット画像を作成しました: {image_file2}")


def save_top_legend_image(labels, styles, out_path_base, ncol=None):
    """
    凡例の外側の余白を完全に切り落とし、PNGとPDFの両方を出力
    out_path_base: 拡張子を除いた出力パス（例: './legend_m_list'）
    """
    fig = plt.figure(figsize=(4.5, 0.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4"])

    handles = []
    for i, (lab, st) in enumerate(zip(labels, styles)):
        color = colors[i % len(colors)]
        if isinstance(st, dict):
            st_use = dict(st)
            st_use.setdefault("color", color)
            st_use.setdefault("linewidth", 2)
            handle = Line2D([0], [0], label=lab, **st_use)
        else:
            handle = Line2D([0], [0], color=color, linestyle=st, linewidth=2, label=lab)
        handles.append(handle)

    legend = fig.legend(
        handles=handles,
        loc="center",
        ncol=(ncol or len(labels)),
        frameon=True, framealpha=1.0, fancybox=False,
        handlelength=2.6, columnspacing=0.9, handletextpad=0.5,
        borderaxespad=0.0, borderpad=0.25
    )

    frame = legend.get_frame()
    frame.set_edgecolor("lightgray")
    frame.set_linewidth(0.8)
    frame.set_facecolor("white")

    # 凡例のBBoxだけで保存（余白ゼロ）
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer)
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    # ★ PNG 出力
    fig.savefig(out_path_base + ".png", dpi=600, bbox_inches=bbox, pad_inches=0.0)
    # ★ PDF 出力
    fig.savefig(out_path_base + ".pdf", dpi=600, bbox_inches=bbox, pad_inches=0.0)

    plt.close(fig)
    print(f"[legend] 出力: {out_path_base}.png / {out_path_base}.pdf")

# 既存の「単一アルゴリズムに m=2,4,6 を重ね描きする関数」で、
# 凡例を出さないように変更（labelは付けるが legend() は呼ばない）
def plot_function_call_multi(prob, alg, roi_type, curves, style_map=None):
    """
    curves: { m(int): np.ndarray(function_call) }
    style_map: { m(int): linestyle(str) }  例 {2:'-', 4:'--', 6:':'}
    """
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.grid(True, which='both', alpha=0.25)

    max_gen = 0
    for m, fc in sorted(curves.items()):
        gen = np.arange(1, len(fc) + 1)      # 横軸＝世代（1始まり）
        max_gen = max(max_gen, gen[-1])
        ls = style_map.get(m, '-') if style_map else '-'
        ax.plot(gen, fc, linewidth=7, linestyle=ls, label=rf"$m={m}$")  # ★labelは付けるがlegendは出さない

    # 軸：縦 0〜500 固定（余白なし）、横は 1..最大世代
    ax.set_xlim(1, max_gen)
    ax.set_ylim(0, 500)
    ax.set_xlabel(r'Generation', fontsize=30)
    ax.set_ylabel(r'Num. of function calls', fontsize=30)
    ax.tick_params(axis='both', labelsize=30)
    ax.tick_params(axis='y', which='both', labelleft=True)  # 全画像でY目盛数値を表示
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.96)

    out_dir = f'../output/function_call_median/{roi_type}/{alg}/{prob}/'
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, 'function_call_progress_no-legend.pdf')
    png_path = os.path.join(out_dir, 'function_call_progress_no-legend.png')
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f"[per-alg] 出力: {pdf_path}")


def plot_function_call_row(prob, roi_type, alg_to_curves, alg_captions=None,
                           use_log=False):
    """
    alg_to_curves: { alg_name: { m(int): np.ndarray(function_call) } }
    use_log=True で軸を log に（画像例の雰囲気）
    """
    alg_list = [k for k,v in alg_to_curves.items() if len(v) > 0]
    if not alg_list:
        return

    n_cols = len(alg_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(15.5, 4.5), sharex=True)

    if n_cols == 1:
        axes = [axes]

    # 軸に描く & 凡例の元ネタ（最初の軸のハンドルを共通凡例に使用）
    legend_handles, legend_labels = None, None

    for ax, alg in zip(axes, alg_list):
        curves = alg_to_curves[alg]
        # m=2,4,6 を重ね描き
        for m, fc in sorted(curves.items()):
            x = np.arange(len(fc))
            line, = ax.plot(x, fc, linewidth=2, label=rf"$m={m}$")
        # 見た目
        ax.set_xlim(1, 500)
        ax.set_ylim(0, 500)
        ax.tick_params(axis='y', which='both', labelleft=True)
        ax.grid(True, which='both', alpha=0.25)
        if use_log:
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'Generation', fontsize=13)
        ax.set_ylabel(r'Num. of function calls', fontsize=13)

        # パネルタイトル（アルゴリズム名）
        title = alg_captions.get(alg, alg) if alg_captions else alg
        ax.set_title(title, fontsize=14)

        # 1つ目の軸の凡例情報を保存（共通凡例として上に出す）
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # 共有凡例（上部）
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center',
                   ncol=min(6, len(legend_labels)), frameon=False,
                   bbox_to_anchor=(0.5, 1.02), fontsize=12)

    # 余白調整（上部に凡例スペース）
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.80, wspace=0.15)

    out_dir = f'../output/function_call_median_multi_row/{roi_type}/{prob}/'
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f'function_call_progress_row.pdf')
    png_path = os.path.join(out_dir, f'function_call_progress_row.png')
    plt.savefig(pdf_path, dpi=600)
    plt.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f"[row] 図を出力: {pdf_path}")
    print(f"[row] 図を出力: {png_path}")


# 追加（基準にするpop_sel）
REF_POP = 'POP'
n = [50000, 50086, 102340, 455126, 3162510]
# ---------- メイン ----------
def main():
    # mごとの線種（凡例と図で統一）
    STYLE_MAP = {2: '-', 4: '--', 6: ':'}

    # 表示ラベル（凡例画像用）。存在しないmは後で間引きます
    LABEL_MAP = {2: r"$m=2$", 4: r"$m=4$", 6: r"$m=6$"}

    for roi_type in ["roi-c", "roi-p"]:
        for prob in problems:
            # この (roi_type, prob) で実際に描けた m を集め、凡例画像を出す
            ms_present = set()

            for alg in algorithms:
                curves = {}

                for m in n_obj_list:
                    # 参照PFロード
                    pf_path = f'../ref_point_dataset/{prob}_d{m}_n{n[m - 2]}.csv'
                    pf_npy  = pf_path.replace('.csv', '.npy')
                    if not os.path.exists(pf_npy):
                        PF = load_pf(prob, m, t, pf_path)
                        np.save(pf_npy, PF)
                    else:
                        PF = np.load(pf_npy)

                    z = load_ref_point(prob, m, t, roi_type)
                    ideal, nadir = PF.min(axis=0), PF.max(axis=0)
                    PF_norm = normalize_points(PF, ideal, nadir)
                    r_radius_elipse = np.array([nadir[i] * r_radius for i in range(m)])

                    # POP で中央値 run を決める
                    ref_vals = []
                    for run in range(n_runs):
                        ref_csv, ref_igd = sol_path(REF_POP, alg, prob, m, run, roi_type)
                        if not os.path.exists(ref_csv):
                            continue

                        if os.path.exists(ref_igd):
                            igd = float(np.loadtxt(ref_igd))
                        else:
                            X = np.loadtxt(ref_csv, delimiter=',', ndmin=2)
                            if roi_type == "roi-c":
                                igd = compute_igd_c_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir, prob, m)
                            else:
                                igd = compute_igd_p_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir, prob, m)
                            np.savetxt(ref_igd, [igd])
                        ref_vals.append((run, igd))

                    if not ref_vals:
                        continue

                    ref_median_run = select_median_run([v for _, v in ref_vals])
                    if ref_median_run is None:
                        continue

                    # 中央値 run の function_call を読み込む（1列ベクトル想定）
                    function_call_path = (
                        f'../output/function_call_results/{roi_type}/{alg}/{prob}/m{m}/'
                        f'function_call_{ref_median_run}th_run.csv'
                    )
                    if not os.path.exists(function_call_path):
                        continue

                    function_call = np.loadtxt(function_call_path)
                    # ※ 累積/非累積は与データ仕様に合わせて。増分にしたいなら↓を有効化
                    # function_call = np.diff(function_call, prepend=0)

                    curves[m] = function_call
                    ms_present.add(m)

                # アルゴリズムごとの画像（凡例なし）を出力
                if curves:
                    plot_function_call_multi(prob, alg, roi_type, curves, style_map=STYLE_MAP)

            # ---- 上部凡例だけの画像（この prob×roi_type で実際に描けた m のみ）----
            if ms_present:
                labs  = [LABEL_MAP[m] for m in sorted(ms_present)]
                lstyles = [STYLE_MAP[m] for m in sorted(ms_present)]
                legend_dir = f'../output/function_call_median/{roi_type}/{prob}/'
                os.makedirs(legend_dir, exist_ok=True)
                legend_path = os.path.join(legend_dir, 'legend_m_list')
                save_top_legend_image(labs, lstyles, legend_path, ncol=len(labs))
                print(f"[legend] 出力: {legend_path}")


if __name__ == "__main__":
    main()
