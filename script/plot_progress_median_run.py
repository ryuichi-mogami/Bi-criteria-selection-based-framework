#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

# =========================
# ユーザー設定
# =========================
n_obj_list   = [2]                 # 目的数 m
problems     = ["DTLZ2"]           # 対象問題
mult_ref     = 1                   # 参照点の個数
roi_type     = "roi-p"             # ROI のタイプ（ディレクトリ名に使う）
algorithms   = ["NSGA2", "BNSG2"]  # 比較するアルゴリズム
n_runs       = 31                  # 試行回数
r_radius     = 0.1                 # ROI 半径（元スケール）
out_dir      = f"../output/igdC_plus_summary_{mult_ref}"
dpi_save     = 600

# 1000 評価ごとにプロット
fevals_start = 1000
fevals_step  = 1000
fevals_end   = 50000  # 最大評価回数（必要に応じて変更）

# POP のファイル名フォーマット（既存と合わせる）
POP_FNAME_FMT = "pop_{nth}th_run_{fevals}fevals.csv"

# PF データのサンプル数（貼ってくれたコードと同じ）
# m = 2,3,4,5,6 に対応していると仮定
PF_SIZES = [50000, 50086, 102340, 455126, 3162510]

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'


# ---------- Utility ----------
def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom

def compute_igd_c_plus(X, PF, PF_norm, z, r_vec, ideal, nadir, prob, m):
    """
    貼ってくれた compute_igd_c_plus をほぼそのまま使用。
    ここでも pivot_{mult_ref} ディレクトリを使ってピボットを共有する。
    X      : アルゴリズムが得た解集合 (N x m)
    PF     : 近似パレートフロント (K x m)
    z      : 参照点 (m,) もしくは (1, m)
    r_vec  : 各目的の ROI 半径（元スケール）(m,) ここでは楕円半径として使用
    """
    # z は 1 つのみを仮定（mult_ref=1）
    z = np.asarray(z).ravel()

    # ピボットの選択（PF 上で z に最も近い点）
    distance_list = np.zeros(len(PF))
    for i, p in enumerate(PF):
        distance_list[i] = np.linalg.norm(p - z)

    pivot_dir = f"../output/pivot_{mult_ref}"
    os.makedirs(pivot_dir, exist_ok=True)
    pivot_file = os.path.join(pivot_dir, f"{prob}_{m}.csv")

    if os.path.exists(pivot_file):
        pivot_id = int(np.loadtxt(pivot_file))
        pivot_point = PF[pivot_id]
    else:
        pivot_id = np.argmin(distance_list)
        np.savetxt(pivot_file, [pivot_id], fmt="%d")
        pivot_point = PF[pivot_id]

    # ROI: 楕円 (diff / r_vec)^2 の和が 1 以下の点を PF から抽出
    diff = PF - pivot_point
    val = np.sum((diff / r_vec) ** 2, axis=1)
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


def compute_igd_p_plus(X, PF, PF_norm, z, r, ideal, nadir, prob, m):   
    less_eq = np.all(PF <= z, axis=1)   # すべての目的で参照点以下（理想側）
    greater_eq = np.all(PF >= z, axis=1)  # すべての目的で参照点以上（非理想側）
    mask = np.logical_or(less_eq, greater_eq)   # どちらか一方を満たす個体を選択
    # print(format(m), prob)
    # print("less_true   =", np.sum(less_eq))
    # print("greater_true=", np.sum(greater_eq))
    # print("union S'    =", np.sum(mask)) 
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

def load_pf(prob, m):
    """PF の CSV を読み込み / npy キャッシュ（既存コードと同じ仕組み）"""
    pf_csv = f"../ref_point_dataset/{prob}_d{m}_n{PF_SIZES[m - 2]}.csv"
    pf_npy = pf_csv.replace(".csv", ".npy")

    if os.path.exists(pf_npy):
        PF = np.load(pf_npy)
    else:
        PF = np.loadtxt(pf_csv, delimiter=",")
        np.save(pf_npy, PF)
    return PF


def load_ref_point(prob, m, t=1):
    """参照点を読み込む（roi-c を想定）"""
    return np.loadtxt(
        f"../ref_point_data/roi-c/m{m}_{prob}_type{t}.csv",
        delimiter=",",
        ndmin=1,
    )


def get_pop_path(alg, prob, m, run, fevals):
    """母集団 CSV のパスを生成（既存 results ディレクトリ構造を踏襲）"""
    if alg == "NSGA2":
        base = f"../output/results_{mult_ref}/emo/{alg}/{prob}/m{m}/"
    else:
        base = f"../output/results_{mult_ref}/{roi_type}/{alg}/{prob}/m{m}/"
    os.makedirs(base, exist_ok=True)
    fname = POP_FNAME_FMT.format(nth=run, fevals=fevals)
    return os.path.join(base, fname)


def get_igd_path(alg, prob, m, run, fevals):
    """IGD-C+ CSV のパスを生成"""
    if alg == "NSGA2":
        base_igd = f"../output/igdC_plus_{mult_ref}/emo-{roi_type}/{alg}/{prob}/m{m}/"
    else:
        base_igd = f"../output/igdC_plus_{mult_ref}/{roi_type}/{alg}/{prob}/m{m}/"
    os.makedirs(base_igd, exist_ok=True)
    fname = POP_FNAME_FMT.format(nth=run, fevals=fevals)
    return os.path.join(base_igd, fname)


def compute_or_load_igd_for_run(alg, prob, m, run, fevals, PF, PF_norm, z, r_vec, ideal, nadir):
    """
    あるアルゴリズム/試行/run/fevals について
    - IGD-C+ ファイルがあれば読む
    - なければ母集団から計算して保存
    """
    igd_path = get_igd_path(alg, prob, m, run, fevals)
    if os.path.exists(igd_path):
        try:
            return float(np.loadtxt(igd_path))
        except Exception:
            # 読めなければ計算し直す
            pass

    pop_path = get_pop_path(alg, prob, m, run, fevals)
    if not os.path.exists(pop_path):
        # この run / fevals のデータが無い場合
        return np.nan

    X = np.loadtxt(pop_path, delimiter=",", ndmin=2)
    if roi_type == "roi-p":
        igd = compute_igd_p_plus(X, PF, PF_norm, z, r_radius, ideal, nadir, prob, m)
    else:
        igd = compute_igd_c_plus(X, PF, PF_norm, z, r_vec, ideal, nadir, prob, m)
    np.savetxt(igd_path, [igd])
    return igd


def main():
    os.makedirs(out_dir, exist_ok=True)

    # ---- ここで評価回数リストを決める ----
    # 200 〜 50000 を 200 ごとにサンプリング（例）
    fevals_list = (
        list(range(100, 1001, 100)) +   # 100,200,...,1000
        list(range(2000, 50001, 1000))  # 2000,3000,...,50000
    )
    # ※ 本当に 1000 ごとで十分なら、ここを
    #    fevals_list = list(range(1000, 50001, 1000))
    #    に変えてください。目盛りだけ 200,10K,... にすることもできます。

    for m in n_obj_list:
        for prob in problems:
            PF = load_pf(prob, m)
            ideal, nadir = PF.min(axis=0), PF.max(axis=0)
            PF_norm = normalize_points(PF, ideal, nadir)

            z = load_ref_point(prob, m, t=1)
            z = np.asarray(z).ravel()
            r_vec = np.full(m, r_radius)

            alg_curves = {}

            for alg in algorithms:
                median_igd_list = []
                for fevals in fevals_list:
                    igd_vals = []
                    for run in range(n_runs):
                        igd = compute_or_load_igd_for_run(
                            alg, prob, m, run, fevals,
                            PF, PF_norm, z, r_vec, ideal, nadir
                        )
                        if not np.isnan(igd):
                            igd_vals.append(igd)

                    if len(igd_vals) == 0:
                        median_igd = np.nan
                    else:
                        median_igd = float(np.median(igd_vals))
                    median_igd_list.append(median_igd)

                alg_curves[alg] = np.array(median_igd_list)

            # ---------- プロット ----------
            fig, ax = plt.subplots(figsize=(9, 6.0))  # 図サイズは好みで調整
            ax.set_xlim(100, 50000)
            ls = ["-", "--", ":"]
            for i, alg in enumerate(algorithms):
                y = alg_curves[alg]
                ax.plot(
                    fevals_list,
                    y,
                    linewidth=7.0,
                    label=alg,
                    linestyle = ls[i],
                )

            # --- x 軸: 200, 10K, 20K, 30K, 40K, 50K 表示 ---
            xticks = [200, 10000, 20000, 30000, 40000, 50000]
            ax.set_xticks(xticks)
            ax.set_xticklabels(["200", "10K", "20K", "30K", "40K", "50K"])

            # ax.set_xlabel("Num. of function evaluations", fontsize=30)

            # --- y 軸: 対数目盛 & 10^0,10^-2,... の表示 ---
            ax.set_yscale("log")
            ax.set_ylim(0.0001,10)
            yticks = [10**e for e in range(0, -5, -2)]  # 10^0,10^-2,...,10^-8
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            # ax.set_ylabel("Average $\\text{IGD}^+ \\text{-C}$ values", fontsize=30)


            ax.tick_params(axis="both", labelsize=30)
            # ★ グリッドは major だけにする
            ax.grid(which="major", linestyle="-", alpha=0.6)

            # minor grid を完全に消す
            ax.grid(which="minor", visible=False)
            # ax.grid(True, which="both", alpha = 0.6)
            # ax.legend(fontsize=10, frameon=False)

            plt.subplots_adjust(left=0.2, right=0.92, bottom=0.18, top=0.96)

            save_dir = os.path.join(out_dir, roi_type, prob, f"m{m}")
            os.makedirs(save_dir, exist_ok=True)
            png_path = os.path.join(save_dir, "igdC_plus_RNSGA2_vs_BNSGA2.png")
            pdf_path = os.path.join(save_dir, "igdC_plus_RNSGA2_vs_BNSGA2.pdf")

            plt.savefig(png_path, dpi=dpi_save)
            plt.savefig(pdf_path, dpi=dpi_save)
            plt.close(fig)

            print(f"保存しました: {png_path}")


if __name__ == "__main__":
    main()
