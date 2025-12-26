#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.stats import wilcoxon
import time

# =========================================
# 設定
# =========================================
n_obj     = [2, 4, 6]
problems  = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7']
ALGO_SETS = {
    'IGD-C': ['BNSGA2', 'BIBEA', 'BSMSEMOA', 'NSGA2'],
    'IGD-P': ['BNSGA2', 'NSGA2'],
}
ROI2METRIC = {
    'roi-c': 'IGD-C',
    'roi-p': 'IGD-P',
}
algorithm_captions = {
    'BNSGA2':   'B-NSGA-II',
    'BIBEA':    'B-IBEA',
    'BSMSEMOA': 'B-SMS-EMOA',
    'RNSGA2':   'R-NSGA-II',
    'gNSGA2':   'g-NSGA-II',
    'NSGA2':    'NSGA-II',
    'BNSGA3':  'B-NSGA-III',
    'BSPEA2': 'B-SPEA2',
}

t        = 1
n_runs   = 31
alpha    = 0.05
r_radius = 0.1

# PF サンプル数（m = 2,4,6 に対応）
n_pf = [50000, 50086, 102340, 455126, 3162510]


# =========================================
# 補助関数 (IGD^+ の計算)
# =========================================
def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom


def classify_ref_point_against_pf(z, PF, tol=1e-9):
    """
    z が PF の内側(不可能)か外側(実行可能)か、または PF 上かを判定。
    ルール（最小化前提）:
      - PF 上:     ∃p∈PF s.t. max_i |p_i - z_i| <= tol
      - 実行可能:  ∃p∈PF s.t. p <= z + tol  （z は PF より上側＝劣側）
      - 不可能:    上記のいずれでもない（= PF の内側＝PF より良い側）
    戻り値: "on-pf" | "feasible" | "infeasible"
    """
    # PF 上（境界上）
    if np.any(np.all(np.abs(PF - z) <= tol, axis=1)):
        return "on-pf"
    # 実行可能（PF のどれかが z を支配/同等以下）
    if np.any(np.all(PF <= (z + tol), axis=1)):
        return "feasible"
    # 内側（不可能）
    return "infeasible"

def compute_igd_c_plus(X, PF, z, r, ideal, nadir, prob, m):
    """
    ROI-C 用 IGD^+-C
    """
    pivot_dir = "./pivot"
    os.makedirs(pivot_dir, exist_ok=True)
    pivot_file = os.path.join(pivot_dir, f"{prob}_{m}.csv")

    # pivot 点の決定（キャッシュあり）
    if os.path.exists(pivot_file):
        pivot_id = int(np.loadtxt(pivot_file))
        pivot_point = PF[pivot_id]
    else:
        distance_list = np.linalg.norm(PF - z, axis=1)
        pivot_id = int(np.argmin(distance_list))
        np.savetxt(pivot_file, [pivot_id], fmt='%d')
        pivot_point = PF[pivot_id]

    # 楕円 ROI 内の PF 点だけを残す
    diff = PF - pivot_point
    val = np.sum((diff / r) ** 2, axis=1)
    mask = val <= 1.0
    S_prime = PF[mask]
    if S_prime.shape[0] == 0:
        return np.nan

    igd_vals = []
    for s in S_prime:
        diff = X - s
        diff_pos = np.maximum(diff, 0.0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))


def compute_igd_p_plus(X, PF, z):
    """
    ROI-P 用 IGD^+-P
    """
    less_eq = np.all(PF <= z, axis=1)
    greater_eq = np.all(PF >= z, axis=1)
    mask = np.logical_or(less_eq, greater_eq)
    S_prime = PF[mask]
    if S_prime.shape[0] == 0:
        return np.nan

    igd_vals = []
    for s in S_prime:
        diff = X - s
        diff_pos = np.maximum(diff, 0.0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))


# =========================================
# メイン：ROIごとに Wilcoxon 集計
# =========================================
os.makedirs('../output/results_table', exist_ok=True)

for roi_type in ["roi-p"]:
    metric = ROI2METRIC[roi_type]
    algorithms = ALGO_SETS[metric]
    # 問題ごとの raw データ格納:
    # results_raw[prob][alg][m] = [runごとのIGD値 ...]
    results_raw = {
        prob: {alg: {m: [] for m in n_obj} for alg in algorithms}
        for prob in problems
    }
    z_region_status = {prob: {m: None for m in n_obj} for prob in problems}
    # IGD 値を読み込み（無ければ計算）
    for run in range(n_runs):
        for prob in problems:
            for m in n_obj:
                # 参照点
                z = np.loadtxt(
                    f'../ref_point_data/{roi_type}/m{m}_{prob}_type{t}.csv',
                    delimiter=',',
                    ndmin=1
                )

                # PF の読み込み（キャッシュ: .npy）
                pf_path = f'../ref_point_dataset/{prob}_d{m}_n{n_pf[m - 2]}.csv'
                pf_npy = pf_path.replace('.csv', '.npy')
                if not os.path.exists(pf_npy):
                    PF = np.loadtxt(pf_path, delimiter=',')
                    np.save(pf_npy, PF)
                else:
                    PF = np.load(pf_npy)

                true_ideal = PF.min(axis=0)
                true_nadir = PF.max(axis=0)

                # ROI-C 用の楕円半径ベクトル
                r_radius_ellipse = (true_nadir * r_radius)

                # 正規化 PF (ROI-P では実質使わないが一応計算しておく)
                norm_npy = pf_path.replace('.csv', '_norm.npy')
                if not os.path.exists(norm_npy):
                    PF_norm = normalize_points(PF, true_ideal, true_nadir)
                    np.save(norm_npy, PF_norm)
                else:
                    PF_norm = np.load(norm_npy)
                if roi_type == "roi-p" and z_region_status[prob][m] is None:
                    z_region_status[prob][m] = classify_ref_point_against_pf(z, PF)
                for alg in algorithms:
                    # ROI-C では gNSGA2 を使わない，ROI-P では RNSGA2 を使わない
                    if roi_type == "roi-c" and alg == "gNSGA2":
                        continue
                    if roi_type == "roi-p" and alg == "RNSGA2":
                        continue

                    if alg == "NSGA2":
                        sol_file = (
                            f'../output/results/emo/{alg}/{prob}/m{m}/'
                            f'pop_{run}th_run_50000fevals.csv'
                        )
                        igd_file = (
                            f'../output/igdC_plus/emo-{roi_type}/{alg}/{prob}/m{m}/'
                            f'pop_{run}th_run_50000fevals.csv'
                        )
                    else:
                        sol_file = (
                            f'../output/results/{roi_type}/{alg}/{prob}/m{m}/'
                            f'pop_{run}th_run_50000fevals.csv'
                        )
                        # IGD ファイルパス
                        igd_dir = 'igdC_plus'
                        igd_file = sol_file.replace('results', igd_dir)

                    if os.path.exists(igd_file):
                        igd_val = float(np.loadtxt(igd_file))
                    else:
                        # 解集合を読み込み，IGD を計算
                        X = np.loadtxt(sol_file, delimiter=',', ndmin=2)
                        if metric == 'IGD-C':
                            igd_val = compute_igd_c_plus(
                                X, PF, z, r_radius_ellipse,
                                true_ideal, true_nadir, prob, m
                            )
                        else:
                            igd_val = compute_igd_p_plus(X, PF, z)

                        os.makedirs(os.path.dirname(igd_file), exist_ok=True)
                        np.savetxt(igd_file, [igd_val], fmt='%.8e')

                    results_raw[prob][alg][m].append(igd_val)

    # =========================================
    # Wilcoxon 検定で + / - / ≈ をカウント
    # =========================================
    if metric == 'IGD-C':
        baseline_alg = 'RNSGA2'
    else:
        baseline_alg = 'gNSGA2'

    baseline_alg = 'NSGA2'  # 修正

    summary = {
        alg: {
            m: {'plus': 0, 'minus': 0, 'approx': 0}
            for m in n_obj
        }
        for alg in algorithms
        if alg != baseline_alg
    }

    for prob in problems:
        for m in n_obj:
            base_vals = np.array(results_raw[prob][baseline_alg][m], dtype=float)

            # ベースラインのデータが無ければスキップ
            if len(base_vals) == 0:
                continue

            for alg in algorithms:
                if alg == baseline_alg:
                    continue
                alg_vals = np.array(results_raw[prob][alg][m], dtype=float)

                if len(alg_vals) == 0:
                    continue

                # NaN を除外
                mask = np.isfinite(base_vals) & np.isfinite(alg_vals)
                base_clean = base_vals[mask]
                alg_clean = alg_vals[mask]

                if len(base_clean) < 2 or len(alg_clean) < 2:
                    continue

                # ほぼ同じなら ≈
                if np.allclose(base_clean, alg_clean):
                    mark = 'approx'
                else:
                    try:
                        _, p = wilcoxon(alg_clean, base_clean)
                    except ValueError:
                        # 差がすべて 0 など
                        mark = 'approx'
                    else:
                        if p < alpha:
                            if alg_clean.mean() < base_clean.mean():
                                mark = 'plus'
                            elif alg_clean.mean() > base_clean.mean():
                                mark = 'minus'
                            else:
                                mark = 'approx'
                        else:
                            mark = 'approx'
                    if alg == 'BNSGA2':
                        ztag_map = {"feasible": "F", "infeasible": "I", "on-pf": "="}
                        ztag = ""
                        if mark == "plus":
                            mark_str = "+"
                        elif mark == "minus":
                            mark_str = "-"
                        else:
                            mark_str = "≈"
                        if roi_type == "roi-p":
                            ztag = f", z={ztag_map.get(z_region_status[prob][m], '?')}"
                        print(
                            f"[info] {roi_type}, {prob}, m={m}, "
                            f"{algorithm_captions[alg]} vs {algorithm_captions[baseline_alg]}: "
                            f"p={p:.4e}, {alg_clean.mean():.4e} > {base_clean.mean():.4e} ({mark_str}){ztag}"
                        )
                summary[alg][m][mark] += 1

    # =========================================
    # LaTeX 表として出力（画像の形）
    # =========================================
    out_path = f'../output/results_table/wilcoxon_{roi_type}.tex'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(r'\begin{table}[t]' + '\n')
        f.write(r'  \centering' + '\n')
        f.write(r'  \footnotesize' + '\n')
        if roi_type == 'roi-c':
            caption = 'ROI-C に対する Wilcoxon 符号付順位和検定の結果.'
        else:
            caption = 'ROI-P に対する Wilcoxon 符号付順位和検定の結果.'
        f.write(f'  \\caption{{{caption}}}\n')
        f.write(f'  \\label{{tab:wilcoxon_{roi_type}}}\n')
        f.write(r'  \begin{tabular}{ccc}' + '\n')
        f.write(r'    (a) B-NSGA-II & (b) B-IBEA & (c) B-SMS-EMOA \\' + '\n')

        alg_order = ['BNSGA2', 'BIBEA', 'BSMSEMOA']

        for j, alg in enumerate(alg_order):
            if alg == baseline_alg:
                continue  # 念のため

            if j == 0:
                f.write('    ')
            else:
                f.write('    & ')

            f.write(r'\begin{tabular}{cccc}' + '\n')
            f.write(r'      \toprule' + '\n')
            f.write(r'      $m$ & $+$ & $-$ & $\approx$ \\' + '\n')
            f.write(r'      \midrule' + '\n')
            for m in n_obj:
                cnt = summary[alg][m]
                f.write(
                    f'      {m} & {cnt["plus"]} & {cnt["minus"]} & {cnt["approx"]} \\\\\n'
                )
            f.write(r'      \bottomrule' + '\n')
            f.write(r'    \end{tabular}')
            if j == len(alg_order) - 1:
                f.write(r'\\' + '\n')
            else:
                f.write('\n')

        f.write(r'  \end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')

    print(f"{roi_type} の Wilcoxon 表を {out_path} に出力しました。")
