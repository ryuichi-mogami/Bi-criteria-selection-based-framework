#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wilcoxon summary table generator (compact, multi-budget, pairwise baselines)
+ ROI-P only: include DRS-handled variant row, e.g., B-NSGA-II-DRS vs NSGA-II.

- For each ROI type (roi-c / roi-p) and each metric (IGD-C / IGD-P):
  compare each target algorithm against its own baseline algorithm.
- For ROI-P only: additionally compare B-NSGA-II-DRS vs NSGA-II.
- For each (problem, m, budget), do paired Wilcoxon signed-rank test using 31 runs.
- Summarize per (target, m, budget): count of + / - / ≈ over 7 problems.
- Output one compact LaTeX table per ROI type per metric:
    ../output/results_table/wilcoxon_{roi_type}_{metric}_compact.tex
"""

import os
import numpy as np
from scipy.stats import wilcoxon

# =========================================
# Settings
# =========================================
n_obj     = [2, 4, 6]
problems  = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7']

# budgets to compare (fevals)
BUDGETS = [10000, 30000, 50000]

t        = 1
n_runs   = 31
alpha    = 0.05
r_radius = 0.1

# PF sample sizes (index by m-2)
n_pf = [50000, 50086, 102340, 455126, 3162510]

ROI2METRIC = {
    'roi-c': 'IGD-C',
    'roi-p': 'IGD-P',
}

# ---- Pairwise comparisons by ROI type
# "target_alg_key" vs "baseline_alg_key"
COMPARISON_PAIRS_BY_ROI = {
    # ROI-Cに載せたい比較（必要に応じて編集）
    "roi-c": [
        ('BNSGA2',   'NSGA2'),
        ('BIBEA',    'IBEA'),
        ('BSMSEMOA', 'SMSEMOA'),
        ('BSPEA2',   'SPEA2'),
        ('BNSGA3',   'NSGA3'),
        ('BNSGA2-drs',  'NSGA2'),
        ('BSPEA2-drs',  'SPEA2'),
    ],
    # ROI-Pに載せたい比較 + DRS対処版を追加
    "roi-p": [
        ('BNSGA2',   'NSGA2'),
        ('BIBEA',    'IBEA'),
        ('BSMSEMOA', 'SMSEMOA'),
        ('BSPEA2',   'SPEA2'),
        ('BNSGA3',   'NSGA3'),
        ('BNSGA2-drs',  'NSGA2'),
        ('BSPEA2-drs',  'SPEA2'),
    ]
}

# Display names (optional)
algorithm_captions = {
    # B-variants
    'BNSGA2':      'B-NSGA-II',
    'BNSGA2-drs':  'B-MNSGA-II',   # ★表記
    'BIBEA':       'B-IBEA',
    'BSMSEMOA':    'B-SMS-EMOA',
    'BNSGA3':      'B-NSGA-III',
    'BSPEA2':      'B-SPEA2',
    'BSPEA2-drs':  'B-MSPEA2',   # ★表記
    # originals
    'NSGA2':    'NSGA-II',
    'IBEA':     'IBEA',
    'SMSEMOA':  'SMS-EMOA',
    'NSGA3':    'NSGA-III',
    'SPEA2':    'SPEA2',
}

# If your baseline directory name differs from the algorithm key, map here.
BASELINE_NAME_MAP = {
    # e.g. 'SMSEMOA': 'SMS-EMOA'  # folder name if different
}

# ★重要：target側（ROI側）のフォルダ名がキーと違う場合のマップ
# 例: 実フォルダが "BNSGA2-DRS" なら {'BNSGA2_DRS': 'BNSGA2-DRS'} にする
TARGET_FOLDER_NAME_MAP = {
    # 'BNSGA2_DRS': 'BNSGA2-DRS',
}

# Roots (adjust if needed)
BASELINE_PATH_ROOT = "../output/results_1/emo"   # originals (non-ROI)
ROI_PATH_ROOT      = "../output/results_1"       # ROI methods: ../output/results/{roi_type}/{ALG}/{PROB}/m{m}/...
IGD_CACHE_ROOTNAME = "igdC_plus"               # results -> igdC_plus (keep your convention)

os.makedirs("../output/results_table", exist_ok=True)


# =========================================
# Helpers
# =========================================
def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom


def classify_ref_point_against_pf(z, PF, tol=1e-9):
    if np.any(np.all(np.abs(PF - z) <= tol, axis=1)):
        return "on-pf"
    if np.any(np.all(PF <= (z + tol), axis=1)):
        return "feasible"
    return "infeasible"


def compute_igd_c_plus(X, PF, z, r, ideal, nadir, prob, m):
    pivot_dir = "./pivot"
    os.makedirs(pivot_dir, exist_ok=True)
    pivot_file = os.path.join(pivot_dir, f"{prob}_{m}.csv")

    if os.path.exists(pivot_file):
        pivot_id = int(np.loadtxt(pivot_file))
        pivot_point = PF[pivot_id]
    else:
        distance_list = np.linalg.norm(PF - z, axis=1)
        pivot_id = int(np.argmin(distance_list))
        np.savetxt(pivot_file, [pivot_id], fmt='%d')
        pivot_point = PF[pivot_id]

    diff = PF - pivot_point
    val = np.sum((diff / r) ** 2, axis=1)
    mask = val <= 1.0
    S_prime = PF[mask]
    if S_prime.shape[0] == 0:
        return np.nan

    igd_vals = []
    for s in S_prime:
        diff2 = X - s
        diff_pos = np.maximum(diff2, 0.0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))


def compute_igd_p_plus(X, PF, z):
    less_eq = np.all(PF <= z, axis=1)
    greater_eq = np.all(PF >= z, axis=1)
    mask = np.logical_or(less_eq, greater_eq)
    S_prime = PF[mask]
    if S_prime.shape[0] == 0:
        return np.nan

    igd_vals = []
    for s in S_prime:
        diff2 = X - s
        diff_pos = np.maximum(diff2, 0.0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))


def get_baseline_folder_name(alg_key: str) -> str:
    return BASELINE_NAME_MAP.get(alg_key, alg_key)


def get_target_folder_name(alg_key: str) -> str:
    return TARGET_FOLDER_NAME_MAP.get(alg_key, alg_key)


def is_roi_target(roi_type: str, alg_key: str) -> bool:
    # ROI側に置かれている「target（B系など）」は、比較ペアの左側に現れるものだけ
    targets = set(a for a, _ in COMPARISON_PAIRS_BY_ROI[roi_type])
    return alg_key in targets


def sol_path_for(roi_type: str, alg: str, prob: str, m: int, run: int, budget: int) -> str:
    """
    - ROI targets: ../output/results_1/{roi_type}/{alg}/{prob}/m{m}/pop_{run}th_run_{budget}fevals.csv
    - Baselines:   ../output/results_1/emo/{alg}/{prob}/m{m}/pop_{run}th_run_{budget}fevals.csv
    """
    if is_roi_target(roi_type, alg):
        folder = get_target_folder_name(alg)
        return os.path.join(
            ROI_PATH_ROOT, roi_type, folder, prob, f"m{m}",
            f"pop_{run}th_run_{budget}fevals.csv"
        )
    else:
        folder = get_baseline_folder_name(alg)
        return os.path.join(
            BASELINE_PATH_ROOT, folder, prob, f"m{m}",
            f"pop_{run}th_run_{budget}fevals.csv"
        )


def igd_cache_path_from_sol(sol_file: str) -> str:
    return sol_file.replace("results", IGD_CACHE_ROOTNAME)


# =========================================
# Main
# =========================================
for roi_type in ["roi-c", "roi-p"]:
    metric = ROI2METRIC[roi_type]
    pairs = COMPARISON_PAIRS_BY_ROI[roi_type]

    targets   = [a for a, _ in pairs]
    baselines = [b for _, b in pairs]
    algorithms_all = sorted(set(targets + baselines))

    # raw data:
    # results_raw[budget][prob][alg][m] -> list of igd values over runs
    results_raw = {
        budget: {
            prob: {alg: {mm: [] for mm in n_obj} for alg in algorithms_all}
            for prob in problems
        }
        for budget in BUDGETS
    }

    z_region_status = {prob: {mm: None for mm in n_obj} for prob in problems}

    # ---- Load / compute IGD values
    for run in range(n_runs):
        for prob in problems:
            for mm in n_obj:
                z = np.loadtxt(
                    f'../ref_point_data/{roi_type}/m{mm}_{prob}_type{t}.csv',
                    delimiter=',',
                    ndmin=1
                )

                pf_path = f'../ref_point_dataset/{prob}_d{mm}_n{n_pf[mm - 2]}.csv'
                pf_npy = pf_path.replace('.csv', '.npy')
                if not os.path.exists(pf_npy):
                    PF = np.loadtxt(pf_path, delimiter=',')
                    np.save(pf_npy, PF)
                else:
                    PF = np.load(pf_npy)

                true_ideal = PF.min(axis=0)
                true_nadir = PF.max(axis=0)
                r_radius_ellipse = (true_nadir * r_radius)

                if roi_type == "roi-p" and z_region_status[prob][mm] is None:
                    z_region_status[prob][mm] = classify_ref_point_against_pf(z, PF)

                for alg in algorithms_all:
                    for budget in BUDGETS:
                        sol_file = sol_path_for(roi_type, alg, prob, mm, run, budget)

                        if not os.path.exists(sol_file):
                            results_raw[budget][prob][alg][mm].append(np.nan)
                            continue

                        igd_file = igd_cache_path_from_sol(sol_file)

                        if os.path.exists(igd_file):
                            igd_val = float(np.loadtxt(igd_file))
                        else:
                            X = np.loadtxt(sol_file, delimiter=',', ndmin=2)
                            if metric == "IGD-C":
                                igd_val = compute_igd_c_plus(
                                    X, PF, z, r_radius_ellipse,
                                    true_ideal, true_nadir, prob, mm
                                )
                            else:
                                igd_val = compute_igd_p_plus(X, PF, z)

                            os.makedirs(os.path.dirname(igd_file), exist_ok=True)
                            np.savetxt(igd_file, [igd_val], fmt="%.8e")

                        results_raw[budget][prob][alg][mm].append(igd_val)

    # ---- Wilcoxon summary counts
    summary = {
        target: {
            mm: {
                budget: {'plus': 0, 'minus': 0, 'approx': 0}
                for budget in BUDGETS
            }
            for mm in n_obj
        }
        for target in targets
    }

    for (target_alg, baseline_alg) in pairs:
        for prob in problems:
            for mm in n_obj:
                for budget in BUDGETS:
                    base_vals = np.array(results_raw[budget][prob][baseline_alg][mm], dtype=float)
                    alg_vals  = np.array(results_raw[budget][prob][target_alg][mm], dtype=float)

                    mask = np.isfinite(base_vals) & np.isfinite(alg_vals)
                    base_clean = base_vals[mask]
                    alg_clean  = alg_vals[mask]

                    if len(base_clean) < 2 or len(alg_clean) < 2:
                        summary[target_alg][mm][budget]['approx'] += 1
                        continue

                    if np.allclose(base_clean, alg_clean):
                        mark = 'approx'
                    else:
                        try:
                            _, p = wilcoxon(alg_clean, base_clean)
                        except ValueError:
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

                    summary[target_alg][mm][budget][mark] += 1

    # ---- LaTeX output
    out_path = f'../output/results_table/wilcoxon_{roi_type}_{metric}_compact.tex'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(r'\begin{table}[t]' + '\n')
        f.write(r'  \centering' + '\n')
        f.write(r'  \footnotesize' + '\n')
        cap_metric = metric
        caption = (
            f'Pair-wise comparison between an original EMO algorithm'
            f'and its BSF version on the seven DTLZ problems for the {roi_type.upper()}. '
        )
        f.write(f'  \\caption{{{caption}}}\n')
        f.write(f'  \\label{{tab:wilcoxon_{roi_type}_{cap_metric}_compact}}\n')
        f.write(r'  \resizebox{\linewidth}{!}{%' + '\n') 
        colspec = 'l c ' + ' '.join(['c c c' for _ in BUDGETS])
        f.write(f'  \\begin{{tabular}}{{{colspec}}}\n')
        f.write(r'    \toprule' + '\n')

        header1 = r'    \multicolumn{2}{c}{Original EMO vs B-EMOAs}'
        for budget in BUDGETS:
            header1 += f' & \\multicolumn{{3}}{{c}}{{{int(budget/1000)}k}}'
        header1 += r' \\'
        f.write(header1 + '\n')

        # ---- cmidrules (add 1-2 for the left block, then budgets)
        cm = r'    \cmidrule(lr){1-2}'
        start = 3
        for _ in BUDGETS:
            cm += f'\\cmidrule(lr){{{start}-{start+2}}}'
            start += 3
        f.write(cm + '\n')

        # ---- header row 2 (m is a normal column header, not a dangling row)
        header2 = r'    Algorithm & $m$'
        for _ in BUDGETS:
            header2 += r' & $+$ & $-$ & $\approx$'
        header2 += r' \\'
        f.write(header2 + '\n')

        f.write(r'    \midrule' + '\n')

        # Body: keep the order as given in pairs (so DRS row can be placed next to BNSGA2)
        for (target_alg, baseline_alg) in pairs:
            method_name = (
                f'{algorithm_captions.get(baseline_alg, baseline_alg)}'
                f' vs {algorithm_captions.get(target_alg, target_alg)} '
            )
            for i, mm in enumerate(n_obj):
                if mm == 4:
                    f.write(f'    {method_name} & {mm}')
                else:
                    f.write(f'    {" " * len(method_name)} & {mm}')

                for budget in BUDGETS:
                    cnt = summary[target_alg][mm][budget]
                    f.write(f' & {cnt["plus"]} & {cnt["minus"]} & {cnt["approx"]}')
                f.write(r' \\' + '\n')

            # if you want a lighter separation, consider \addlinespace instead of midrule
            if target_alg != pairs[-1][0]:
                f.write(r'    \midrule' + '\n')

        f.write(r'    \bottomrule' + '\n')
        f.write(r'  \end{tabular}' + '\n')
        f.write(r'  }% ' + '\n')
        f.write(r'\end{table}' + '\n')

    print(f"[done] {roi_type} / {metric} -> {out_path}")
