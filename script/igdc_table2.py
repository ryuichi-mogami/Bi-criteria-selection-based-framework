#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np

# =========================
# Settings
# =========================
start_all = time.perf_counter()

n_obj    = [2, 4, 6]
problems = ['DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7']

ALGO_SETS = {
    'IGD-C': ['BNSGA2', 'BIBEA', 'BSMSEMOA', 'BSPEA2', 'BNSGA3','BNSGA2-drs', 'BSPEA2-drs', 'RNSGA2'],
    'IGD-P': ['BNSGA2', 'BIBEA', 'BSMSEMOA', 'BSPEA2', 'BNSGA3', 'BNSGA2-drs', 'BSPEA2-drs', 'gNSGA2'],
}
ROI2METRIC = {
    'roi-c': 'IGD-C',
    'roi-p': 'IGD-P',
}
algorithm_captions = {
    'BNSGA2':      'B-NSGA-II',
    'BIBEA':       'B-IBEA',
    'BSMSEMOA':    'B-SMS-EMOA',
    'RNSGA2':      'R-NSGA-II',
    'gNSGA2':      'g-NSGA-II',
    'BNSGA2-drs':  'B-MNSGA-II',
    'BNSGA3':      'B-NSGA-III',
    'BSPEA2':      'B-SPEA2',
    'BSPEA2-drs':  'B-MSPEA2',
}

t        = 1
n_runs   = 31
r_radius = 0.1

# DTLZ PF size mapping (your original)
n_pf = [50000, 50086, 102340, 455126, 3162510]  # index by (m-2)

# Output
output_path = f'../output/results_table/igdc_igdp_side_by_side.tex'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# =========================
# Utilities
# =========================
def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom

def compute_igd_c_plus(X, PF, PF_norm, z, r, ideal, nadir, prob, m):
    pivot_dir = "../output/pivot"
    os.makedirs(pivot_dir, exist_ok=True)
    pivot_file = os.path.join(pivot_dir, f"{prob}_{m}.csv")

    # pivot
    if os.path.exists(pivot_file):
        pivot_id = int(np.loadtxt(pivot_file))
        pivot_point = PF[pivot_id]
    else:
        distance_list = np.zeros(len(PF))
        for i, p in enumerate(PF):
            distance_list[i] = np.linalg.norm(p - z)
        pivot_id = int(np.argmin(distance_list))
        np.savetxt(pivot_file, [pivot_id], fmt='%d')
        pivot_point = PF[pivot_id]

    # ROI mask (ellipse in objective space)
    diff = PF - pivot_point
    val = np.sum((diff / r) ** 2, axis=1)
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
    less_eq    = np.all(PF <= z, axis=1)
    greater_eq = np.all(PF >= z, axis=1)
    mask = np.logical_or(less_eq, greater_eq)
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

def safe_nanmean(arr):
    v = float(np.nanmean(arr))
    return v

# =========================
# Main: compute both ROI-C and ROI-P, then output ONE table
# =========================
results_raw_by_roi = {}  # roi_type -> results_raw dict

for roi_type in ["roi-c", "roi-p"]:
    metric = ROI2METRIC[roi_type]
    algorithms = ALGO_SETS[metric]

    # results_raw[prob][alg][m] = list of igd values (len = n_runs)
    results_raw = {
        prob: {
            alg: {m: [] for m in n_obj}
            for alg in algorithms
        }
        for prob in problems
    }

    for run in range(n_runs):
        for prob in problems:
            for m in n_obj:
                # ref point
                z = np.loadtxt(
                    f'../ref_point_data/{roi_type}/m{m}_{prob}_type{t}.csv',
                    delimiter=',', ndmin=1
                )

                # PF load/cache
                pf_path = f'../ref_point_dataset/{prob}_d{m}_n{n_pf[m - 2]}.csv'
                pf_npy  = pf_path.replace('.csv', '.npy')
                if not os.path.exists(pf_npy):
                    PF = np.loadtxt(pf_path, delimiter=',')
                    np.save(pf_npy, PF)
                else:
                    PF = np.load(pf_npy)

                true_ideal = PF.min(axis=0)
                true_nadir = PF.max(axis=0)

                # ellipse radius per objective (your original)
                r_radius_elipse = np.array([true_nadir[i] * r_radius for i in range(m)])

                # PF_norm cache
                norm_npy = pf_path.replace('.csv', '_norm.npy')
                if not os.path.exists(norm_npy):
                    PF_norm = normalize_points(PF, true_ideal, true_nadir)
                    np.save(norm_npy, PF_norm)
                else:
                    PF_norm = np.load(norm_npy)

                # per algorithm
                for alg in algorithms:
                    # your original skip rules
                    if alg == "RNSGA2" and roi_type == "roi-p":
                        continue
                    if alg == "gNSGA2" and roi_type == "roi-c":
                        continue

                    # solution file path
                    if alg in {"NSGA2", "IBEA", "SMSEMOA", "NSGA3", "SPEA2"}:
                        sol_file = f'../output/results_1/emo/{alg}/{prob}/m{m}/pop_{run}th_run_50000fevals.csv'
                    else:
                        sol_file = f'../output/results_1/{roi_type}/{alg}/{prob}/m{m}/pop_{run}th_run_50000fevals.csv'

                    # cached igd file
                    igd_file = sol_file.replace('results_1', 'igdC_plus_1')
                    igd_file = igd_file.replace('emo', f'emo_{roi_type}')

                    if os.path.exists(igd_file):
                        igd_val = float(np.loadtxt(igd_file))
                    else:
                        X = np.loadtxt(sol_file, delimiter=',', ndmin=2)
                        if len(X) < 100:
                            print(f'警告: {sol_file} の解の数が 100 未満です。')

                        if roi_type == "roi-c":
                            igd_val = compute_igd_c_plus(
                                X, PF, PF_norm, z, r_radius_elipse, true_ideal, true_nadir, prob, m
                            )
                        else:
                            igd_val = compute_igd_p_plus(
                                X, PF, PF_norm, z, r_radius_elipse, true_ideal, true_nadir, prob, m
                            )

                        os.makedirs(os.path.dirname(igd_file), exist_ok=True)
                        np.savetxt(igd_file, [igd_val], fmt='%.8e')

                    results_raw[prob][alg][m].append(igd_val)

    results_raw_by_roi[roi_type] = results_raw

# =========================
# Output: one side-by-side table (ROI-C left, ROI-P right)
# =========================
alg_c = ALGO_SETS['IGD-C']
alg_p = ALGO_SETS['IGD-P']

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(r'\begin{table*}[t]' + '\n')
    f.write(r'  \centering' + '\n')
    f.write(r'  \footnotesize' + '\n')
    f.write(r'  \caption{Comparison results for ROI-C (left) and ROI-P (right).}' + '\n')
    f.write(r'  \label{tab:igdc_igdp_side_by_side}' + '\n\n')

    # Column format (no vertical rules; booktabs-friendly)
    col_fmt = (
        'c c '
        + ' '.join('c' for _ in alg_c)
        + r' @{\hspace{1.5em}} '
        + ' '.join('c' for _ in alg_p)
    )
    f.write(r'  \resizebox{\linewidth}{!}{%' + '\n')
    f.write(f'  \\begin{{tabular}}{{{col_fmt}}}\n')
    f.write(r'    \toprule' + '\n')

    # Header row 1: ROI-C / ROI-P group labels
    n_c = len(alg_c)
    n_p = len(alg_p)
    f.write(
        '    ' +
        r'\multicolumn{2}{c}{}' +
        f' & \\multicolumn{{{n_c}}}{{c}}{{ROI-C}}' +
        f' & \\multicolumn{{{n_p}}}{{c}}{{ROI-P}}' +
        r' \\' + '\n'
    )

    # cmidrules for groups
    c_start = 3
    c_end   = 2 + n_c
    p_start = 3 + n_c
    p_end   = 2 + n_c + n_p
    f.write(f'    \\cmidrule(lr){{{c_start}-{c_end}}}\\cmidrule(lr){{{p_start}-{p_end}}}\n')

    # Header row 2: column names
    header = 'Problem & $m$ & ' \
             + ' & '.join(algorithm_captions[s] for s in alg_c) \
             + ' & ' \
             + ' & '.join(algorithm_captions[s] for s in alg_p) \
             + r' \\'
    f.write(f'    {header}\n')
    f.write(r'    \midrule' + '\n')

    # Body
    for prob in problems:
        first = True
        for m in n_obj:
            cells = []
            if first:
                cells.append(f'\\multirow{{{len(n_obj)}}}{{*}}{{{prob}}}')
                first = False
            else:
                cells.append('')
            cells.append(str(m))

            # ROI-C block
            rc = results_raw_by_roi['roi-c']
            mus_c = {alg: safe_nanmean(rc[prob][alg][m]) for alg in alg_c}
            baseline_c = 'RNSGA2'
            base_c = mus_c.get(baseline_c, np.nan)
            shade_c = {}
            for alg in alg_c:
                if np.isnan(base_c) or np.isnan(mus_c[alg]):
                    shade_c[alg] = 0
                else:
                    shade_c[alg] = 20 if mus_c[alg] < base_c else 0

            for alg in alg_c:
                mu = mus_c[alg]
                cells.append(f'\\cellcolor{{black!{shade_c[alg]}}}{mu:.4f}')

            # ROI-P block
            rp = results_raw_by_roi['roi-p']
            mus_p = {alg: safe_nanmean(rp[prob][alg][m]) for alg in alg_p}
            baseline_p = 'gNSGA2'
            base_p = mus_p.get(baseline_p, np.nan)
            shade_p = {}
            for alg in alg_p:
                if np.isnan(base_p) or np.isnan(mus_p[alg]):
                    shade_p[alg] = 0
                else:
                    shade_p[alg] = 20 if mus_p[alg] < base_p else 0

            for alg in alg_p:
                mu = mus_p[alg]
                cells.append(f'\\cellcolor{{black!{shade_p[alg]}}}{mu:.4f}')

            f.write('    ' + ' & '.join(cells) + r' \\' + '\n')
        if prob != problems[-1]:
            f.write(r'\midrule' + '\n')

    f.write(r'    \bottomrule' + '\n')
    f.write(r'  \end{tabular}' + '\n')
    f.write(r'  }%' + '\n')
    f.write(r'\end{table*}' + '\n')

elapsed_all = time.perf_counter() - start_all
print(f'統合テーブルを {output_path} に出力しました。')
print(f'[time] total: {elapsed_all:.2f}s')
