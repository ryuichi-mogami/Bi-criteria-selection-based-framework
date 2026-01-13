#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.stats import wilcoxon
import argparse
# =========================================
# Settings
# =========================================

r_radius = 0.1

# PF sample sizes (index by m-2)
n_pf = [50000, 50086, 102340, 455126, 3162510]

ROI2METRIC = {
    'roi-c': 'IGD-C',
    'roi-p': 'IGD-P',
}

# =========================================
# Helpers
# =========================================

def classify_ref_point_against_pf(z, PF, tol=1e-9):
    if np.any(np.all(np.abs(PF - z) <= tol, axis=1)):
        return "on-pf"
    if np.any(np.all(PF <= (z + tol), axis=1)):
        return "feasible"
    return "infeasible"


def compute_igd_c_plus(X_norm, PF_norm, z_norm, r, prob, m):
    pivot_dir = "./output/pivot"
    os.makedirs(pivot_dir, exist_ok=True)
    pivot_file = os.path.join(pivot_dir, f"{prob}_{m}.csv")

    if os.path.exists(pivot_file):
        pivot_id = int(np.loadtxt(pivot_file))
        pivot_point = PF_norm[pivot_id]
    else:
        distance_list = np.linalg.norm(PF_norm - z_norm, axis=1)
        pivot_id = int(np.argmin(distance_list))
        np.savetxt(pivot_file, [pivot_id], fmt='%d')
        pivot_point = PF_norm[pivot_id]

    diff = PF_norm - pivot_point
    val = np.sum((diff / r) ** 2, axis=1)
    mask = val <= 1.0
    S_prime = PF_norm[mask]
    if S_prime.shape[0] == 0:
        return np.nan

    igd_vals = []
    for s in S_prime:
        diff2 = X_norm - s
        diff_pos = np.maximum(diff2, 0.0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))


def compute_igd_p_plus(X_norm, PF_norm, z_norm):
    less_eq = np.all(PF_norm <= z_norm, axis=1)
    greater_eq = np.all(PF_norm >= z_norm, axis=1)
    mask = np.logical_or(less_eq, greater_eq)
    S_prime = PF_norm[mask]
    if S_prime.shape[0] == 0:
        return np.nan

    igd_vals = []
    for s in S_prime:
        diff2 = X_norm - s
        diff_pos = np.maximum(diff2, 0.0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))

def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom

def run(n_obj, problem_name, alg, run_id, roi_type, mult_ref,fevals): 
    t = 1 
    # ref point
    z = np.loadtxt(
        f'../ref_point_data/{roi_type}/m{n_obj}_{problem_name}_type{t}.csv',
        delimiter=',', ndmin=1
    )
    # PF load/cache
    pf_path = f'../ref_point_dataset/{problem_name}_d{n_obj}_n{n_pf[n_obj - 2]}.csv'
    pf_npy  = pf_path.replace('.csv', '.npy')
    if not os.path.exists(pf_npy):
        PF = np.loadtxt(pf_path, delimiter=',')
        np.save(pf_npy, PF)
    else:
        PF = np.load(pf_npy)

    true_ideal = PF.min(axis=0)
    true_nadir = PF.max(axis=0)
    z_norm = normalize_points(z, true_ideal, true_nadir)

    # ellipse radius per objective (your original)
    # r_radius_elipse = np.array([true_nadir[i] * r_radius for i in range(n_obj)])

    # PF_norm cache
    norm_npy = pf_path.replace('.csv', '_norm.npy')
    if not os.path.exists(norm_npy):
        PF_norm = normalize_points(PF, true_ideal, true_nadir)
        np.save(norm_npy, PF_norm)
    else:
        PF_norm = np.load(norm_npy)

    if alg in {"NSGA2", "IBEA", "SMSEMOA", "NSGA3", "SPEA2"}:
        sol_file = f'../output/results_{mult_ref}/emo/{alg}/{problem_name}/m{n_obj}/pop_{run_id}th_run_{fevals}fevals.csv'
    else:
        sol_file = f'../output/results_{mult_ref}/{roi_type}/{alg}/{problem_name}/m{n_obj}/pop_{run_id}th_run_{fevals}fevals.csv'

    # cached igd file
    igd_file = sol_file.replace(f'results_{mult_ref}', f'igdC_plus_{mult_ref}')
    igd_file = igd_file.replace('emo', f'emo_{roi_type}')

    if os.path.exists(igd_file):
        igd_val = float(np.loadtxt(igd_file))
    else:
        X = np.loadtxt(sol_file, delimiter=',', ndmin=2)
        X_norm = normalize_points(X, true_ideal, true_nadir)    
        if ROI2METRIC[roi_type] == "IGD-C":
            igd_val = compute_igd_c_plus(
                X_norm, PF_norm, z_norm, r_radius, problem_name, n_obj
            )
        else:
            igd_val = compute_igd_p_plus(
                X_norm, PF_norm, z_norm
            )
        os.makedirs(os.path.dirname(igd_file), exist_ok=True)
        np.savetxt(igd_file, [igd_val], fmt="%.8e")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_obj', type=int)
    parser.add_argument('--problem_name', type=str)
    parser.add_argument('--alg', type=str)
    parser.add_argument('--roi_type', type=str)
    parser.add_argument('--mult_ref', type=int)
    args = parser.parse_args()

    n_obj = args.n_obj   
    problem_name = args.problem_name
    alg = args.alg
    roi_type = args.roi_type
    mult_ref = args.mult_ref
    #100から1000まで100刻み、2000から50000まで1000刻み
    for run_id in range(31):
        for fevals in list(range(100,1100,100)) + list(range(2000,51000,1000)):
            # print(f"Evaluations: {fevals}")
            run(n_obj, problem_name, alg, run_id, roi_type,mult_ref, fevals)