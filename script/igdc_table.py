import os
import numpy as np
import glob
import re
from scipy.stats import wilcoxon
import cProfile, pstats

import time

start_all = time.perf_counter()  # 全体計測開始

n_obj        = [2,4,6] 
problems     = ['DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7']#'DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7' , 'SDTLZ1', 'SDTLZ2', 'SDTLZ3', 'SDTLZ4'
ALGO_SETS = {
    'IGD-C': ['BNSGA2', 'BIBEA', 'BSMSEMOA', 'RNSGA2'],
    'IGD-P': ['BNSGA2','gNSGA2'],
}
ROI2METRIC = {
    'roi-c': 'IGD-C',
    'roi-p': 'IGD-P',
}
algorithm_captions = {
    'BNSGA2':   'B-NSGA-II',
    'BIBEA':   'B-IBEA',
    'BSMSEMOA':   'B-SMS-EMOA',
    'RNSGA2':   'R-NSGA-II',
    'gNSGA2':   'g-NSGA-II',
}

t         = 1
n_runs    = 31
alpha     = 0.05
r_radius  = 0.1
def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom

def compute_igd_c_plus(X, PF, PF_norm, z, r, ideal, nadir, prob, m):
    # X_norm = normalize_points(X, ideal, nadir)
    # z_norm = normalize_points(z, ideal, nadir)
    pivot_dir = "../output/pivot"
    os.makedirs(pivot_dir, exist_ok=True)
    pivot_file = os.path.join(pivot_dir, f"{prob}_{m}.csv")

    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    # print(f"[pivot calc] {prob}-m{m}-run{run}: {t1 - t0:.2f}s")

    t0 = time.perf_counter()
    del_mask = np.full(len(PF), False)
    diff = PF - pivot_point
    val = np.sum((diff/r)**2, axis = 1)
    mask = val <= 1.0
    S_prime = PF[mask]
    t1 = time.perf_counter()
    # print(f"[mask calc] {prob}-m{m}-run{run}: {t1 - t0:.2f}s")
    print(len(S_prime))
    print(str(m),prob)
    t0 = time.perf_counter()
    if S_prime.shape[0] == 0:
        return np.nan
    igd_vals = []
    for s in S_prime:
        diff = X - s
        diff_pos = np.maximum(diff, 0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    t1 = time.perf_counter()
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


output_path = f'../output/results_table/igdc_table.tex'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
n = [50000, 50086, 102340, 455126, 3162510]

for roi_type in ["roi-p"]:
    metric = ROI2METRIC[roi_type]
    algorithms = ALGO_SETS[metric]

    results_raw = {
        prob: {                               # 第1層キー: 問題名 (prob)
            alg: {                            # 第2層キー: アルゴリズム名 (alg)
                m: []                         # 第3層キー: 目的数 (m)
                for m in n_obj                # ← 第3層ループ
            }for alg in algorithms            # ← 第2層ループ
        }for prob in problems                  # ← 第1層ループ
    }

    for run in range(n_runs):
        for prob in problems:
            for m in n_obj:
                z  = np.loadtxt(
                    f'../ref_point_data/{roi_type}/m{m}_{prob}_type{t}.csv',
                    delimiter=',', ndmin=1
                )
                pf_path = f'../ref_point_dataset/{prob}_d{m}_n{n[m - 2]}.csv'
                pf_npy = pf_path.replace('.csv', '.npy')
                if not os.path.exists(pf_npy):
                    PF = np.loadtxt(pf_path, delimiter=',')
                    np.save(pf_npy, PF)
                else:
                    PF = np.load(pf_npy) 
                true_ideal = PF.min(axis=0)    # [ideal_x, ideal_y]
                true_nadir = PF.max(axis=0)    # [nadir_x, nadir_y]
                print("true_ideal:", true_ideal)
                print("true_nadir:", true_nadir)
                data = []
                for i in range(m):
                    data.append(true_nadir[i] *  r_radius)
                r_radius_elipse = np.array(data)
                # ループの中で……
                norm_npy     = pf_path.replace('.csv', '_norm.npy')
                #   2) ref_point も同様に正規化
                ref_norm = normalize_points(z, true_ideal, true_nadir)
                if not os.path.exists(norm_npy):
                    # 正規化
                    PF_norm = normalize_points(PF, true_ideal, true_nadir)
                    # バイナリで保存
                    np.save(norm_npy, PF_norm)
                else:
                    # バイナリ読み込みは C 実装なので非常に高速
                    PF_norm  = np.load(norm_npy)
                for alg in algorithms:
                    if alg == "RNSGA2" and roi_type =="roi-p":
                        continue
                    if alg == "gNSGA2" and roi_type =="roi-c":
                        continue
                    if alg == "NSGA2" or alg == "IBEA" or alg == "SMSEMOA":
                        sol_file = f'../output/results/emo/{alg}/{prob}/m{m}/pop_{run}th_run_50000fevals.csv'
                    else:
                        sol_file = f'../output/results/{roi_type}/{alg}/{prob}/m{m}/pop_{run}th_run_50000fevals.csv'
                    igdc_file = sol_file.replace('results', 'igdC_plus')
                    igdc_file = igdc_file.replace('emo', f'emo_{roi_type}')
                    if os.path.exists(igdc_file):
                        igdc_val = float(np.loadtxt(igdc_file))
                    else:
                        X = np.loadtxt(sol_file, delimiter=',', ndmin=2)
                        if len(X) < 100:
                            print(f'警告: {sol_file} の解の数が 100 未満です。')
                        if roi_type == "roi-c":  
                            igdc_val = compute_igd_c_plus(
                                X, PF,PF_norm,z, r_radius_elipse, true_ideal, true_nadir, prob, m
                            )
                        elif roi_type == "roi-p":
                            igdc_val = compute_igd_p_plus(
                                X, PF,PF_norm,z, r_radius_elipse, true_ideal, true_nadir, prob, m
                            )
                        os.makedirs(os.path.dirname(igdc_file), exist_ok=True)
                        np.savetxt(igdc_file, [igdc_val], fmt='%.8e')
                    results_raw[prob][alg][m].append(igdc_val)

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(r'\begin{table}[t]' + '\n')
        f.write(r'  \centering' + '\n')
        f.write(r'  \footnotesize' + '\n')
        caption_text = f"IGD$^+$-Cの31試行の平均値を示す. " if metric == 'IGD-C' \
                else f"IGD$^+$-Pの31試行の平均値を示す. "
        f.write(f'  \\caption{{{caption_text}}}\n')
        label_key = 'igdc' if metric == 'IGD-C' else 'igdp'
        f.write(f'  \\label{{tab:{label_key}}}' + '\n\n')

        col_fmt = 'c c ' + ' '.join('c' for _ in algorithms)
        f.write(r'  \resizebox{\linewidth}{!}{%' + '\n') 
        f.write(f'  \\begin{{tabular}}{{{col_fmt}}}' + '\n')
        f.write(r'    \toprule' + '\n')

        header = 'Problem & $m$ & ' + ' & '.join(algorithm_captions[s] for s in algorithms) + r' \\'
        f.write(f'    {header}' + '\n')
        for prob in problems:
            f.write(r'    \midrule' + '\n')
            first = True
            for m in n_obj:
                cells = []
                if first:
                    cells.append(f'\\multirow{{{len(n_obj)}}}{{*}}{{{prob}}}')
                    first = False
                else:
                    cells.append('')
                cells.append(str(m))

                # 各手法の平均を計算して昇順ソート
                mus = {alg: float(np.nanmean(results_raw[prob][alg][m])) for alg in algorithms}
                # ベースライン（表ごとに切替）
                if metric == 'IGD-C':
                    baseline_alg = 'RNSGA2'
                else:
                    baseline_alg = 'gNSGA2'
                ua_baseline = mus.get(baseline_alg, np.nan)

                # idx_map = {alg: i for i, alg in enumerate(algorithms)}
                # sorted_sels = sorted(
                #     algorithms,
                #     key=lambda s: (mus[s], -idx_map[s])
                # )
                #first place shading
                # levels = [20,0,0,0]
                # shade = {alg: levels[i] for i, alg in enumerate(sorted_sels)}
                
                # baseline over shading
                shade = {}
                for alg in algorithms:
                    if np.isnan(ua_baseline) or np.isnan(mus[alg]):
                        shade[alg] = 0
                    else:
                        shade[alg] = 20 if mus[alg] < ua_baseline else 0

                for i, alg in enumerate(algorithms):
                    mu = mus[alg]
                #     # Wilcoxon マークの計算（標準出力と同じロジック）
                #     if i == 0:
                #         mark = ''
                #     else:
                #         marks = []
                #         for prev in pop_selection[:i]:
                #             pv = np.array(results_raw[prob][alg][m][prev])
                #             cv = np.array(results_raw[prob][alg][m][sel])
                #             if np.allclose(cv, pv):
                #                 marks.append('$\\approx$')
                #             else:
                #                 _, p = wilcoxon(cv, pv)
                #                 if p < alpha:
                #                     # μ(sel) が前の平均より小さければ「+」、大きければ「−」
                #                     marks.append('$+$' if mu < pv.mean() else '$−$')
                #                 else:
                #                     marks.append('$\\approx$')
                #         mark = f'({", ".join(marks)})'

                #     # セルにマークを付けて書き込む
                #     cells.append(f'\\cellcolor{{black!{shade[sel]}}}{mu:.4f}{mark}')
                # for i, sel in enumerate(pop_selection):
                #     # mu = np.mean(results_raw[prob][alg][m][sel])
                #     mu = mus[sel]
                #     # ペアごとの Wilcoxon 検定
                #     # ・i==1（2番目）は 0番目と比較
                #     # ・i==3（4番目）は 2番目と比較
                #     if i in (1, 3):
                #         prev_sel = pop_selection[i - 1]
                #         pv = np.array(results_raw[prob][alg][m][prev_sel])
                #         cv = np.array(results_raw[prob][alg][m][sel])

                #         if np.allclose(cv, pv):
                #             mark_symbol = '$\\approx$'
                #         else:
                #             _, p = wilcoxon(cv, pv)
                #             if p < alpha:
                #                 # μ(sel) が μ(prev) より小さければ「+」，大きければ「−」
                #                 mark_symbol = '$+$' if mu < pv.mean() else '$−$'
                #             else:
                #                 mark_symbol = '$\\approx$'

                #         mark = f'({mark_symbol})'
                #     else:
                #         mark = ''

                    # セルにマークを付けて書き込む
                    if alg == "BSMSEMOA" and m >= 4:
                        cells.append('-')
                    else:
                        cells.append(f'\\cellcolor{{black!{shade[alg]}}}{mu:.4f}')
                    # cells.append(f'\\cellcolor{{black!{shade[sel]}}}{mu:.4f}{mark}')

                line = '    ' + ' & '.join(cells) + r' \\'
                f.write(line + '\n')

        f.write(r'    \bottomrule' + '\n')
        f.write(r'  \end{tabular}' + '\n')
        f.write(r'  }%    ← resizebox の閉じ' + '\n')
        f.write(r'\end{table}' + '\n\n')
            

    print(f'テーブルを {output_path} に出力しました。')