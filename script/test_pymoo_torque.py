import os
import sys
sys.path.insert(0, os.path.abspath("../pymoo"))
sys.path.insert(0, os.path.abspath(".."))
import click

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.ibea import IBEA
from pymoo.algorithms.moo.moead import MOEAD

# from pymoo.algorithms.moo.bibea import BIBEA
# from pymoo.algorithms.moo.bnsga2 import BNSGA2
# from pymoo.algorithms.moo.bsms import BSMSEMOA

from bsf.bibea import BIBEA
from bsf.bnsga2 import BNSGA2
from bsf.bsms import BSMSEMOA

# from pymoo.algorithms.moo.rnsga2 import RNSGA2
# from pymoo.algorithms.moo.gnsga2 import gNSGA2

from bsf.rnsga2 import RNSGA2
from bsf.gnsga2 import gNSGA2


from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination

from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.decomposition.pbi import PBI

import numpy as np

import argparse
# @click.command()
# #@click.option('--torque_flag', is_flag=True, help='Run by Torque')
# @click.option('--n_obj', '-m', required=False, default=2, type=int, help='Number of objectives.')
# @click.option('--problem_name', '-p', required=True, type=str, help='Problem.')
# @click.option('--alg', '-a', required=True, type=str, help='Name of EMOA.')
# @click.option('--run_id', '-r', required=False, default=0, type=int, help='Run ID.')
def run(n_obj, problem_name, alg, run_id, roi_type):  
    emo_max_fess = 50000
    termination = get_termination("n_eval", emo_max_fess)    

    # WFGのときはn_var (変数の数) の指定が必要
    if problem_name.startswith("wfg") or problem_name.startswith("WFG"):
        if n_obj == 2:
            k = 4
        else:
            k = 2 * (n_obj - 1)
        l = 20  # 慣例的固定値
        n_var = k + l
        problem = get_problem(problem_name, n_obj=n_obj, n_var=n_var)
    else:#DTLZのとき
        problem = get_problem(problem_name, n_obj=n_obj)

    if alg == "BNSGA2":
        algorithm = BNSGA2(pop_size=100, roi_type = roi_type)
    elif alg == "BIBEA":
        algorithm = BIBEA(pop_size=100, roi_type = roi_type)
    elif alg == "BSMSEMOA":
        algorithm = BSMSEMOA(pop_size=100, n_offspring=1, roi_type = roi_type)
    elif alg == "RNSGA2":
        algorithm = RNSGA2(pop_size=100)
    elif alg == "gNSGA2":
        #change capital problem_name to match file name dtlz2 -> DTLZ2
        problem_name = problem_name.upper()
        algorithm = gNSGA2(n_obj=n_obj, problem_name=problem_name, pop_size=100)
    elif alg == "NSGA2":
        algorithm = NSGA2(pop_size=100)
    elif alg == "SMSEMOA":
        algorithm = SMSEMOA(pop_size=100, n_offspring=1)
    elif alg == "IBEA":
        algorithm = IBEA(pop_size=100)
    # else:
    #     print(f"{alg} is not available.")
    #     exit()
    # algorithm = BNSGA2(pop_size=100)
    # algorithm = BIBEA(pop_size=100)
    # algorithm = BSMSEMOA(pop_size=100)        
    #algorithm = NSGA2(pop_size=100)
    #algorithm = SMSEMOA(pop_size=100)
    # moead_ref_dirs = get_reference_directions("energy", n_obj, 100, seed=1)
    #algorithm = MOEAD(moead_ref_dirs, n_neighbors=15, prob_neighbor_mating=0.9, decomposition=PBI())
    # algorithm = NSGA3(moead_ref_dirs)

    res = minimize(problem, algorithm, termination, seed=run_id, verbose=False, save_history=True)
    # print("count_each:", res.algorithm.survival.count_each, len(res.algorithm.survival.count_each))
    for idx, fevals in enumerate(range(100, 50001, 100)):
        F_gen = res.history[idx].pop.get("F") 
        res_dir_path = os.path.join('../output/results', f'{roi_type}/{alg}/{problem_name.upper()}/m{n_obj}')
        os.makedirs(res_dir_path, exist_ok=True)
        res_file_path = os.path.join(res_dir_path, f'pop_{run_id}th_run_{fevals}fevals.csv')
        # print(res_file_path)
        np.savetxt(res_file_path, F_gen, delimiter=',') 
    
    if alg != "RNSGA2" and alg != "gNSGA2" and roi_type != "emo":
        res_dir_path = os.path.join('../output/function_call_results', f'{roi_type}/{alg}/{problem_name.upper()}/m{n_obj}')
        os.makedirs(res_dir_path, exist_ok=True)
        count_each = np.array(res.algorithm.survival.count_each)
        res_file_path = os.path.join(res_dir_path, f'function_call_{run_id}th_run.csv')
        np.savetxt(res_file_path, count_each, fmt="%d")  
    # for idx, fevals in enumerate(range(100, 1000, 100)):
    #     F_gen = res.history[idx].pop.get("F") 
    #     res_dir_path = os.path.join('./results', f'{alg}/{problem_name.upper()}/m{n_obj}')
    #     os.makedirs(res_dir_path, exist_ok=True)
    #     res_file_path = os.path.join(res_dir_path, f'pop_{run_id}th_run_{fevals}fevals.csv')
    #     # print(res_file_path)
    #     np.savetxt(res_file_path, F_gen, delimiter=',') 

    # for idx, fevals in enumerate(range(1000, 50001, 1000)):
    #     F_gen = res.history[(idx + 1) * 10 - 1].pop.get("F") 
    #     res_dir_path = os.path.join('./results', f'{alg}/{problem_name.upper()}/m{n_obj}')
    #     os.makedirs(res_dir_path, exist_ok=True)
    #     res_file_path = os.path.join(res_dir_path, f'pop_{run_id}th_run_{fevals}fevals.csv')
    #     # print(res_file_path)
    #     np.savetxt(res_file_path, F_gen, delimiter=',') 

    # for idx in range(20, 491, 10):
    #     F_gen = res.history[idx].pop.get("F")
    #     print(str(idx)+":"+str(idx*100))
    #     res_dir_path = os.path.join('./results', f'{alg}/{problem_name.upper()}/m{n_obj}')
    #     os.makedirs(res_dir_path, exist_ok=True)
    #     res_file_path = os.path.join(res_dir_path, f'pop_{run_id}th_run_{idx*100}fevals.csv')
    #     np.savetxt(res_file_path, F_gen, delimiter=',') 

    # for fx in res.F:
    #     print(f"{fx[0]} {fx[1]}")
    
    # res_dir_path = os.path.join('./results', f'{alg}/{problem_name.upper()}/m{n_obj}')
    # os.makedirs(res_dir_path, exist_ok=True)
    # res_file_path = os.path.join(res_dir_path, f'pop_{run_id}th_run_50000fevals.csv')
    # np.savetxt(res_file_path, res.F, delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_obj', type=int)
    parser.add_argument('--problem_name', type=str)
    parser.add_argument('--alg', type=str)
    parser.add_argument('--run_id', type=int)
    parser.add_argument('--roi_type', type=str)
    args = parser.parse_args()

    n_obj = args.n_obj   
    problem_name = args.problem_name
    alg = args.alg
    run_id = args.run_id
    roi_type = args.roi_type
    run(n_obj, problem_name, alg, run_id, roi_type)
