import os
import sys
sys.path.insert(0, os.path.abspath("./pymoo"))
import click

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.ibea import IBEA
from pymoo.algorithms.moo.moead import MOEAD

from pymoo.algorithms.moo.bibea import BIBEA
from pymoo.algorithms.moo.bnsga2 import BNSGA2
from pymoo.algorithms.moo.bsms import BSMSEMOA
from pymoo.algorithms.moo.rnsga2 import RNSGA2

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination

from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.decomposition.pbi import PBI

import numpy as np

# @click.command()
# #@click.option('--torque_flag', is_flag=True, help='Run by Torque')
# @click.option('--n_obj', '-m', required=False, default=2, type=int, help='Number of objectives.')
# @click.option('--problem_name', '-p', required=True, type=str, help='Problem.')
# @click.option('--alg', '-a', required=True, type=str, help='Name of EMOA.')
# @click.option('--run_id', '-r', required=False, default=0, type=int, help='Run ID.')
def run(n_obj, problem_name, alg, run_id):  
    emo_max_fess = 50000
    termination = get_termination("n_eval", emo_max_fess)    

    # WFGのときはn_var (変数の数) の指定が必要
    #problem = get_problem(problem_name, n_obj=n_obj, n_var=10)

    # DTLZのとき
    problem = get_problem(problem_name, n_obj=n_obj)
            
    # algorithm = BNSGA2(pop_size=100)
    algorithm = BIBEA(pop_size=100)
    # algorithm = BSMSEMOA(pop_size=100)        
    #algorithm = NSGA2(pop_size=100)
    #algorithm = SMSEMOA(pop_size=100)
    # moead_ref_dirs = get_reference_directions("energy", n_obj, 100, seed=1)
    #algorithm = MOEAD(moead_ref_dirs, n_neighbors=15, prob_neighbor_mating=0.9, decomposition=PBI())
    # algorithm = NSGA3(moead_ref_dirs)

    res = minimize(problem, algorithm, termination, seed=run_id, verbose=False)

    # for fx in res.F:
    #     print(f"{fx[0]} {fx[1]}")
    
    res_dir_path = os.path.join('./results', f'{problem_name.upper()}_m{n_obj}')
    os.makedirs(res_dir_path, exist_ok=True)
    res_file_path = os.path.join(res_dir_path, f'final_pop_{run_id}th_run.csv')
    np.savetxt(res_file_path, res.F, delimiter=',')

if __name__ == '__main__':
    n_obj = 2
    problem_name = "dtlz4"
    alg = "dumy"
    run_id = 0
    
    run(n_obj, problem_name, alg, run_id)
