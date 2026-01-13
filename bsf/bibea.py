import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath("../pymoo"))
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
# from pymoo.indicators.hv.exact import ExactHypervolume
# from pymoo.indicators.hv.exact_2d import ExactHypervolume2D
# from pymoo.indicators.hv.monte_carlo import ApproximateMonteCarloHypervolume
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function
# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize

from pymoo.operators.selection.rnd import RandomSelection
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

def asf1(point, ref_point, weight):
    scalar_value = np.max(weight * (point - ref_point))
    return scalar_value    

class FitnessAssignment(Survival):

    def __init__(self, roi_type, kappa=0.05, bq_indicator="epsilon") -> None:
        super().__init__(filter_infeasible=True)
        self.kappa = kappa
        self.bq_indicator = bq_indicator
        self.roi_type = roi_type
        self._ref_point_cache = {}
        self.count = 0
        self.count_each = []
        self.flag = None
        self.n = [50000, 50086, 102340, 455126, 3162510]

    def _load_ref_point(self, n_obj: int, problem_name: str) -> np.ndarray:
        key = (n_obj, problem_name) 
        if key in self._ref_point_cache:
            return self._ref_point_cache[key]
        ref_file = f"/home/mogami/bicriteria_pbemo/ref_point_data/{self.roi_type}/m{n_obj}_{problem_name}_type1.csv" 
        ref_point = np.loadtxt(ref_file, delimiter=",", dtype=float) 
        self._ref_point_cache[key] = ref_point 
        return ref_point

    def ibea_env(self, problem, pop, *args, n_survive=None, ideal=None, nadir=None, **kwargs):
        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # if the boundary points are not provided -> estimate them from pop
        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)

        # the number of objectives
        _, n_obj = F.shape
        PQ_size = len(F)
           
        # the final indices of surviving individuals
        survivor_list = list(range(PQ_size))
        denom = (nadir - ideal).astype(float)
        zero_mask = np.isclose(denom, 0.0)
        denom_safe = denom.copy()
        denom_safe[zero_mask] = 1.0  # 0除算回避
        normalized_F = (F - ideal) / denom_safe
        normalized_F[:, zero_mask] = 0.0  # 定数列は差0にする

        if self.bq_indicator == "epsilon":
            # epsilon_matrix = np.zeros((pop_size, pop_size))    
            # for i, Fi in enumerate(normalized_F):
            #     for j, Fj in enumerate(normalized_F):
            #         # This should not be np.max(Fi - Fj)
            #         epsilon_value = np.max(Fj - Fi)
            #         epsilon_matrix[j][i] = epsilon_value            
            bqi_matrix = np.max(normalized_F[:, None, :] - normalized_F[None, :, :], axis=2)
        else:
            raise ValueError(f"{self.bq_indicator} is not available")
            
        bqi_max = np.max(bqi_matrix)
        fitness_arr = np.zeros(PQ_size)
        for i in range(PQ_size):
            for j in range(PQ_size):            
                if i != j:
                    fitness_arr[i] += -np.exp(-bqi_matrix[j][i] / (bqi_max * self.kappa))              
                    
        while len(survivor_list) > n_survive:
            # The worst individual is removed from P \cup Q
            worst_id = np.argmin(fitness_arr)
            fitness_arr[worst_id] = np.inf
            survivor_list.remove(worst_id) 
            # Update the fitness values
            for i in range(PQ_size):
                if i != worst_id:
                    fitness_arr[i] += np.exp(-bqi_matrix[worst_id][i] / (bqi_max * self.kappa))
                    
        for i in range(PQ_size):        
            pop[i].set("fitness", fitness_arr[i])
        
        survivors = pop[survivor_list]
        
        return Population.create(*survivors)
        
    def _do(self, problem, pop, *args, n_survive=None, ideal=None, nadir=None, roi_radius=0.1, **kwargs):
        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # if the boundary points are not provided -> estimate them from pop
        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)
        
        # the number of objectives
        _, n_obj = F.shape
        pf_path = f'/home/mogami/bicriteria_pbemo/ref_point_dataset/{problem.name()}_d{n_obj}_n{self.n[n_obj - 2]}.csv'
        pf_npy = pf_path.replace('.csv', '.npy')
        if not os.path.exists(pf_npy):
            PF = np.loadtxt(pf_path, delimiter=',')
            np.save(pf_npy, PF)
        else:
            PF = np.load(pf_npy)
        true_nadir = PF.max(axis=0)
        PQ_size = len(F)
        # get the objective space ref point 
        ref_point = self._load_ref_point(n_obj, problem.name())
        # the final indices of surviving individuals
        survivor_list = [] # list(range(PQ_size))

        # normalized_F = (F - ideal) / (nadir - ideal)
        # normalized_ref_point = (ref_point - ideal) / (nadir - ideal)
        
        # 1 Find the pivot point
        if self.roi_type == "roi-c":
            nondom_id_list = find_non_dominated(F)
            dist_arr = np.full(len(F), np.inf)
            for i in nondom_id_list:
                dist_arr[i] = np.linalg.norm(F[i] - ref_point)

            pivot_id = np.argmin(dist_arr)
            pivot_point = F[pivot_id]
            dist_arr_pivot = np.zeros(len(F))
            for i, p in enumerate(F):
                dist_arr_pivot[i] = np.linalg.norm(p - pivot_point)
            
            data = []
            for i in range(n_obj):
                data.append(roi_radius)
                # data.append(true_nadir[i] * roi_radius)
            r_radius_elipse = np.array(data)
            diff = F - pivot_point
            val = np.sum((diff/r_radius_elipse)**2, axis = 1)
            sel_mask = val <= 1.0
        elif self.roi_type == "roi-p":
            less_eq    = np.all(F <= ref_point, axis=1)   # ≼ z 側（第3象限寄り）
            greater_eq = np.all(F >= ref_point, axis=1)   # ≽ z 側（第1象限寄り）

            # nondom_id_list = find_non_dominated(F)
            # nondom_mask = np.zeros(len(F), dtype=bool)
            # nondom_mask[nondom_id_list] = True

            # if np.any(less_eq):
            #     side_mask = less_eq        # 実行可能 z → ≺ z 側
            # else:
            #     side_mask = greater_eq     # 実行不可能 z → ≻ z 側
        
            # sel_mask = side_mask & nondom_mask
            # # less_eqに一つでもTrueが含まれていればless側のみ採用、なければgreater側のみ
            # if self.flag is None:#バージョン2
            #     self.flag = bool(np.any(less_eq))
            
            # if self.flag:
            #     sel_mask = less_eq
            # else:
            #     sel_mask = greater_eq
            sel_mask = np.logical_or(greater_eq, less_eq) #バージョン1
            # in_Q3 = np.all(F <= ref_point, axis=1)
            # if np.any(in_Q3):
            #     sel_mask = np.logical_or(greater_eq, less_eq)
            # else:
            #     sel_mask = ~in_Q3
            # if np.any(less_eq): #バージョン3
            #     sel_mask = less_eq        # 実行可能 z → ≺ z 側
            # else:
            #     sel_mask = greater_eq     # 実行不可能 z → ≻ z 側
        n_in = np.sum(sel_mask)
        # print(n_in)
        if n_in <= n_survive:
            sel_idx = np.where(sel_mask)[0]                  
            survivor_list.extend(sel_idx)

            # weight_vec = np.full(n_obj, 1.0/n_obj)
            # for i, p in enumerate(normalized_F):
            #     dist_arr_pivot[i] = asf1(p, ref_point, weight_vec)            

            # To avoid selecting duplicated IDs, assign the infinity value to IDs already selected.
            if self.roi_type == "roi-c":
                for i in survivor_list:
                    dist_arr_pivot[i] = np.inf
                n = n_survive - len(survivor_list) 
                sel_idx = np.argsort(dist_arr_pivot)[:n] 
                survivor_list.extend(sel_idx)
            elif self.roi_type == "roi-p":
                # dist_to_z = np.linalg.norm(F - ref_point, axis=1)

                # # R_out: less でも greater でもない個体（第2,4象限など）
                # mixed_mask = ~(less_eq | greater_eq)

                # # side 側の「非劣解ではない」個体（R^in に入っていない side 側）
                # side_dom_mask = side_mask & ~nondom_mask

                # # R_out（mixed）で、まだ選ばれていないもの
                # rout_mask = mixed_mask

                # # side 側とは反対の eq 側（less ↔ greater）で、まだ選ばれていないもの
                # opposite_side_mask = (~side_mask) & (less_eq | greater_eq)
                
                # remaining = n_survive - len(survivor_list)
                # # 1. R_out（less でも greater でもない個体）から補充
                # cand1 = np.where(rout_mask)[0]
                # if remaining > 0 and len(cand1) > 0:
                #     order = cand1[np.argsort(dist_to_z[cand1])]
                #     take = order[:remaining]
                #     survivor_list.extend(take.tolist())
                #     remaining -= len(take)
                # # 2. side 側の「非劣解ではない」個体から距離の近い順に補充
                # cand2 = np.where(side_dom_mask)[0]
                # if remaining > 0 and len(cand2) > 0:
                #     order = cand2[np.argsort(dist_to_z[cand2])]
                #     take = order[:remaining]
                #     survivor_list.extend(take.tolist())
                #     remaining -= len(take)
                
                # # 3. side 側とは反対の eq 側から補充
                # cand3 = np.where(opposite_side_mask)[0]
                # if remaining > 0 and len(cand3) > 0:
                #     order = cand3[np.argsort(dist_to_z[cand3])]
                #     take = order[:remaining]
                #     survivor_list.extend(take.tolist())
                #     remaining -= len(take)
                dist_arr = np.full(len(F), np.inf)
                unselected_list = np.array([i for i in range(len(F)) if i not in survivor_list], dtype=int)
                for i in unselected_list:
                    dist_arr[i] = np.linalg.norm(F[i] - ref_point)
                n = n_survive - len(survivor_list)
                sel_idx = np.argsort(dist_arr)[:n]
                survivor_list.extend(sel_idx)

            for fit, i in enumerate(survivor_list):        
                pop[i].set("fitness", n_survive - fit)

            survivors = pop[survivor_list]
            self.count_each.append(self.count)       
            return Population.create(*survivors)                
        else:
            trimmed_pop = pop[sel_mask]
            self.count += 1 
            self.count_each.append(self.count)  
            # print("=== trimmed_pop info ===")
            # print(f"type: {type(trimmed_pop)}, len: {len(trimmed_pop)}")
            # print("F values:")
            # print(trimmed_pop.get("F"))
            # print("=========================") 
            return self.ibea_env(problem, trimmed_pop, n_survive=n_survive)

    
# ---------------------------------------------------------------------------------------------------------
# Binary Tournament
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        a_cv, b_cv = pop[a].CV[0], pop[b].CV[0]
        fitness_a = pop[a].get("fitness")
        fitness_b = pop[b].get("fitness")
    
        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)
        # both solutions are feasible
        else:
            if fitness_a > fitness_b:
                S[i] = a
            else:
                S[i] = b

    return S[:, None].astype(int, copy=False)

# ---------------------------------------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------------------------------------

class BIBEA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=RandomSelection(),
                 #selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(),
                 mutation=PM(),
                 survival=None,
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 normalize=True,
                 output=MultiObjectiveOutput(),
                 roi_type="roi-c",
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """
        if survival is None:
            survival = FitnessAssignment(roi_type=roi_type)
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)

        self.normalize = normalize

    # def _advance(self, infills=None, **kwargs):
    #     # merge the offsprings with the current population
    #     if infills is not None:
    #         pop = Population.merge(self.pop, infills)
    #     else:
    #         pop = self.pop

    #     self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self,
    #                                 ideal=None, nadir=None, ref_point=self.ref_point **kwargs)


parse_doc_string(BIBEA.__init__)
