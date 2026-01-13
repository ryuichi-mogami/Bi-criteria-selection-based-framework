import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath("../pymoo"))
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import vectorized_cdist
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.core.population import Population
from pymoo.operators.selection.rnd import RandomSelection
# ---------------------------------------------------------------------------------------------------------
# Environmental Survival (in the original paper it is referred to as archiving)
# ---------------------------------------------------------------------------------------------------------


class SPEA2Survival(Survival):

    def __init__(self, alpha = 0, normalize=False, filter_infeasible=True):
        super().__init__(filter_infeasible)

        # whether the survival should considered normalized distance or just raw
        self.normalize = normalize
        self.alpha = alpha
        # an object keeping track of normalization points
        self.norm = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        # 元の目的値 F を取得
        F_raw = pop.get("F").astype(float, copy=False)

        # f̄(x) = 平均値 を個体ごとに計算 (shape: (N, 1))
        f_mean = F_raw.mean(axis=1, keepdims=True)

        # Ishibuchi 論文の補正式で F を補正
        F = (1.0 - self.alpha) * F_raw + self.alpha * f_mean

        # the domination matrix to see the relation for each solution to another
        M = Dominator().calc_domination_matrix(F)

        # the number of solutions each individual dominates
        S = (M == 1).sum(axis=0)

        # the raw fitness of each solution - strength of its dominators
        R = ((M == -1) * S).sum(axis=1)

        # determine what k-th nearest neighbor to consider
        k = int(np.sqrt(len(pop)))
        if k >= len(pop):
            k = len(pop) - 1

        # if normalization is enabled keep track of ideal and nadir
        if self.normalize:

            # initialize the first time and then always update the boundary points
            if self.norm is None:
                self.norm = HyperplaneNormalization(F.shape[1])
            self.norm.update(F)

            ideal, nadir = self.norm.ideal_point, self.norm.nadir_point

            _F = (F - ideal) / (nadir - ideal)
            dists = vectorized_cdist(_F, _F, fill_diag_with_inf=True)

        # if no normalize is required simply use the F values from the population
        else:
            dists = vectorized_cdist(F, F, fill_diag_with_inf=True)

        # the distances sorted for each individual
        sdists = np.sort(dists, axis=1)

        # inverse distance as part of the fitness
        D = 1 / (sdists[:, k] + 2)

        # the actual fitness value used to determine who survives
        SPEA_F = R + D

        # set all the attributes to the population
        pop.set(SPEA_F=SPEA_F, SPEA_R=R, SPEA_D=D)

        # get all the non-dominated solutions
        survivors = list(np.where(np.all(M >= 0, axis=1))[0])

        # if we normalize give boundary points most importance - give the boundary points in the nds set the lowest fit.
        if self.normalize:
            I = vectorized_cdist(self.norm.extreme_points, F[survivors]).argmin(axis=1)
            pop[survivors][I].set("SPEA_F", -1.0)

        # identify the remaining individuals to choose from
        H = set(survivors)
        rem = np.array([k for k in range(len(pop)) if k not in H])

        # if not enough solutions, will up by F
        if len(survivors) < n_survive:

            # sort them by the fitness values (lower is better) and append them
            rem_by_F = rem[SPEA_F[rem].argsort()]
            survivors.extend(rem_by_F[:n_survive - len(survivors)])

        # if too many, delete based on distances
        elif len(survivors) > n_survive:

            # remove one individual per loop, until we hit n_survive
            while len(survivors) > n_survive:
                i = dists[survivors][:, survivors].min(axis=1).argmin()
                survivors = [survivors[j] for j in range(len(survivors)) if j != i]

        return pop[survivors]


class FitnessAssignment(Survival):

    def __init__(self, roi_type, roi_id, alpha) -> None:
        super().__init__(filter_infeasible=True)
        self.roi_type = roi_type
        self.roi_id = roi_id
        self._ref_point_cache = {}
        self.count = 0
        self.count_each = []
        self.flag = None
        self.alpha = alpha 
        self.n = [50000, 50086, 102340, 455126, 3162510]

    def _load_ref_point(self, n_obj: int, problem_name: str) -> np.ndarray:
        key = (n_obj, problem_name) 
        if key in self._ref_point_cache:
            return self._ref_point_cache[key]
        ref_file = f"/home/mogami/bicriteria_pbemo/ref_point_data/{self.roi_type}/m{n_obj}_{problem_name}_type{self.roi_id}.csv"
        print(f"Loading reference point from: {ref_file}")
        ref_point = np.loadtxt(ref_file, delimiter=",", dtype=float) 
        self._ref_point_cache[key] = ref_point 
        return ref_point
    
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
            sel_mask = np.logical_or(greater_eq, less_eq) #バージョン1
        n_in = np.sum(sel_mask)

        if n_in <= n_survive:
            sel_idx = np.where(sel_mask)[0]                  
            survivor_list.extend(sel_idx)
       
            if self.roi_type == "roi-c":
                for i in survivor_list:
                    dist_arr_pivot[i] = np.inf
                n = n_survive - len(survivor_list) 
                sel_idx = np.argsort(dist_arr_pivot)[:n] 
                survivor_list.extend(sel_idx)
            elif self.roi_type == "roi-p":

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
            return SPEA2Survival(self.alpha).do(problem=problem, pop=trimmed_pop, n_survive=n_survive)
        

# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection
# ---------------------------------------------------------------------------------------------------------


def spea_binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f, = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:
            S[i] = compare(a, pop[a].get("SPEA_F"), b, pop[b].get("SPEA_F"), method='smaller_is_better',
                           return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------------------------------------


class BSPEA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=RandomSelection(),
                #  selection=TournamentSelection(spea_binary_tournament),
                 crossover=SBX(),
                 mutation=PM(),
                 survival=None,   
                #  survival=SPEA2Survival(normalize=True),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 roi_type="roi-c",
                 roi_id=1,
                 alpha = 0,
                 **kwargs):
        if survival is None:
            survival = FitnessAssignment(roi_type=roi_type, roi_id=roi_id, alpha=alpha)
        """

        SPEA2 - Strength Pareto EA 2

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
        self.termination = DefaultMultiObjectiveTermination()


parse_doc_string(BSPEA2.__init__)

