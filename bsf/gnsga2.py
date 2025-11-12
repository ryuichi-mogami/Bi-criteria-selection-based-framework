import numpy as np
import warnings
import sys
import os
sys.path.insert(0, os.path.abspath("../pymoo"))
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible
from pymoo.operators.selection.rnd import RandomSelection

# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------
def _is_one_sided(F, z):
    less = np.any(F < z)
    more = np.any(F > z)
    return (less ^ more)

def _g_transform(F, z, penalty=5.0):
    return F if _is_one_sided(F, z) else (F + penalty)

def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(RankAndCrowding):
    
    def __init__(self, nds=None, crowding_func="cd"):
        super().__init__(nds, crowding_func)
        self._ref_point_cache = {}

    def _load_ref_point(self, n_obj: int, problem_name: str) -> np.ndarray:
        key = (n_obj, problem_name) 
        if key in self._ref_point_cache:
            return self._ref_point_cache[key]
        ref_file = f"/home/mogami/bicriteria_pbemo/ref_point_data/roi-p/m{n_obj}_{problem_name}_type1.csv" 
        ref_point = np.loadtxt(ref_file, delimiter=",", dtype=float) 
        self._ref_point_cache[key] = ref_point 
        return ref_point

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        _, n_obj = F.shape
        ref_point = self._load_ref_point(n_obj, problem.name())
        F_orig = pop.get("F")

        less = np.all(F <= ref_point, axis=1) & np.any(F < ref_point, axis=1)
        more = np.all(F >= ref_point, axis=1) & np.any(F > ref_point, axis=1)
        one_side = np.logical_or(less, more)
        
        Fp = F.copy()
        penalty = 5
        Fp[~one_side] = Fp[~one_side] + penalty

        pop.set("F", Fp)
        survivors = super()._do(problem, pop, n_survive=n_survive, **kwargs)
        pop.set("F", F_orig)
        return survivors
# =========================================================================================================
# Implementation
# =========================================================================================================


class gNSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=RandomSelection(),
                #  selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                #  survival=RankAndCrowding(),
                 survival=RankAndCrowdingSurvival(),
                 output=MultiObjectiveOutput(),
                 n_obj=None,
                 problem_name=None,
                 g_penalty=5.0,
                 **kwargs):
        self.g_penalty = float(g_penalty)
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


parse_doc_string(gNSGA2.__init__)
