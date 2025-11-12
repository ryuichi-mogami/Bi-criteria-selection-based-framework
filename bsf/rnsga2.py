import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath("../pymoo"))
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import get_extreme_points_c

# =========================================================================================================
# Implementation
# =========================================================================================================


class RNSGA2(NSGA2):

    def __init__(self,
                 ref_points = None,
                 epsilon=0.001,
                 normalization="front",
                 weights=None,
                 extreme_points_as_reference_points=False,
                 **kwargs):
        """

        Parameters
        ----------

        ref_points : {ref_points}
        epsilon : float
        weights : np.array
        normalization : {{'no', 'front', 'ever'}}
        extreme_points_as_reference_points : bool

        """
        # self.ref_points = ref_points
        self.epsilon = epsilon
        self.weights = weights
        self.normalization = normalization
        self.selection = RandomSelection()

        super().__init__(**kwargs)

        self.survival = RankAndModifiedCrowdingSurvival(ref_points, epsilon, weights, normalization,
                                                        extreme_points_as_reference_points)


class RankAndModifiedCrowdingSurvival(Survival):

    def __init__(self, ref_points,
                 epsilon,
                 weights,
                 normalization,
                 extreme_points_as_reference_points
                 ) -> None:

        super().__init__(True)
        self.n_obj = None
        self.ref_points = None
        self._ref_point_cache = {}
        self.epsilon = epsilon
        self.extreme_points_as_reference_points = extreme_points_as_reference_points

        self.weights = None
        # if self.weights is None:
        #     self.weights = np.full(self.n_obj, 1 / self.n_obj)

        self.normalization = normalization
        # self.ideal_point = np.full(self.n_obj, np.inf)
        # self.nadir_point = np.full(self.n_obj, -np.inf)
        self.ideal_point = None
        self.nadir_point = None

    def _load_ref_point(self, n_obj: int, problem_name: str) -> np.ndarray:
        key = (n_obj, problem_name) 
        if key in self._ref_point_cache:
            return self._ref_point_cache[key]
        ref_file = f"/home/mogami/bicriteria_pbemo/ref_point_data/roi-c/m{n_obj}_{problem_name}_type1.csv" 
        ref_point = np.loadtxt(ref_file, delimiter=",", dtype=float) 
        self._ref_point_cache[key] = ref_point 
        return ref_point

    def _do(self, problem, pop, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F")
        if self.n_obj is None:
            self.n_obj = F.shape[1]
            self.weights = np.full(self.n_obj, 1 / self.n_obj)
            self.ideal_point = np.full(self.n_obj, np.inf)
            self.nadir_point = np.full(self.n_obj, -np.inf)
        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F)
        if self.ref_points is None:
            self.ref_points = self._load_ref_point(self.n_obj, problem.name())
        if self.normalization == "ever":
            # find or usually update the new ideal point - from feasible solutions
            self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
            self.nadir_point = np.max(np.vstack((self.nadir_point, F)), axis=0)

        elif self.normalization == "front":
            front = fronts[0]
            if len(front) > 1:
                self.ideal_point = np.min(F[front], axis=0)
                self.nadir_point = np.max(F[front], axis=0)

        elif self.normalization == "no":
            self.ideal_point = np.zeros(self.n_obj)
            self.nadir_point = np.ones(self.n_obj)

        if self.extreme_points_as_reference_points:
            self.ref_points = np.row_stack([self.ref_points, get_extreme_points_c(F, self.ideal_point)])

        # calculate the distance matrix from ever solution to all reference point
        dist_to_ref_points = calc_norm_pref_distance(F, self.ref_points, self.weights, self.ideal_point,
                                                     self.nadir_point)

        for k, front in enumerate(fronts):

            # save rank attributes to the individuals - rank = front here
            pop[front].set("rank", np.full(len(front), k))

            # number of individuals remaining
            n_remaining = n_survive - len(survivors)

            # the ranking of each point regarding each reference point (two times argsort is necessary)
            rank_by_distance = np.argsort(np.argsort(dist_to_ref_points[front], axis=0), axis=0)

            # the reference point where the best ranking is coming from
            ref_point_of_best_rank = np.argmin(rank_by_distance, axis=1)

            # the actual ranking which is used as crowding
            ranking = rank_by_distance[np.arange(len(front)), ref_point_of_best_rank]

            if len(front) <= n_remaining:

                # we can simply copy the crowding to ranking. not epsilon selection here
                crowding = ranking
                I = np.arange(len(front))

            else:

                # Distance from solution to every other solution and set distance to itself to infinity
                dist_to_others = calc_norm_pref_distance(F[front], F[front], self.weights, self.ideal_point,
                                                         self.nadir_point)
                np.fill_diagonal(dist_to_others, np.inf)

                # the crowding that will be used for selection
                crowding = np.full(len(front), np.nan)

                # solutions which are not already selected - for
                not_selected = np.argsort(ranking)

                # until we have saved a crowding for each solution
                while len(not_selected) > 0:

                    # select the closest solution
                    idx = not_selected[0]

                    # set crowding for that individual
                    crowding[idx] = ranking[idx]

                    # need to remove myself from not-selected array
                    to_remove = [idx]

                    # Group of close solutions
                    dist = dist_to_others[idx][not_selected]
                    group = not_selected[np.where(dist < self.epsilon)[0]]

                    # if there exists solution with a distance less than epsilon
                    if len(group):
                        # discourage them by giving them a high crowding
                        crowding[group] = ranking[group] + np.round(len(front) / 2)

                        # remove group from not_selected array
                        to_remove.extend(group)

                    not_selected = np.array([i for i in not_selected if i not in to_remove])

                # now sort by the crowding (actually modified rank) ascending and let the best survive
                I = np.argsort(crowding)[:n_remaining]

            # set the crowding to all individuals
            pop[front].set("crowding", crowding)

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        # inverse of crowding because nsga2 does maximize it (then tournament selection can stay the same)
        pop.set("crowding", -pop.get("crowding"))

        return pop[survivors]


def calc_norm_pref_distance(A, B, weights, ideal, nadir,zero_safe=1e-12,return_squared=False):
    # D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    # N = ((D / (nadir - ideal)) ** 2) * weights
    # N = np.sqrt(np.sum(N, axis=1) * len(weights))
    # return np.reshape(N, (A.shape[0], B.shape[0]))
    A = np.asarray(A, dtype=float)
    B = np.atleast_2d(np.asarray(B, dtype=float))
    w = np.asarray(weights, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)
    n_a, m = A.shape

    denom = nadir - ideal
    denom[np.abs(denom) < zero_safe] = 1.0
    A_norm = (A - ideal) / denom
    B_norm = (B - ideal) / denom

    D = A_norm[:, None, :] - B_norm[None, :, :] # (n_a, n_b, m)

    dist2 = np.sum((D**2) * w[None, None, :], axis=2) * m # (n_a, n_b)

    if return_squared:
        return dist2
    return np.sqrt(dist2)
parse_doc_string(RNSGA2.__init__)
