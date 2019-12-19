import numpy as np
from numba import jit, f8, i8


def NSGA2(population, fitness, k):
    fitness = fitness.astype(np.float64)
    dominate_table = get_dominatetable(fitness)
    fronts = get_fronts(dominate_table, k)
    sorted_fronts = [sort_by_CrowdingDist(fitness, front) for front in fronts]

    indices = []
    for front in sorted_fronts:
        indices += front
    next_population = population[indices[:k]]

    return next_population


def get_paretofront(population, fitness):
    fitness = fitness.astype(np.float64)
    dominate_table = get_dominatetable(fitness)
    front = get_fronts(dominate_table, k=1)
    return population[front[0]]


@jit(i8[:, :](f8[:, :]), nopython=True)
def get_dominatetable(fitness):
    n_fits = fitness.shape[0]

    dominate_table = np.zeros((n_fits, n_fits), dtype=np.int64)
    for i in range(n_fits):
        for j in range(i+1, n_fits):
            fit_i = fitness[i]
            fit_j = fitness[j]
            if np.all(fit_i == fit_j):
                continue
            elif np.all(fit_i >= fit_j):
                dominate_table[i, j] = 0
                dominate_table[j, i] = 1
            elif np.all(fit_j >= fit_i):
                dominate_table[i, j] = 1
                dominate_table[j, i] = 0

    return dominate_table


def get_fronts(dominate_table, k):
    #: (2) ランク順にソート
    K = min(k, dominate_table.shape[0])
    fronts = []
    selected_indices = []

    while len(selected_indices) < K:
        rank = dominate_table.sum(1)

        #: すでに選ばれているインデックスは除外する
        current_front = list(np.setdiff1d(np.where(rank == 0)[0],
                             selected_indices))
        fronts.append(current_front)
        selected_indices += current_front

        #: 支配の解除
        dominate_table[:, current_front] = 0

    return fronts


def sort_by_CrowdingDist(fitness, front):
    distances = np.zeros(len(front))

    fitness_ = fitness[front]
    sorted_indices = np.argsort(fitness_, axis=0)
    norms = fitness_.max(0) - fitness_.min(0)

    for i in range(1, sorted_indices.shape[0]):
        for j in range(sorted_indices.shape[1]):
            if i == 0:
                distances[sorted_indices[i, j]] = np.inf
            elif i == (sorted_indices.shape[0]-1):
                distances[sorted_indices[i, j]] = np.inf
            elif norms[j] == 0:
                continue
            else:
                prev_val = fitness_[sorted_indices[i-1, j], j]
                next_val = fitness_[sorted_indices[i+1, j], j]
                distances[sorted_indices[i, j]] += (next_val - prev_val) / norms[j]

    sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
    sorted_front = [idx for idx, dist in sorted_front]
    return sorted_front
