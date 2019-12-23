import numpy as np
from numba import jit, f8, i8


class Constrainter:

    def __init__(self, config):

        self.config = config

        self.discrete_constraints = self.config.discrete_constraints

    def constraint(self, population):
        if not isinstance(population, np.ndarray):
            raise TypeError("populaiton must be ndarray")
        else:
            population = population.astype(np.float64)

        population = self.add_onehot_constraint(population)
        population = self.add_discrete_constraint(population)
        return population

    def add_discrete_constraint(self, population):
        for fname, constraints in self.discrete_constraints.items():
            idx = self.config.fname_to_idx(fname)
            constraints = np.array(constraints).astype(np.float64)
            population[:, idx] = _discrete(population[:, idx], constraints)

        return population

    def add_onehot_constraint(self, population):
        return population


@jit(f8[:](f8[:], f8[:]), nopython=True)
def _discrete(population, constraints):
    for i in range(population.shape[0]):
        idx = (np.abs(constraints - population[i])).argmin()
        population[i] = constraints[idx]

    return population
