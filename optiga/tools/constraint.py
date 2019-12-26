import numpy as np
import pandas as pd
from numba import jit, f8, i8


class Constrainter:

    def __init__(self, config):

        self.config = config

        self.discrete_constraints = self.config.discrete_constraints

        self.onehot_groups = self.config.onehot_groups

        self.onehot_constraints = self.config.onehot_constraints

        self.user_constraint_func = None

    def constraint(self, population):
        if not isinstance(population, np.ndarray):
            raise TypeError("populaiton must be ndarray")
        else:
            population = population.astype(np.float64)

        population = self.add_onehot_constraint(population)
        population = self.add_discrete_constraint(population)

        if self.user_constraint_func is not None:
            population = pd.DataFrame(population,
                                      columns=self.config.feature_names)
            population = self.user_constraint_func(population)
            population = population.values

        return population

    def add_discrete_constraint(self, population):
        for fname, constraints in self.discrete_constraints.items():
            idx = self.config.fname_to_idx(fname)
            constraints = np.array(constraints).astype(np.float64)
            population[:, idx] = _discrete(population[:, idx], constraints)

        return population

    def add_onehot_constraint(self, population):
        for uid, group in self.onehot_groups.items():
            columns = [self.config.fname_to_idx(fname) for fname in group]
            constraints = np.array(
                self.onehot_constraints[uid]).astype(np.float64)
            population[:, columns] = _onehot(population[:, columns],
                                             constraints)

        return population


@jit(f8[:](f8[:], f8[:]), nopython=True)
def _discrete(arr, constraints):
    for i in range(arr.shape[0]):
        idx = (np.abs(constraints - arr[i])).argmin()
        arr[i] = constraints[idx]

    return arr


@jit(f8[:, :](f8[:, :], f8[:]), nopython=True)
def _onehot(arr, valuerange):
    #: if valuerange is [1, 1], equals to np.ones(arr.shape[0])
    lowerlim = valuerange[0]
    upperlim = valuerange[1]
    constants = np.random.uniform(lowerlim,
                                  upperlim, 
                                  arr.shape[0])
    #: fill all zero rows (invalid rows) by constant
    arr[arr.sum(1) == 0] = 1

    #: create onehot array
    onehot_arr = np.zeros(arr.shape)
    nonzero_elements = (arr != 0)
    columns = np.arange(arr.shape[1])
    for i in range(arr.shape[0]):
        selected_col = np.random.choice(columns[nonzero_elements[i]])
        selected_val = arr[i, selected_col]
        if selected_val <= upperlim and selected_val >= lowerlim:
            onehot_arr[i, selected_col] = selected_val
        else:
            onehot_arr[i, selected_col] = constants[i]

    return onehot_arr
