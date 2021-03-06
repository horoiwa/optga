import numpy as np
import pandas as pd
from numba import jit, f8, i8


class Constrainter:

    def __init__(self, config):

        self.config = config

        self.discrete_constraints = self.config.discrete_constraints

        self.onehot_groups = self.config.onehot_groups

        self.onehot_constraints = self.config.onehot_constraints

        self.sumtotal_groups = self.config.sumtotal_groups

        self.sumtotal_constraints = self.config.sumtotal_constraints

        self.user_constraint_func = None

    def constraint(self, population):
        if not isinstance(population, np.ndarray):
            raise TypeError("populaiton must be ndarray")
        else:
            population = population.astype(np.float64)

        if self.onehot_constraints is not None:
            population = self.add_onehot_constraint(population)
        if self.sumtotal_constraints is not None:
            population = self.add_sumtotal_constraint(population)
        if self.discrete_constraints is not None:
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
                self.onehot_constraints[uid][:2]).astype(np.float64)
            n = np.int64(self.onehot_constraints[uid][2])

            population[:, columns] = _onehot(population[:, columns],
                                             constraints, n)

        return population

    def add_sumtotal_constraint(self, population):
        for uid, group in self.sumtotal_groups.items():
            columns = [self.config.fname_to_idx(fname) for fname in group]
            constraints = np.array(
                self.sumtotal_constraints[uid]).astype(np.float64)
            population[:, columns] = _sumtotal(population[:, columns],
                                               constraints)
        return population


@jit(f8[:](f8[:], f8[:]), nopython=True)
def _discrete(arr, constraints):
    for i in range(arr.shape[0]):
        idx = (np.abs(constraints - arr[i])).argmin()
        arr[i] = constraints[idx]

    return arr


@jit(f8[:, :](f8[:, :], f8[:], i8), nopython=True)
def _onehot(arr, valuerange, n):
    #: if valuerange is [1, 1], equals to np.ones(arr.shape[0])
    lowerlim = valuerange[0]
    upperlim = valuerange[1]
    constants = np.random.uniform(lowerlim,
                                  upperlim,
                                  arr.shape[0])
    #: fill all zero rows (invalid rows) by constant
    arr[(arr != 0).sum(1) < n] = 1

    #: create onehot array
    onehot_arr = np.zeros(arr.shape)
    nonzero_elements = (arr != 0)
    columns = np.arange(arr.shape[1])

    for i in range(arr.shape[0]):
        selected_cols = np.random.choice(columns[nonzero_elements[i]],
                                         n, replace=False)
        selected_vals = arr[i][selected_cols]

        for j in range(n):
            selected_col = selected_cols[j]
            selected_val = selected_vals[j]

            if (selected_val <= upperlim) and (selected_val >= lowerlim):
                onehot_arr[i][selected_col] = selected_val
            else:
                onehot_arr[i][selected_col] = constants[i]

    return onehot_arr


@jit(f8[:, :](f8[:, :], f8[:]), nopython=True)
def _sumtotal(arr, valuerange):
    #: test onehot constraints
    lowerlim = valuerange[0]
    upperlim = valuerange[1]

    sumtotals = arr.sum(1)
    for i in range(arr.shape[0]):
        sumtotal = sumtotals[i]
        if lowerlim <= sumtotal and upperlim >= sumtotal:
            continue
        else:
            total_alt = valuerange[(np.abs(valuerange - sumtotal)).argmin()]
            arr[i] = (arr[i] / sumtotal) * total_alt

    return arr
