from abc import ABCMeta, abstractmethod
import copy

import numpy as np
from numba import jit, f8, i8


class BaseMate:

    def __init__(self, config):
        self.config = config

        self.birth_rate = self.config.birth_rate

        self.groups = self.config.group_variables_indices

    @abstractmethod
    def mate(self, population):
        raise NotImplementedError()


class MateCxTwoPoints(BaseMate):

    def mate(self, population):
        population_clone = copy.deepcopy(population)
        population_clone = np.random.permutation(
            np.vstack([population_clone for _ in range(self.birth_rate)]))
        population = np.vstack([population for _ in range(self.birth_rate)])

        cxpoints = np.array([(min(a, b), max(a, b)) if a != b else (a-1, b+1)
                             for a, b in zip(
                                 np.random.randint(1, population.shape[1]-1,
                                                   population.shape[0]),
                                 np.random.randint(1, population.shape[1]-1,
                                                   population.shape[0]))])

        mask = apply_cxpoints(np.ones(population.shape).astype(np.int64),
                              cxpoints.astype(np.int64))

        #: group variables must be exchanged together
        for group in self.groups:
            group = np.array(group).astype(np.int64)
            mask[:, group] = adjust_group(mask[:, group])

        mask = mask.astype(bool)
        population *= mask
        population_clone *= ~mask

        children = population + population_clone
        return children


@jit(i8[:, :](i8[:, :], i8[:, :]), nopython=True)
def apply_cxpoints(mask, cxpoints):
    for i in range(mask.shape[0]):
        mask[i, cxpoints[i, 0]:cxpoints[i, 1]] = 0
    return mask


@jit(i8[:, :](i8[:, :]), nopython=True)
def adjust_group(arr, ):
    for i in range(arr.shape[0]):
        arr[i, :] = np.round(np.mean(arr[i, :]))
    return arr


if __name__ == "__main__":
    arr1 = np.arange(1, 13).reshape(3, 4)
    arr2 = np.zeros((3, 4))
