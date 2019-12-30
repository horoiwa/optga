from abc import ABCMeta, abstractmethod
import copy

import numpy as np
from numba import jit, f8, i8


class BaseMate:

    def __init__(self, config):
        self.config = config

        self.birth_rate = self.config.birth_rate

        self.group_indices = self.config.group_variables_indices

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

        mask = np.ones(population.shape)
        for i in range(mask.shape[0]):
            #: この処理はint32なのでそこそこ高速
            mask[i, cxpoints[i, 0]:cxpoints[i, 1]] = 0

        mask = mask.astype(bool)
        population *= mask
        population_clone *= ~mask

        children = population + population_clone
        return children


def get_mate_mask(self, mask, cxpoints):
    mask[i, cxpoints[i, 0]:cxpoints[i, 1]] = 0
    return mask

if __name__ == "__main__":
    arr1 = np.arange(1, 13).reshape(3, 4)
    arr2 = np.zeros((3, 4))
