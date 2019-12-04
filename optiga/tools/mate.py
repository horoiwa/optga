from abc import ABCMeta, abstractmethod
import copy

import numpy as np


class BaseMate:

    def __init__(self, birth_rate):
        self.birth_rate = birth_rate

    @abstractmethod
    def mate(self, population):
        raise NotImplementedError()


class MateCxTwoPoints(BaseMate):

    def mate(self, population):
        population_clone = copy.deepcopy(population)
        population_clone = np.random.permutation(
            np.vstack([population_clone for _ in range(self.birth_rate)]))
        population = np.vstack([population for _ in range(self.birth_rate)])

        cxpoints = [(min(a, b), max(a, b))
                    for a, b in zip(
                        np.random.randint(0, population.shape[1],
                                          population.shape[0]),
                        np.random.randint(0, population.shape[1],
                                          population.shape[0]))]

        mask = np.ones(population.shape)
        for i in range(mask.shape[0]):
            #: この処理はint32なのでそこそこ高速
            mask[i, cxpoints[i][0]:cxpoints[i][1]] = 0

        mask = mask.astype(bool)
        population *= mask
        population_clone *= ~mask

        offspring = population + population_clone
        return offspring


if __name__ == "__main__":
    arr1 = np.arange(1, 13).reshape(3, 4)
    arr2 = np.zeros((3, 4))
