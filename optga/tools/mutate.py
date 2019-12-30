from abc import abstractmethod

import numpy as np


class BaseMutate:

    def __init__(self, config):

        self.config = config

        self.upperlim = self.config.upperlim

        self.lowerlim = self.config.lowerlim

        self.mutpb = self.config.mutpb

        self.indpb = self.config.indpb

    @abstractmethod
    def mutate(self, population):
        raise NotImplementedError()


class MutateUniform(BaseMutate):

    def mutate(self, population):
        mask = [bool(np.random.binomial(1, self.mutpb))
                for i in range(population.shape[0])]

        population[mask] = self.mutate_(population[mask])
        return population

    def mutate_(self, population):
        mutation = np.zeros(population.shape)
        for i in range(mutation.shape[1]):
            mutation[:, i] = np.random.uniform(self.upperlim[i],
                                               self.lowerlim[i],
                                               mutation.shape[0])

        #: indpbで0のマスク
        mask = np.random.binomial(1, self.indpb, population.shape).astype(bool)
        mutation *= mask
        population *= ~mask

        return population + mutation
