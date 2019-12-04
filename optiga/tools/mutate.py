from abc import abstractmethod


class BaseMutate:

    def __init__(self):
        pass

    @abstractmethod
    def mutate(self, population):
        raise NotImplementedError()


class MutateUniform(BaseMutate):

    def mutate(self, population):
        return population
