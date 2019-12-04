from abc import abstractmethod


class BaseMutate:

    def __init__(self, config):

        self.config = config

        self.mutpb = self.config.mutpb

        self.indpb = self.config.indpb

    @abstractmethod
    def mutate(self, population):
        raise NotImplementedError()


class MutateUniform(BaseMutate):

    def mutate(self, population):
        return population
