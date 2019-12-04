from optiga.tools.mate import MateCxTwoPoints
from optiga.tools.mutate import MutateUniform
from optiga.tools.select import SelectNSGA2


class EvolutionStrategy:

    def __init__(self, config):
        self.config = config

        self.mater = globals()[self.config.mate](self.config.birth_rate)

        self.mutater = globals()[self.config.mutate]()

    def mate(self, population):

        offspring = self.mater.mate(population)

        return offspring

    def mutate(self, population):

        offspring = self.mutater.mutate(population)

        return offspring

    def select(self, population, size):

        return population
