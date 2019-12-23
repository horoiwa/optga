from optiga.tools.mate import MateCxTwoPoints
from optiga.tools.mutate import MutateUniform
from optiga.tools.select import SelectNSGA2
from optiga.tools.constraint import Constrainter


class EvolutionStrategy:

    def __init__(self, config):
        self.config = config

        self.mater = globals()[self.config.mate](self.config)

        self.mutater = globals()[self.config.mutate](self.config)

        self.selecter = globals()[self.config.select]

        self.constrainter = Constrainter(self.config)

    def mate(self, population):

        offspring = self.mater.mate(population)

        return offspring

    def mutate(self, population):

        offspring = self.mutater.mutate(population)

        return offspring

    def select(self, population, fitness, size):

        selected_population = self.selecter(population, fitness, size)

        return selected_population

    def constraint(self, population):
        population = self.constrainter.constraint(population)
        return population
