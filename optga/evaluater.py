import numpy as np


class Evaluator:

    def __init__(self, config, models):

        self.config = config

        self.objective_names = (list(self.config.objectives)
                                if self.config.objectives else [])

        self.models = models

    def evaluate(self, population):
        fitness = np.zeros((population.shape[0], len(self.objective_names)))
        for i, oname in enumerate(self.objective_names):
            fitness[:, i] = self.models[oname](population)
        return fitness
