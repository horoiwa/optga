import numpy as np

class Spawner:

    def __init__(self, config):
        self.config = config

        self.n_cols = len(self.config.feature_names)

    def spawn(self, n):
        population = np.zeros((n, self.n_cols))
        for i in range(self.n_cols):
            population[:, i] = np.random.uniform(self.config.lowerlim[i],
                                                 self.config.upperlim[i], n)
        return population
