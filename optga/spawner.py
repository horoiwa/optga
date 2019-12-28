import numpy as np
import pandas as pd


class Spawner:
    """Spawner spawns population as pd.DataFrame
    """

    def __init__(self, config):
        self.config = config

        self.n_cols = len(self.config.feature_names)

    def spawn(self, n):
        population = np.zeros((n, self.n_cols))
        for i in range(self.n_cols):
            population[:, i] = np.random.uniform(self.config.lowerlim[i],
                                                 self.config.upperlim[i], n)
        population = pd.DataFrame(population,
                                  columns=self.config.feature_names)
        return population
