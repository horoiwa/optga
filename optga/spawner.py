import numpy as np
import pandas as pd


class Spawner:
    """Spawner spawns population as pd.DataFrame
    """

    def __init__(self, config):
        self.config = config

        self.n_cols = len(self.config.feature_names)

    def spawn(self, n, mode="uniform"):
        if mode == "uniform":
            return self.spawn_uniform(n)
        elif mode == "sobol":
            return self.spawn_sobol(n)
        else:
            raise ValueError(f"No such mode: {mode}")

    def spawn_uniform(self, n):
        population = np.zeros((n, self.n_cols))
        for i in range(self.n_cols):
            population[:, i] = np.random.uniform(self.config.lowerlim[i],
                                                 self.config.upperlim[i], n)
        population = pd.DataFrame(population,
                                  columns=self.config.feature_names)
        return population

    def spawn_sobol(self, n):
        population = np.zeros((n, self.n_cols))


if __name__ == "__main__":
    from optga.optimizer import Optimizer

    initial_population = np.zeros((20, 5))
    for i in range(initial_population.shape[1]):
        initial_population[:, i] = np.random.uniform(
            i, i*10, size=initial_population.shape[0])

    initial_population = pd.DataFrame(initial_population)
    optimizer = Optimizer(sample_data=initial_population)
    pop = optimizer.spawn_population(10, mode="uniform")
    print(pop)
