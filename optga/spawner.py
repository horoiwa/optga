import numpy as np
import pandas as pd

from optga.tools.sobol import i4_sobol_generate

SKIP = 2


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
            raise ValueError(f"No such option: {mode}")

    def spawn_uniform(self, n):
        population = np.zeros((n, self.n_cols))
        for i in range(self.n_cols):
            population[:, i] = np.random.uniform(self.config.lowerlim[i],
                                                 self.config.upperlim[i], n)
        population = pd.DataFrame(population,
                                  columns=self.config.feature_names)
        return population

    def spawn_sobol(self, n):
        """Spawn population based on sobol sequence
        !!Caution : num of dimension must be lower than 40

        Parameters
        ----------
        n : int
            number of samples

        Returns
        -------
        pd.DataFrame
            spawn samples
        """
        if self.n_cols > 40:
            raise Exception("Sobol spawner is not available"
                            "when dimension greater than 40")
        global SKIP

        population = i4_sobol_generate(self.n_cols, n, skip=SKIP)
        SKIP += n
        for i in range(self.n_cols):
            diff = self.config.upperlim[i] - self.config.lowerlim[i]
            population[:, i] = population[:, i] * diff
            population[:, i] = population[:, i] + self.config.lowerlim[i]

        population = pd.DataFrame(population,
                                  columns=self.config.feature_names)
        return population


if __name__ == "__main__":
    from optga.optimizer import Optimizer

    initial_population = np.zeros((20, 5))
    for i in range(initial_population.shape[1]):
        initial_population[:, i] = np.random.uniform(
            i, i*10, size=initial_population.shape[0])

    initial_population = pd.DataFrame(initial_population)
    optimizer = Optimizer(sample_data=initial_population)
    pop = optimizer.spawn_population(10, mode="sobol")
    print(pop)
