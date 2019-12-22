import random

import numpy as np

from optiga.optimizer import Optimizer
from optiga.support import (get_linear_model, get_onemax_model,
                            get_onemax_samples)


class TestSpawner:

    def setup_method(self):
        pop_size = 500
        length = 10

        init_population = get_onemax_samples(pop_size, length)
        """
        init_population["2"] = np.random.randint(0, 3,
                                                 init_population.shape[0])

        init_population["3"] = np.random.randint(0, 3,
                                                 init_population.shape[0])

        init_population[["4", "5"]] = np.ones((init_population.shape[0], 2))
        for idx in range(init_population.shape[0]):
            init_population.loc[idx, random.choice(["4", "5"])] = 0

        for idx in range(init_population.shape[0]):
            row = samples[idx, ["6", "7", "8"]]
        """
        optimizer = Optimizer(samples=init_population)

        model1 = get_onemax_model()
        optimizer.add_objective("ones", model1, direction="maximize")

        model2 = get_linear_model(length)
        optimizer.add_objective("linear_min", model2, direction="minimize")

        optimizer.add_discrete_constraint(fname="2", constraints=[0, 1, 2])
        optimizer.add_discrete_constraint(fname="3", constraints=[0, 1, 2])

        optimizer.add_onehot_group_constraint(group=["4", "5"])

        optimizer.add_sumN_group_constraint(group=["6", "7", "8"],
                                            lower=0, upper=2)

        self.optimizer = optimizer

    def teardown_method(self):
        pass

    def test_init_population(self):
        samples = self.optimizer.samples

        tmp = []
        for val in samples["2"]:
            tmp.append(val in [0, 1, 2])
        assert np.all(tmp)

        tmp = []
        for val in samples["3"]:
            tmp.append(val in [0, 1, 2])
        assert np.all(tmp)

        assert np.all(samples[["4", "5"]].sum(1) == 1.0)
