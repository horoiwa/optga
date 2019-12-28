import pytest

import pandas as pd

from optga.optimizer import Optimizer
from optga.support import (get_linear_model, get_onemax_model,
                           get_onemax_samples)


class TestSpawner:

    def setup_method(self):
        pop_size = 500
        length = 10

        init_population = get_onemax_samples(pop_size, length)

        optimizer = Optimizer(samples=init_population)

        model1 = get_onemax_model()
        optimizer.add_objective("ones", model1, direction="maximize")

        model2 = get_linear_model(length)
        optimizer.add_objective("linear_min", model2, direction="minimize")

        optimizer.add_discrete_constraint(fname="2", constraints=[0, 1, 2])
        optimizer.add_discrete_constraint(fname="3", constraints=[0, 1, 2])

        optimizer.add_onehot_groupconstraint(group=["4", "5", "6"])

        optimizer.add_sumtotal_groupconstraint(group=["7", "8", "9"],
                                               lower=2, upper=2)

        optimizer.compile()
        self.optimizer = optimizer
        population = self.optimizer.spawner.spawn(30)
        population = self.optimizer.strategy.constraint(population.values)
        self.population = pd.DataFrame(
            population, columns=self.optimizer.config.feature_names)

    def teardown_method(self):
        del self.optimizer

    def test_discrete_constraint(self):
        population = self.population
        for idx in range(population.shape[0]):
            assert population.loc[idx, "2"] in [0, 1, 2]
            assert population.loc[idx, "3"] in [0, 1, 2]

    def test_onehot_constraint(self):
        population = self.population
        for idx in range(population.shape[0]):
            row = population.loc[idx, ["4", "5", "6"]]
            assert (row != 0).sum() == 1

    def test_sumtotal_constraint(self):
        population = self.population
        for idx in range(population.shape[0]):
            row = population.loc[idx, ["7", "8", "9"]]
            print(row.sum())
            assert row.sum() == pytest.approx(2.0, 0.1)
