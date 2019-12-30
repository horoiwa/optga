import pytest

import pandas as pd

from optga.optimizer import Optimizer
from optga.support import (get_linear_model, get_onemax_model,
                           get_onemax_samples)


class TestMate:

    def setup_method(self):
        pop_size = 500
        length = 10

        init_population = get_onemax_samples(pop_size, length)

        optimizer = Optimizer(samples=init_population)

        model1 = get_onemax_model()
        optimizer.add_objective("ones", model1, direction="maximize")

        model2 = get_linear_model(length)
        optimizer.add_objective("linear_min", model2, direction="minimize")

        optimizer.add_discrete_constraint(fname="1", constraints=[0, 1, 2])

        optimizer.add_onehot_groupconstraint(group=["3", "4", "5"],
                                             lower=1, upper=1)

        optimizer.add_sumtotal_groupconstraint(group=["7", "8", "9"],
                                               lower=2, upper=2)

        optimizer.compile()

        self.optimizer = optimizer
        self.strategy = self.optimizer.strategy
        self.init_population = self.optimizer.spawn_population(100)

    def teardown_method(self):
        del self.optimizer

    def test_init_population(self):
        population = self.init_population
        for i in range(population.shape[0]):
            row = population.iloc[i, :]
            assert row["1"] in [0, 1, 2]
            assert row[["3", "4", "5"]].sum() == pytest.approx(1.0, 0.01)
            assert row[["7", "8", "9"]].sum() == pytest.approx(2.0, 0.01)

        for group in self.optimizer.config.group_variables:
            assert group in [["3", "4", "5"], ["7", "8", "9"]]

        for group in self.optimizer.config.group_variables_indices:
            assert group in [[3, 4, 5], [7, 8, 9]]

    def test_mate_simple(self):
        population = pd.DataFrame(self.strategy.mate(self.init_population),
                                  columns=self.optimizer.config.feature_names)
        for i in range(population.shape[0]):
            row = population.iloc[i, :]
            assert row["1"] in [0, 1, 2]

    def est_mate_group(self):
        population = pd.DataFrame(self.strategy.mate(self.init_population),
                                  columns=self.optimizer.config.feature_names)
        for i in range(population.shape[0]):
            row = population.iloc[i, :]
            assert row[["3", "4", "5"]].sum() == pytest.approx(1.0, 0.01)
            assert row[["7", "8", "9"]].sum() == pytest.approx(2.0, 0.01)

    def test_mutate(self):
        population = self.strategy.mutate(self.init_population)
