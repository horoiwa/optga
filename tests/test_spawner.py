import pytest

from optga.optimizer import Optimizer
from optga.support import (get_linear_model, get_onemax_model,
                           get_onemax_samples)


class TestSpawner:

    def setup_method(self):
        pop_size = 500
        length = 10

        init_population = get_onemax_samples(pop_size, length)

        optimizer = Optimizer(sample_data=init_population)

        model1 = get_onemax_model()
        optimizer.add_objective("ones", model1.predict, direction="maximize")

        model2 = get_linear_model(length)
        optimizer.add_objective("linear_min", model2.predict,
                                direction="minimize")

        optimizer.add_discrete_constraint(fname="2", constraints=[0, 1, 2])
        optimizer.add_discrete_constraint(fname="3", constraints=[0, 1, 2])

        optimizer.add_onehot_groupconstraint(group=["4", "5"])

        optimizer.add_sumtotal_groupconstraint(group=["6", "7", "8"],
                                               lower=5, upper=5)

        self.optimizer = optimizer

    def teardown_method(self):
        del self.optimizer

    def test_spawn(self):
        population = self.optimizer.spawn_population(100)
        for i in range(population.shape[0]):
            row = population.iloc[i, :]
            assert row["2"] in [0, 1, 2]
            assert row["3"] in [0, 1, 2]
            assert row[["4", "5"]].sum() == pytest.approx(1.0, 0.01)
            assert row[["6", "7", "8"]].sum() == pytest.approx(5.0, 0.01)
