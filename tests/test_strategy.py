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

        optimizer.add_onehot_groupconstraint(group=["3", "4", "5"])

        optimizer.add_sumtotal_groupconstraint(group=["7", "8", "9"],
                                               lower=2, upper=2)

        optimizer.compile()

        self.optimizer = optimizer
        self.strategy = self.optimizer.strategy
        self.init_population = self.optimizer.spawn_population(100)

    def teardown_method(self):
        del self.optimizer

    def test_mate(self):
        children = self.strategy.mate(self.init_population)

    def test_mutate(self):
        children = self.strategy.mutate(self.init_population)
