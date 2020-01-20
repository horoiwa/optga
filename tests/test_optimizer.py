import numpy as np

from optga.optimizer import Optimizer
from optga.support import (get_linear_model, get_onemax_model,
                           get_onemax_samples)


class TestOptimizer:

    def setup_method(self):

        init_population = get_onemax_samples(500, 10)

        self.optimizer = Optimizer(sample_data=init_population)

        model1 = get_onemax_model()

        self.optimizer.add_objective("ones", model1.predict,
                                     direction="maximize")

        model2 = get_linear_model(10)

        self.optimizer.add_objective("linear_min", model2.predict,
                                     direction="minimize")

    def teardown_method(self):

        del self.optimizer

    def test_evaluate(self):

        X = get_onemax_samples(500, 10)

        fitness = self.optimizer.evaluate_population(X)

        assert np.all(fitness.columns == ["ones", "linear_min"])
