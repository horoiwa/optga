from optga.optimizer import Optimizer
from optga.support import (get_linear_model, get_onemax_model,
                           get_onemax_samples)


class TestConfig:

    def setup_method(self):

        init_population = get_onemax_samples(500, 10)

        self.optimizer = Optimizer(sample_data=init_population)

    def teardown_method(self):

        del self.optimizer

    def test_setpb(self):

        self.optimizer.set_mutpb(0.5)

        self.optimizer.set_indpb(0.4)

        assert self.optimizer.config.mutpb == 0.5

        assert self.optimizer.config.indpb == 0.4
