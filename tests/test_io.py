import shutil
import os

import pandas as pd

from optga.optimizer import Optimizer
from optga.support import (get_linear_model, get_onemax_model,
                           get_onemax_samples)


class TestIO:

    def setup_method(self):
        pop_size = 500
        length = 10

        init_population = get_onemax_samples(pop_size, length)

        optimizer = Optimizer(sample_data=init_population)

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

        self.export_dir = os.path.join(os.path.expanduser("~"), "optga")

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def teardown_method(self):
        del self.optimizer
        shutil.rmtree(self.export_dir)

    def test_io_config(self):
        self.optimizer.export_config(
            os.path.join(self.export_dir, "config.json"))
        self.optimizer.reload_config(
            os.path.join(self.export_dir, "config.json"))
        self.optimizer.show_config()

    def test_export_result(self):
        self.optimizer.show_config()
        self.optimizer.run(population_size=500, n_gen=10)

        self.optimizer.export_result(self.export_dir)
        assert os.path.exists(os.path.join(self.export_dir, "X_init.csv"))
        assert os.path.exists(os.path.join(self.export_dir, "Y_init.csv"))

        assert os.path.exists(os.path.join(self.export_dir, "X_pareto.csv"))
        assert os.path.exists(os.path.join(self.export_dir, "Y_pareto.csv"))

        assert os.path.exists(
            os.path.join(self.export_dir, "log_ones.csv"))
        assert os.path.exists(
            os.path.join(self.export_dir, "log_linear_min.csv"))
