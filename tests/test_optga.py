import pytest
import os
import shutil

from optga.optimizer import Optimizer
from optga.support import (get_linear_model, get_onemax_model,
                           get_onemax_samples)


class Testoptga:

    def setup_method(self):

        length = 20
        init_popualtion = get_onemax_samples(500, length)
        optimizer = Optimizer(sample_data=init_popualtion)

        model1 = get_onemax_model()
        optimizer.add_objective("ones", model1.predict, direction="maximize")

        model2 = get_linear_model(length)
        optimizer.add_objective("linear_min", model2.predict,
                                direction="minimize")

        model3 = get_linear_model(length)
        optimizer.add_objective("linear_max", model3.predict,
                                direction="maximize")

        optimizer.add_discrete_constraint(fname="2", constraints=[0, 1, 2])
        optimizer.add_discrete_constraint(fname="3", constraints=[0, 1, 2])

        optimizer.add_valuerange_constraint(fname="2", lower=0, upper=2)
        optimizer.add_valuerange_constraint(fname="3", lower=0, upper=2)

        optimizer.add_onehot_groupconstraint(group=["4", "5"])
        optimizer.add_onehot_groupconstraint(group=["6", "7"])

        optimizer.add_sumtotal_groupconstraint(group=["8", "9", "10"],
                                               lower=1, upper=1)
        optimizer.add_sumtotal_groupconstraint(group=["11", "12"],
                                               lower=0, upper=1)

        def user_func(X):
            print("test func is called!!")
            print(X.columns)
            return X

        optimizer.add_user_constraint(user_func)

        self.optimizer = optimizer
        self.tempdir = os.path.join(os.path.expanduser("~"), "temp")
        if not os.path.exists(self.tempdir):
            os.makedirs(self.tempdir)

    def teardown_method(self):
        del self.optimizer
        shutil.rmtree(self.tempdir)

    def test_GA(self):
        self.optimizer.show_config()
        self.optimizer.export_config(os.path.join(self.tempdir, "config.json"))
        self.optimizer.reload_config(os.path.join(self.tempdir, "config.json"))

        self.optimizer.run(population_size=500, n_gen=10)

        X = self.optimizer.pareto_front["X_pareto"]
        for i in range(X.shape[0]):
            row = X.iloc[i, :]
            assert row["2"] in [0, 1, 2]
            assert row["3"] in [0, 1, 2]
            assert row[["4", "5"]].sum() == pytest.approx(1.0, 0.01)
            assert row[["6", "7"]].sum() == pytest.approx(1.0, 0.01)
            assert row[["8", "9", "10"]].sum() == pytest.approx(1.0, 0.01)
            assert row[["11", "12"]].sum() <= 1.0 + 0.01
            assert row[["11", "12"]].sum() >= 0.01
