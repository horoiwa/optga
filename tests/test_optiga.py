from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
import pickle

import matplotlib.pyplot as plt

from optiga import __version__
from optiga.optimizer import Optimizer
from optiga.support import (get_linear_model, get_onemax_model,
                            get_onemax_samples)


def test_version():
    assert __version__ == '0.1.0'


def main():
    """細かくndarrayの挙動をチェックしつつ
    """
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    stream_handler = StreamHandler()
    handler_format = Formatter(
        '(%(levelname)s)[%(asctime)s]\n%(message)s')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    logger.info("Start Main test")

    pop_size = 500
    length = 20

    init_popualtion = get_onemax_samples(pop_size, length)

    optimizer = Optimizer(samples=init_popualtion)

    model1 = get_onemax_model()
    optimizer.add_objective("ones", model1, direction="maximize")

    model2 = get_linear_model(length)
    optimizer.add_objective("linear_min", model2, direction="minimize")

    if False:
        model3 = get_linear_model(length)
        optimizer.add_objective("linear_max", model3, direction="maximize")

    optimizer.add_discrete_constraint(fname="2", constraints=[0, 1, 2])
    optimizer.add_discrete_constraint(fname="3", constraints=[0, 1, 2])

    optimizer.add_onehot_groupconstraint(group=["4", "5"])
    optimizer.add_onehot_groupconstraint(group=["6", "7"])

    optimizer.add_sumequal_groupconstraint(group=["8", "9", "10"],
                                           lower=0, upper=2)
    optimizer.add_sumequal_groupconstraint(group=["11", "12"],
                                           lower=0, upper=5)

    optimizer.run(population_size=pop_size, n_gen=100)

    Y = optimizer.pareto_front["Y"]
    X = optimizer.pareto_front["X"]

    if True:
        print("Result")
        sample_Y = optimizer.pareto_front["sample_Y"]
        print(Y.shape)
        plt.scatter(Y['ones'], Y['linear_min'])
        plt.scatter(sample_Y['ones'], sample_Y['linear_min'])
        plt.show()

        X.to_csv("example/X.csv", index=False)
        Y.to_csv("example/y.csv", index=False)

    print(X)
    #print("Model1 MAX:", length)
    #print("Model2 MIN:", model2.get_min_value())
    #print("Model3 MAX:", model3.get_max_value())
