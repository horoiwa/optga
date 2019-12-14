from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger

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

    pop_size = 1000
    length = 100

    init_popualtion = get_onemax_samples(pop_size, length)

    optimizer = Optimizer(samples=init_popualtion)

    model1 = get_onemax_model()
    optimizer.add_objective("ones", model1, direction="maximize")

    model2 = get_linear_model(length)

    print("Model1 MAX:", length)
    print("Model2 MAX:", model2.get_max_value())
    print("Model2 MIN:", model2.get_min_value())

    optimizer.add_objective("linear", model2, direction="minimize")

    optimizer.run(population_size=pop_size, n_gen=300)
