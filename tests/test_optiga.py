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

    pop_size = 30
    length = 10

    init_popualtion = get_onemax_samples(pop_size, length)

    optimizer = Optimizer(samples=init_popualtion)

    model = get_onemax_model()
    optimizer.add_objective("ones", model, direction="maximize")

    model = get_linear_model(length)
    optimizer.add_objective("linear", model, direction="maximize")

    optimizer.run(population_size=10, n_gen=1)
