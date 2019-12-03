from optiga import __version__
from logging import getLogger, DEBUG, INFO, StreamHandler, Formatter

from optiga.optimizer import Optimizer
from optiga.support import get_onemax

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

    init_popualtion = get_onemax(30, 10)

    optimizer = Optimizer(sample=init_popualtion)
    optimizer.run(n_gen=1)

