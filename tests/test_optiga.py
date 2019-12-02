from optiga import __version__
from logging import getLogger, INFO, StreamHandler, Formatter


def test_version():
    assert __version__ == '0.1.0'


def main():
    """細かくndarrayの挙動をチェックしつつ
    """
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    stream_handler = StreamHandler()
    handler_format = Formatter(
        '(%(levelname)s)[%(asctime)s]\n%(message)s')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    logger.info("Start Main test")
