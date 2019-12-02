from abc import ABCMeta, abstractmethod

import numpy as np


class AbstractBaseMate(mateclass=ABCMeta):

    def __init__(self):
        pass

    def mate(self, population):
        run_mate()

    @abstractmethod
    def run_mate(self):
        raise NotImplementedError()


class CxTwoPoints(AbstractBaseMate):

    def run_mate(self):
        pass



if __name__ == "__main__":
    arr1 = np.arange(1, 13).reshape(3, 4)
    arr2 = np.zeros((3, 4))
