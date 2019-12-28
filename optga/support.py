import numpy as np
import pandas as pd


def get_onemax_samples(size, length):
    df = pd.DataFrame(np.random.rand(size, length))
    df.columns = [str(i) for i in range(length)]
    return df


def get_onemax_model():
    class model:
        def predict(self, samples):
            if not isinstance(samples, pd.DataFrame):
                raise TypeError("Samples must be DataFrame")
            return samples.sum(1)

    return model()


def get_linear_model(length):
    class model:
        def __init__(self, length):
            self.weights = np.random.randn(length).T

        def predict(self, samples):
            if not isinstance(samples, pd.DataFrame):
                raise TypeError("Samples must be DataFrame")
            return np.dot(samples, self.weights)

        def get_max_value(self):
            return self.weights[self.weights > 0].sum()

        def get_min_value(self):
            return self.weights[self.weights <= 0].sum()

    return model(length)
