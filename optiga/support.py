import numpy as np
import pandas as pd



def get_onemax(size, length):
    df = pd.DataFrame(np.random.rand(size, length))
    df.columns = [str(i) for i in range(length)]
    return df
