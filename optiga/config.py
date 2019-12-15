from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass_json
@dataclass
class OptConfig:

    mate: str = "MateCxTwoPoints"

    mutate: str = "MutateUniform"

    select: str = "SelectNSGA2"

    birth_rate: int = 3

    mutpb: float = 0.2

    indpb: float = 0.1

    objectives: Dict[str, str] = None

    #: upperlim and lowerlim
    limits: Dict[str, List[float]] = None

    #: single constraints
    single_constraints: Dict[str, List[int]] = None

    #: onehot, sumequal
    group_constraints: Dict[str, List[int]] = None

    @property
    def weights(self):
        weights = [1.0 if val == "maximize" else -1.0 if "minimize" else None
                   for val in self.objectives.values()]
        return np.array(weights)

    @property
    def upperlim(self):
        return [val[1] for val in self.limits.values()]

    @property
    def lowerlim(self):
        return [val[0] for val in self.limits.values()]

    @property
    def objective_names(self):
        return [key for key in self.objectives.keys()]

    @property
    def feature_names(self):
        return [key for key in self.limits.keys()]

    @property
    def fname_to_index(self):
        return {key: n for n, key in enumerate(self.limits.keys())}

    @property
    def index_to_fname(self):
        return {n: key for n, key in enumerate(self.limits.keys())}

    def convert_fnames_to_indices(self, fnames):
        return [self.fname_to_index[fname] for fname in fnames]

    def convert_indices_to_fnames(self, indices):
        return [self.index_to_fname[idx] for idx in indices]
