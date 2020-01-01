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
    discrete_constraints: Dict[str, List[float]] = None

    #: onehot
    onehot_groups: Dict[str, List[str]] = None

    onehot_constraints: Dict[str, List[float]] = None

    #: sum equal, less, more group constraints
    sumtotal_groups: Dict[str, List[str]] = None

    sumtotal_constraints: Dict[str, List[float]] = None

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
    def group_variables(self):
        groups = []
        if self.onehot_groups is not None:
            for group in self.onehot_groups.values():
                groups.append(group)
        if self.sumtotal_groups is not None:
            for group in self.sumtotal_groups.values():
                groups.append(group)
        return groups

    @property
    def group_variables_indices(self):
        groups = []
        for group in self.group_variables:
            groups.append([self.fname_to_idx(fname) for fname in group])
        return groups

    @property
    def indices_fnames_dict(self):
        return {i: fname for i, fname in enumerate(self.feature_names)}

    @property
    def fnames_indices_dict(self):
        return {fname: i for i, fname in enumerate(self.feature_names)}

    def fname_to_idx(self, fname):
        return self.fnames_indices_dict[fname]

    def idx_to_fname(self, idx):
        return self.indices_fnames_dict[idx]
