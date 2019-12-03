from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Dict, List


@dataclass_json
@dataclass
class OptConfig:

    mate: str = "CxTwoPointsMate"

    mutate: str = "UniformMutation"

    select: str = "SelectNSGA2"

    objectives: Dict[str, str] = None

    #: upperlim and lowerlim
    limits: Dict[str, List[float]] = None

    #: single constraints
    single_constraints: Dict[str, List[int]] = None

    #: onehot, sumequal
    group_constraints: Dict[str, List[int]] = None

    @property
    def upperlim(self):
        return [val[1] for val in self.limits.values()]

    @property
    def lowerlim(self):
        return [val[0] for val in self.limits.values()]

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
