from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class StartegyConfig:

    test int = 0


@dataclass_json
@dataclass
class OptimizerConfig:

    test int = 0
