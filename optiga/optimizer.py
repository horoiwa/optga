from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger

import numpy as np
import pandas as pd
from deap import creator

from optiga.config import OptConfig
from optiga.evaluater import Evaluator, OneMaxEvaluator
from optiga.spawner import Spawner
from optiga.strategy import Strategy


class Optimizer:

    def __init__(self, sample):

        if not isinstance(sample, pd.DataFrame):
            raise Exception("Sample data must be DataFrame")

        self.sample = sample

        self.config = OptConfig()

        self.single_constraints = {}

        self.group_constraints = {}

        self.config.limits = self._get_limits(self.sample)

        self.models = {}

        self.prep()

    def prep(self):

        self.evaluator = Evaluator(self.config, self.models)

        self.spawner = Spawner(self.config)

        self.strategy = Strategy(self.config)

    def add_objective(self, objname, model, direction):
        raise NotImplementedError()

    def add_single_constraint(self, constraint_type, group_name, fnames):
        raise NotImplementedError()

    def add_group_constraint(self, constraint_type, group_name, fnames):
        raise NotImplementedError()

    def run(self, n_gen, logfile=False):
        """run evolutional optimization
        Todo: 最初と最後だけIndividualになる方式で

        Parameters
        ----------
        n_gen : int
            number of generation
        ""[summary]
        """
        logger = getLogger("RUN_GA")
        logger.setLevel(DEBUG)
        stream_handler = StreamHandler()
        handler_format = Formatter(
            '(%(levelname)s)[%(asctime)s]\n%(message)s')
        stream_handler.setFormatter(handler_format)
        logger.addHandler(stream_handler)

        logger.info(f"Start GA optimization: {n_gen} gens")
        logger.info(f"Settings:\n{self.show_config(stdout=False)}")

        self.prep()

        self._validate_models()

    def run_epoch(self):
        pass

    def save_config(self, path=None):
        path = "config.json" if not path else path
        raise NotImplementedError()

    def show_config(self, stdout=True):
        config = self.config.to_json(indent=2, ensure_ascii=False)
        if stdout:
            print(config)
        else:
            return config

    def _get_limits(self, sample):
        limits = {}
        for col in sample.columns:
            limits[col] = [sample[col].min(),
                           sample[col].max()]
        return limits

    def _validate_models(self):
        pass

class IslandsOptimizer(Optimizer):
    pass
