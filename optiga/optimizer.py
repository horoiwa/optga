from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
import copy

import numpy as np
import pandas as pd
from deap import creator

from optiga.config import OptConfig
from optiga.evaluater import Evaluator
from optiga.spawner import Spawner
from optiga.strategy import EvolutionStrategy


class Optimizer:

    def __init__(self, samples):

        if not isinstance(samples, pd.DataFrame):
            raise Exception("Sample data must be DataFrame")

        self.samples = samples

        self.config = OptConfig()

        self.single_constraints = {}

        self.group_constraints = {}

        self.config.limits = self._get_limits(self.samples)

        self.models = {}

    def add_objective(self, objname, model, direction):
        #: validate model using samples
        try:
            model.predict(self.samples)
        except:
            raise Exception("Invalid predict model")

        if direction not in ["maximize", "minimize"]:
            raise KeyError(f'direction must be "maximize" or "minimize"')

        self.models[objname] = model

        if not self.config.objectives:
            self.config.objectives = {objname: direction}
        else:
            self.config.objectives[objname] = direction

    def add_single_constraint(self, constraint_type, group_name, fnames):
        raise NotImplementedError()

    def add_group_constraint(self, constraint_type, group_name, fnames):
        raise NotImplementedError()

    def run(self, population_size, n_gen, logfile=False):
        """run evolutional optimization
        Todo: 最初と最後だけIndividualになる方式で

        Parameters
        ----------
        population_size : int
            number of individuals of one generation
        n_gen : int
            number of generation
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

        self._prep()
        self._validate()

        population = self.spawner.spawn(population_size).values
        for n in range(n_gen):
            logger.info("====Generation {n} ====")
            population = self.run_generation(population, population_size)

    def run_generation(self, population, population_size):
        import time
        ancestor = copy.deepcopy(population)

        print(ancestor)
        print()

        children = self.strategy.mate(population)

        start = time.time()
        print(children)
        print()

        children = self.strategy.mutate(children)

        print(children)
        print("Mutate:", time.time()-start)

        offspring = pd.DataFrame(np.vstack([ancestor, children]),
                                 columns=self.config.feature_names)
        fitness = self.evaluator.evaluate(offspring)
        next_population = self.strategy.select(offspring, fitness, population_size)

        return next_population

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

    def _prep(self):

        self.spawner = Spawner(self.config)

        self.strategy = EvolutionStrategy(self.config)

        self.evaluator = Evaluator(self.config, self.models)

    def _validate(self):
        if not self.config.objectives:
            raise Exception("Objective not found")
        elif not len(self.models):
            raise Exception("Model not found")

        for key in self.config.objectives.keys():
            if key not in self.models:
                raise Exception(f"model {key} not found")

        if None in self.config.weights:
            raise Exception("Unexpected Error")


class IslandsOptimizer(Optimizer):

    def __init__(self, samples, n_jobs):
        super().__init__(samples)
        self.n_jobs = n_jobs
