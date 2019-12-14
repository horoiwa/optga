from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
import copy
from collections import defaultdict

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

        self.history = {}

        self.pareto_fronts = None

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

        self._init_envs()
        self._validate()
        history = defaultdict(lambda: {"Average": [],
                                       "MAX": [],
                                       "MIN": []})

        population = self.spawner.spawn(population_size).values
        for n in range(n_gen):
            population, stats = self.run_generation(population,
                                                    population_size)
            for obj_name in self.config.objective_names:
                history[obj_name]["Average"].append(
                    stats.loc[obj_name, "Average"])
                history[obj_name]["MAX"].append(
                    stats.loc[obj_name, "MAX"])
                history[obj_name]["MIN"].append(
                    stats.loc[obj_name, "MIN"])

            if n % 10 == 0:
                logger.info(f"====Generation {n} ====")
                logger.info(stats)

        logger.info("GA optimization finished gracefully")
        for obj_name in self.config.objective_names:
            logger.info(f"====GA RESULT: {obj_name} ====")
            df = self._get_history(history, obj_name)
            self.history[obj_name] = df
            logger.info(df)

    def run_generation(self, population, population_size):

        ancestor = copy.deepcopy(population)

        children = self.strategy.mate(population)
        children = self.strategy.mutate(children)

        offspring = pd.DataFrame(np.vstack([ancestor, children]),
                                 columns=self.config.feature_names)
        fitness = self.evaluator.evaluate(offspring)
        stats = self._get_stats(fitness)
        weighted_fitness = fitness * self.config.weights

        next_population = self.strategy.select(offspring.values,
                                               weighted_fitness,
                                               population_size)

        return next_population, stats

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

    def _get_stats(self, fitness=None):
        stats = pd.DataFrame(index=self.config.objective_names)
        if fitness is None:
            return stats.T
        else:
            stats["MAX"] = fitness.max(0)
            stats["MIN"] = fitness.min(0)
            stats["Average"] = fitness.mean(0)
            return stats

    def _get_history(self, history, obj_name):
        df = pd.DataFrame()
        df["MAX"] = history[obj_name]["MAX"]
        df["MIN"] = history[obj_name]["MIN"]
        df["Average"] = history[obj_name]["Average"]
        return df

    def _init_envs(self):

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


class PallarelOptimizer(Optimizer):

    def __init__(self, samples, n_jobs):
        super().__init__(samples)
        self.n_jobs = n_jobs
