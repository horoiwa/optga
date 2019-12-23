from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
import uuid
import copy
from collections import defaultdict

import numpy as np
import pandas as pd

from optiga.config import OptConfig
from optiga.evaluater import Evaluator
from optiga.spawner import Spawner
from optiga.strategy import EvolutionStrategy
from optiga.tools.nsga2 import get_paretofront


def get_logger():
    logger = getLogger("RUN_GA")
    logger.setLevel(DEBUG)
    stream_handler = StreamHandler()
    handler_format = Formatter(
        '(%(levelname)s)[%(asctime)s]\n%(message)s')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    return logger


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

        self.pareto_front = {}

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

    def add_discrete_constraint(self, fname, constraints):
        """Add discrete_constraint

        Parameters
        ----------
        fname: str
            feature_name

        constraints : List[numeric]
            List of discrete values

        Raises
        ------
        Exception
            if fname no exists in self.config.feature_names
        """
        if fname not in self.config.feature_names:
            raise Exception(f"{fname} not in {self.config.feature_names}")

        if self.config.discrete_constraints is None:
            self.config.discrete_constraints = {fname: constraints}
        else:
            self.config.discrete_constraints[fname] = constraints

    def add_onehot_groupconstraint(self, group):
        """Add onehot group constraints

        Parameters
        ----------
        group: List[fname]
            List of feature names

        Raises
        ------
        Exception
            [description]
        """
        for fname in group:
            if fname not in self.config.feature_names:
                raise Exception(f"{fname} not in {self.config.feature_names}")

        if self.config.onehot_constraints is None:
            self.config.onehot_constraints = [group]
        else:
            self.config.onehot_constraints += [group]

    def add_sumequal_groupconstraint(self, group, lower, upper):
        """Add sum equal constraints
        if lower_lim == upper_lim, sum equal constraints

        Parameters
        ----------
        group : List[fname]
            List of feature_names
        lower : float
            lower lim
        upper : float
            upper lim

        """
        for fname in group:
            if fname not in self.config.feature_names:
                raise Exception(f"{fname} not in {self.config.feature_names}")

        if lower > upper:
            lower, upper = upper, lower

        uid = str(uuid.uuid4())
        if self.config.sumN_constraints is None:
            self.config.sumN_groups = {uid: group}
            self.config.sumN_constraints = {uid: [lower, upper]}
        else:
            self.config.sumN_groups.update({uid: group})
            self.config.sumN_constraints.update({uid: [lower, upper]})

    def spawn_population(self, n):
        raise NotImplementedError()

    def compile(self):

        self.spawner = Spawner(self.config)

        self.strategy = EvolutionStrategy(self.config)

        self.evaluator = Evaluator(self.config, self.models)

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
        logger = get_logger()

        logger.info(f"Start GA optimization: {n_gen} gens")
        logger.info(f"Settings:\n{self.show_config(stdout=False)}")

        self.compile()
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

        self._post_run(population)

    def run_generation(self, population, population_size):

        ancestor = copy.deepcopy(population)

        children = self.strategy.mate(population)
        children = self.strategy.mutate(children)
        #children = self.strategy.constraint(children)

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

    def _post_run(self, population):
        population = pd.DataFrame(population,
                                  columns=self.config.feature_names)
        fitness = self.evaluator.evaluate(population)
        weighted_fitness = fitness * self.config.weights

        pareto_front = get_paretofront(population.values, weighted_fitness)
        pareto_front = pd.DataFrame(pareto_front,
                                    columns=self.config.feature_names)

        pareto_fitness = self.evaluator.evaluate(pareto_front)
        pareto_fitness = pd.DataFrame(pareto_fitness,
                                      columns=self.config.objective_names)

        self.pareto_front["X"] = pareto_front
        self.pareto_front["Y"] = pareto_fitness

        self.pareto_front["sample_X"] = self.samples
        self.pareto_front["sample_Y"] = pd.DataFrame(
            self.evaluator.evaluate(self.samples),
            columns=self.config.objective_names)

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
