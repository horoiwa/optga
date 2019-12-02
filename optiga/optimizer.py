from deap import creator

from optiga.config import OptimizerConfig


class SimpleOptimizer:

    def __init__(self):
        self.config = OptimizerConfig()
        self._reload_config()


    def generate_config(self, path=None):
        path = "optimizer_config.json" if not path else path
        raise NotImplementedError()

    def reload_config(self, path):
        raise NotImplementedError()

    def _reload_config(self):
        self.mate_metho = None
        self.mutate_method = None
        self.select_method = None
        raise NotImplementedError()

    def mate(self):
        pass

    def mutate(self):
        pass

    def select(self):
        pass



class MLOptimizer(SimpleOptimizer):

    def __init__(self):
        super().__init__()
