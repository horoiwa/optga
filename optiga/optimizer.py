from deap import creator

from optiga.config import OptimizerConfig


class SimpleOptimizer:

    def __init__(self, config_path=None):
        self.config = OptimizerConfig()
        self.generate_config(config_path)

        self._reload_config()

    def generate_config(self, path=None):
        path = "optimizer_config.json" if not path else path
        raise NotImplementedError()

    def reload_config(self, path):
        raise NotImplementedError()

    def _reload_config(self):
        """mate mutate selectの動的追加

        Raises
        ------
        NotImplementedError
            [description]
        """
        self.mate_metho = None
        self.mutate_method = None
        self.select_method = None
        raise NotImplementedError()


class MLOptimizer(SimpleOptimizer):

    def __init__(self):
        super().__init__()
