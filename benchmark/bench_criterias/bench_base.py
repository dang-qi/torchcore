
class BenchBase(object):
    def __init__(self, cfg, logger=None):
        self._cfg = cfg
        self._logger = logger
    
    def update(self, targets, pred):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def update_parameters(self, parameters={}):
        pass