from .bench_base import BenchBase
import numpy as np

class CategoryAccuracy(BenchBase):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger)
        self._correct = 0
        self._total=0

    def update(self, targets, pred):
        cat = targets['category'].detach().numpy()
        pre = pred['category'].detach().cpu().numpy()
        cls = np.argmax( pre, axis=1 ).ravel()

        total = len(cls)
        correct = len(np.where( cat == cls )[0])

        self._total = self._total + total
        self._correct = self._correct + correct

    def summary(self):
        accuracy = float(self._correct)/self._total
        if self._logger is not None:
            self._logger.info('{} '.format(accuracy))
        print('Category Accuracy : {}'.format(accuracy) )
        self._correct = 0
        self._total=0
    