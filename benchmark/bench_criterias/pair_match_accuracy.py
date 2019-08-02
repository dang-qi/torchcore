from .bench_base import BenchBase
import numpy as np

class PairMatchAccuracy(BenchBase):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger)
        self._match_correct = 0
        self._match_total=0

    def update(self, targets, pred):
        match_label = targets['pair_labels'].detach().numpy()
        pre = pred['match'].detach().cpu().numpy()
        cls = np.argmax( pre, axis=1 ).ravel()
        total = len(cls)
        correct = len(np.where(match_label == cls)[0])
        self._match_total += total
        self._match_correct += correct

    def summary(self):
        accuracy = float(self._match_correct)/ self._match_total
        if self._logger is not None:
            self._logger.info('{} '.format(accuracy))
        print('Match Accuracy : {}'.format(accuracy))
        self._match_correct = 0
        self._match_total=0