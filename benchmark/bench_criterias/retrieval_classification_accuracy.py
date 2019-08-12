from .bench_base import BenchBase
import numpy as np

class RetrievalClassificationAccuracy(BenchBase):
    def __init__(self, cfg, k_vecs, logger=None):
        super().__init__(cfg, logger)
        self._k_vecs = k_vecs
        self._class_num = cfg.dnn.NETWORK.NCLASSES
        self._batch_size = cfg.dnn.NETWORK.BATCH_SIZE
        self._sample_each_class = []
        for k_vec in k_vecs:
            self._sample_each_class.append(k_vec.shape[0] / self._class_num) # should be 5, 10, 20)
        self._top_n = [3, 10] # TODO should add to cfg
        self._correct = np.zeros((len(k_vecs),len(self._top_n)))
        self._total = np.zeros((len(k_vecs),len(self._top_n)))

    def update(self, targets, pred):
        # sample_each_class = k
        pre = pred['embeddings'].detach().cpu().numpy()
        category = targets['category'].detach().cpu().numpy()
        h,w = pre.shape
        pre = pre.reshape(h,1,w)
        for ind_vec, vec in enumerate(self._k_vecs): # the dimension of each vec is (k*class_num, feature_dim)
            diff = np.linalg.norm(pre-vec, ord=2, axis=-1) # (batch_size, k*class_num) 
            sorted_ind = np.argsort(diff)
            for i, n in enumerate(self._top_n):
                ind = sorted_ind[:,:n]
                ind_of_top_n = np.floor(ind/ self._sample_each_class[ind_vec]).astype(int) # (batch_size, top_n)
                max_indexes = np.zeros(self._batch_size)
                for j, row in enumerate(ind_of_top_n):
                    count = np.bincount(row)
                    if len(np.flatnonzero(count==count.max()))>1:
                        pass # there is more than one max value in this case
                    max_indexes[j] = np.random.choice(np.flatnonzero(count==count.max()))
                self._correct[ind_vec][i] += len(np.where(max_indexes == category)[0])
                self._total[ind_vec][i] += len(category)

    def summary(self):
        for i, sample_num in enumerate(self._sample_each_class):
            for j, n in enumerate(self._top_n):
                accuracy = float(self._correct[i][j]/ self._total[i][j])
                if self._logger is not None:
                    self._logger.info('{} '.format(accuracy))
                print('Sample each class:{}, Top {} Accuracy : {}'.format(sample_num, n, accuracy))
        self._correct = np.zeros((len(self._k_vecs),len(self._top_n)))
        self._total = np.zeros((len(self._k_vecs),len(self._top_n)))