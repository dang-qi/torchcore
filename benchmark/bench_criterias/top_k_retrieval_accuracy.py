from .bench_base import BenchBase
import numpy as np

class TopKRetrievalAccuracy(BenchBase):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger)

    def update(self, targets, pred):
        pre = pred['embeddings'].detach().cpu().numpy()
        uid = targets['uid'].detach().cpu().numpy()
        h,w = pre.shape
        pre = pre.reshape(h,1,w)
        inds = np.arange(0,pre.shape[0])
        parts = [inds[i:i+16] for i in range(0, pre.shape[0], 16)]
        diffs = []
        for part in parts:
            diff = np.linalg.norm(pre[part]-self._embeddings, ord=2, axis=-1) # (batch_size, shop_item_num) 
            diffs.append(diff)
        diff = np.concatenate(diffs, axis=0)
            
        #diff = np.linalg.norm(pre-self._embeddings, ord=2, axis=-1) # (batch_size, shop_item_num) 
        sorted_ind = np.argsort(diff, axis=-1)
        self._total += sorted_ind.shape[0]
        for i, n in enumerate(self._k):
            inds = sorted_ind[:,:n]
            for j,ind in enumerate(inds): # ind is the k indexes with smallest distance 
                if uid[j] not in self._valid_uids:
                    continue
                pre_uid = self._embedding_uids[ind]
                if uid[j] in pre_uid:
                    self._correct[i] += 1
        invalid_num = 0
        for auid in uid:
            if auid not in self._valid_uids:
                invalid_num += 1
        self._total -= invalid_num

    def summary(self):
        accuracies = self._correct / float(self._total)
        for i, k in enumerate(self._k):
            print('top {} accuracy: {}'.format(k, accuracies[i]))
            if self._logger is not None:
                self._logger.info('{} '.format(accuracies[i]))

    def update_parameters(self, parameters):
        if 'top_k_retrieval_accuracy' in parameters:
            param = parameters['top_k_retrieval_accuracy']
            self._k = param['k']
            self._embeddings = param['embeddings']
            self._embedding_uids = param['embedding_uids']
            if 'valid_uids' in param:
                self._valid_uids = param['valid_uids']
            self._correct = np.zeros(len(self._k), dtype=int)
            self._total = 0

