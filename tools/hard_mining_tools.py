import numpy as np
class HardMining(object):
    def __init__(self):
        pass

class TripletHardMiner(object):
    def __init__(self, cfg, margin=0, p=2, top_k=5000):
        self._cfg = cfg
        self._margin = margin
        self._p = p # the power of distance
        self._cur_ind = 0
        self._batch_size = cfg.dnn.NETWORK.BATCH_SIZE
        self._dist = None
        self._top_k = top_k

    def update(self, output, target):
        anchor = output['anchor'].detach().cpu().numpy()
        pos = output['positive'].detach().cpu().numpy()
        neg = output['negative'].detach().cpu().numpy()
        dist = np.linalg.norm(anchor-pos, ord=self._p, axis=-1) - np.linalg.norm(anchor-neg, ord=self._p, axis=-1) + self._margin
        dist = np.maximum(dist, 0)
        if self._dist is None:
            self._dist = dist
        else:
            self._dist = np.concatenate((self._dist, dist), axis=None)

    def get_hard_examples(self):
        if self._dist is None:
            print('Wrong, there is no samples')
        # get the index of triplets with bigest distance(error)
        ind = np.argpartition(self._dist, -self._top_k)[-self._top_k:]
        ind_bigger_than_zero = np.where(self._dist>0)
        if len(ind) > len(ind_bigger_than_zero[0]):
            ind = ind_bigger_than_zero
            print('The negative sample number is {}'.format(len(ind_bigger_than_zero[0])))
        return ind
        

