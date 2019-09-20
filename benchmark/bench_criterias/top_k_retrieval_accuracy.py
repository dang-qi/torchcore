from .bench_base import BenchBase
import numpy as np
import json

class TopKRetrievalAccuracy(BenchBase):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger)

    def update(self, targets, pred):
        pre = pred['embeddings'].detach().cpu().numpy()
        uid = targets['uid'].detach().cpu().numpy()
        h,w = pre.shape
        pre = pre.reshape(h,1,w)
        inds = np.arange(0,pre.shape[0])
        parts = [inds[i:i+4] for i in range(0, pre.shape[0], 4)]
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
            for j,ind in enumerate(inds): # ind is the k indexes with smallest distance, j is the index of the item in this batch
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

        #save param for json file
        if self._save_json:
            self.save_json(sorted_ind, targets)



    def summary(self):
        accuracies = self._correct / float(self._total)
        print('query number is {}'.format(self._total))
        for i, k in enumerate(self._k):
            print('top {} accuracy: {}'.format(k, accuracies[i]))
            if self._logger is not None:
                self._logger.info('{} '.format(accuracies[i]))
        if self._save_json:
            with open('result.json', 'w') as f:
                json.dump(self._result,f)

    def update_parameters(self, parameters):
        if 'top_k_retrieval_accuracy' in parameters:
            param = parameters['top_k_retrieval_accuracy']
            self._k = param['k']
            self._embeddings = param['embeddings']
            self._embedding_uids = param['embedding_uids']
            if 'valid_uids' in param:
                self._valid_uids = param['valid_uids']
            if 'image_ids' in param:
                self._image_ids = param['image_ids']
            if 'ori_boxes' in param:
                self._ori_boxes = param['ori_boxes']
            self._save_json = False
            if 'save_json' in param:
                self._save_json = param['save_json']
                self._result = []
            self._correct = np.zeros(len(self._k), dtype=int)
            self._total = 0

    def save_json(self, sorted_ind, targets):
        inds = sorted_ind[:,:20]
        q_ids = targets['image_ids'].cpu().numpy()
        q_boxes = targets['ori_boxes']
        for i, box in enumerate(q_boxes):
            q_boxes[i] = box.cpu().numpy()
        q_cls = targets['category'].cpu().numpy()
        for i, ind in enumerate(inds): # iterate for each item in the batch
            gallery_bbox = self._ori_boxes[ind].tolist()
            gallery_image_id = self._image_ids[ind].tolist()
            query_image_id = q_ids[i].item()
            query_bbox = q_boxes[i].tolist()
            query_cls = q_cls[i].item()
            query_score = 1.0
            result = {'gallery_bbox': gallery_bbox, 'gallery_image_id': gallery_image_id,'query_image_id': query_image_id, 
                        'query_bbox':query_bbox, 'query_cls':query_cls, 'query_score':query_score}
            self._result.append(result)
