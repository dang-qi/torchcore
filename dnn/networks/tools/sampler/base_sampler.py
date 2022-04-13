from abc import abstractmethod
import torch
from .sample_result import SampleResult


class BaseSampler():
    def __init__(self, total_num, pos_ratio, add_gt_as_sample=False):
        self.total_num = total_num
        self.pos_num = int(total_num*pos_ratio)
        self.add_gt_as_sample=add_gt_as_sample

    @abstractmethod
    def sample_pos(self,match_result, expect_num):
        pass

    @abstractmethod
    def sample_neg(self,match_result, expect_num):
        pass

    def sample(self, match_result, gt_boxes, anchors, gt_labels):
        if self.add_gt_as_sample:
            anchors = torch.cat([gt_boxes, anchors])
            match_result.add_gt(gt_labels)
        pos_ind = self.sample_pos(match_result, self.pos_num)
        expect_neg = self.total_num-pos_ind.numel()
        neg_ind = self.sample_neg(match_result, expect_neg)
        return SampleResult(pos_ind, neg_ind, gt_boxes, anchors,match_result)
