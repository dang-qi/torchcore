import torch
from .base_sampler import BaseSampler
from .sample_result import SampleResult
from ..box_matcher.match_result import MatchResult
from .build import SAMPLER_REG

@SAMPLER_REG.register(force=True)
class RandomSampler(BaseSampler):
    def __init__(self, total_num, pos_ratio, add_gt_as_sample=False):
        super().__init__(total_num, pos_ratio)
        self.add_gt_as_sample = add_gt_as_sample

    def sample_pos(self, match_result, expect_num):
        pos_ind = torch.where(match_result.matched_ind >=0)[0]
        if pos_ind.numel()> expect_num:
            permute_ind = torch.randperm(len(pos_ind))[:expect_num].to(pos_ind.device)
            pos_ind = pos_ind[permute_ind]
        return pos_ind

    def sample_neg(self, match_result, expect_num):
        neg_ind = torch.where(match_result.matched_ind ==MatchResult.NEGATIVE_MATCH)[0]
        if neg_ind.numel()> expect_num:
            permute_ind = torch.randperm(len(neg_ind))[:expect_num].to(neg_ind.device)
            neg_ind = neg_ind[permute_ind]
        return neg_ind