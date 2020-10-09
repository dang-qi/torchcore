import torch

class PosNegSampler():
    def __init__(self, pos_num, neg_num):
        self.pos_num = pos_num
        self.neg_num = neg_num

    def sample(self, pos_ind, neg_ind):
        if len(pos_ind) <= self.pos_num:
            pos_num = len(pos_ind)
            neg_num = self.pos_num - len(pos_ind) + self.neg_num
        else:
            pos_num = self.pos_num
            neg_num = self.neg_num

        out_pos = torch.randperm(len(pos_ind))
        out_pos = out_pos[:pos_num]
        out_neg = torch.randperm(len(neg_ind))
        out_neg = out_neg[:neg_num]
        #pos_ind = pos_ind[out_pos]
        #neg_ind = neg_ind[out_neg]
        return out_pos, out_neg

    def sample_batch(self, pos_inds, neg_inds):
        out_pos_inds = []
        out_neg_inds = []
        for pos_ind, neg_ind in zip(pos_inds, neg_inds):
            out_pos, out_neg = self.sample(pos_ind, neg_ind)
            out_pos_inds.append(out_pos)
            out_neg_inds.append(out_neg)

        return out_pos_inds, out_neg_inds