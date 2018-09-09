import torch
import torch.nn as nn

import nms_cpu

# Currently only working on CPU. Implement the gpu version for speed boost.

class NMS( nn.Module ):
    def __init__( self, thresh ):
        super().__init__()
        self._thresh = thresh

    def forward( self, dets, scores ):
        device = dets.device

        x1 = dets[:,0]
        y1 = dets[:,1]
        x2 = dets[:,2]
        y2 = dets[:,3]

        areas = ((x2 - x1 + 1) * (y2 - y1 + 1)).cpu()
        order = (scores.sort(0, descending=True)[1]).cpu()
        dets_cpu = dets.cpu()

        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)

        nms_cpu.forward(keep, num_out, dets_cpu, order, areas, self._thresh)
        return keep[:num_out[0]].to( device )
