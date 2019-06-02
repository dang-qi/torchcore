import torch
import torch.nn as nn
import numpy as np

class AddGt( nn.Module ):
    def __init__( self ):
        super().__init__()
        #self._sigma2 = sigma*sigma

    def forward( self, targets, outputs ):
        if self.training :
            outputs['rois'] = torch.cat([ outputs['rois'], targets['gtboxes']], 0)
            outputs['roilabels'] = torch.cat([ outputs['roilabels'], targets['gtlabels']], 0)
            outputs['roibatches'] = torch.cat([ outputs['roibatches'], targets['gtbatches']], 0)

            # Removing Small ROIS
            rois = outputs['rois'].detach().cpu().numpy()
            w = (rois[:,2] - rois[:,0]).ravel()
            h = (rois[:,3] - rois[:,1]).ravel()

            keep = np.where((w>=5) & (h>=5))[0]

            outputs['rois'] = outputs['rois'][keep]
            outputs['roilabels'] = outputs['roilabels'][keep]
            outputs['roibatches'] = outputs['roibatches'][keep]

        return outputs
