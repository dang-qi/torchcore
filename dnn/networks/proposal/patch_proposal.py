import torch.nn as nn
import torchvision

from .functions_patch_proposal import BuildTargetsFunction, PruneFunction
from dnn.networks.losses import SmoothL1Loss
from tools.torch_tools import layer_init

class Net( nn.Module ):
    def _init_weights( self ):
        for name, m in self.named_modules() :
            if len(name)>0 and name.split('.')[0] :
                layer_init(m)

    def __init__( self, cfg ):
        super().__init__()
        self._cfg = cfg
        self._nclasses = cfg.dnn.NETWORK.NCLASSES

        self.conv1 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )
        self.deconv1 = nn.ConvTranspose2d( 512, 256, kernel_size=3, stride=2, padding=1, output_padding=1 )
        self.conv2 = nn.Conv2d( 256, 256, kernel_size=3, padding=1 )
        self.deconv2 = nn.ConvTranspose2d( 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1 )
        self.conv3 = nn.Conv2d( 128, 128, kernel_size=3, padding=1 )

        self.scores = nn.Conv2d( 128, self._nclasses*2, kernel_size=1 )
        self.rois = nn.Conv2d( 128, self._nclasses*4, kernel_size=1 )

        self._init_weights()

    def forward( self, feat ):
        feat = self.conv1( feat )
        feat = self.deconv1( feat )
        feat = self.conv2( feat )
        feat = self.deconv2( feat )
        feat = self.conv3( feat )

        out = {}
        out['scores'] = self.scores( feat )
        out['rois'] = self.rois( feat )

        return out

class Loss( nn.Module ):
    def __init__( self, cfg ):
        super().__init__()
        self._cfg = cfg
        self._nclasses = cfg.dnn.NETWORK.NCLASSES

        self.scores_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.rois_loss = SmoothL1Loss(self._cfg.dnn.PROPOSAL.SMOOTH_L1_SIGMA)

    def forward( self, inputs, nets, targets ):
        scores_target, rois_target = BuildTargetsFunction.apply( nets['scores'], inputs['rois'], inputs['roibatches'],
                                                                 targets['gtboxes'], targets['gtlabels'], targets['gtbatches'],
                                                                 self._nclasses, targets['batch_labels'],
                                                                 self._cfg.dnn.PROPOSAL.EXCLUSIVE )

        scores_target, rois_mask = PruneFunction.apply( scores_target,
                                                        self._cfg.dnn.PROPOSAL.SELECTION_BATCH_SIZE,
                                                        self._cfg.dnn.PROPOSAL.FG_RATIO )

        scores_target = scores_target.contiguous().view( [ -1, *scores_target.shape[2:] ] )
        scores_target = scores_target.detach()

        scores = nets['scores']
        scores = scores.view([-1,2,*scores.shape[2:]])
        scores_loss = self.scores_loss( scores, scores_target )

        rois_mask = rois_mask.detach()
        rois_target = rois_target.detach()

        rois = nets['rois']
        rois = rois * rois_mask
        rois_target = rois_target * rois_mask

        rois_target = rois_target.permute([0,2,3,1]).contiguous().view(-1,4)
        rois = rois.permute([0,2,3,1]).contiguous().view(-1,4)
        rois_loss = self.rois_loss( rois_target, rois )

        rois_loss = self.rois_loss(rois_target, rois)
        rois_loss = rois_loss.sum()

        weights = self._cfg.dnn.PROPOSAL.WEIGHTS

        loss = weights['scores']*scores_loss + weights['rois']*rois_loss

        return loss
