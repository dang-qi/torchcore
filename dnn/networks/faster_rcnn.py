import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from .two_stage_detector import TwoStageDetector
from rpn import MyRegionProposalNetwork,MyAnchorGenerator

class FasterRCNN(TwoStageDetector):
    def __init__(self, backbone, num_classes, neck = None, rpn=None, roi_extractor=None, bbox_heads=None, cfg=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        self.backbone = backbone

        if rpn is None:
            anchor_generator = MyAnchorGenerator(sizes=cfg.rpn.anchor_sizes, aspect_ratios=cfg.rpn.aspect_ratios)
            rpn_head = RPNHead(self.backbone.out_channels, anchor_generator.num_anchors_per_location()[0])
            pre_nms_top_n = dict(training=cfg.rpn.train_pre_nums_top_n, testing=cfg.rpn.test_pre_nums_top_n)
            post_nms_top_n = dict(training=cfg.rpn.train_post_nums_top_n, testing=cfg.rpn.test_post_nums_top_n)
            self.rpn = MyRegionProposalNetwork(anchor_generator, 
                                             rpn_head, 
                                             cfg.rpn.fg_iou_thresh, 
                                             cfg.rpn.bg_iou_thresh, 
                                             cfg.rpn.batch_size_per_image, 
                                             cfg.rpn.positive_fraction,
                                             pre_nms_top_n, 
                                             post_nms_top_n, 
                                             cfg.rpn.nms_thresh)
        
        if roi_extractor is None:
            self.roi_extractor = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2)                                    
        
        if bbox_heads is None:
            resolution = self.roi_extractor.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                self.backbone.out_channels * resolution ** 2,
                representation_size)
            
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

            roi_heads = RoIHeads(
            # Box
            self.roi_extractor, box_head, box_predictor,
            cfg.roi_head.fg_iou_thresh, cfg.roi_head.bg_iou_thresh,
            cfg.roi_head.batch_size_per_image, cfg.roi_head.ositive_fraction,
            cfg.roi_head.reg_weights,
            cfg.roi_head.score_thresh, cfg.roi_head.nms_thresh, cfg.roi_head.detections_per_img)
        
        self.neck = neck
        self.rpn = rpn
        self.roi_extractor = roi_extractor
        self.roi_heads = roi_heads


# from torchvision
class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
            