import torch
from functools import partial
import math
from torch import nn
from torch.functional import meshgrid
#from ..heads.retina_head import RetinaHead
from ..tools import AnchorBoxesCoder
#from ..rpn import MyRegionProposalNetwork, RegionProposalNetwork
from torchvision.ops.boxes import batched_nms, box_area, box_iou, nms
from ..losses import FocalLossSigmoid,FocalLoss,SigmoidFocalLoss
from ..heads.build import build_head
from ..losses.build import build_loss
from .build import DETECTION_HEAD_REG

INF=1000000
BG_LABLE=1000000


@DETECTION_HEAD_REG.register()
class FCOSHead(nn.Module):
    def __init__(self, 
                 head,
                 num_class,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 strides=(8, 16, 32, 64, 128),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='SigmoidFocalLoss',
                     gamma=2.0,
                     alpha=0.25,
                     reduction='mean'
                     ),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='BCEWithLogitsLoss',
                     reduction='mean'
                 ),
                 #loss_centerness=dict(
                 #    type='CrossEntropyLoss',
                 #    use_sigmoid=True,
                 #    loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 test_cfg=dict(nms_pre=1000,
                               min_bbox_size=0,
                               score_thr=0.05,
                               iou_threshold=0.5,
                               #nms=dict(type='nms', iou_threshold=0.5),
                               max_per_img=100)):
        super(FCOSHead, self).__init__()
        self.head = build_head(head)
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius =center_sample_radius
        self.centerness_on_reg = centerness_on_reg
        self.strides=strides
        self.num_class = num_class
        assert num_class < BG_LABLE
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.test_cfg = test_cfg

    def forward(self, inputs, features, targets=None):
        # convert features to list
        if isinstance(features, dict):
            features = list(features.values())
        elif torch.is_tensor(features):
            features = [features]

        #pred_out: List(tuple(class_pred(N,class_num,h,w), bbox_pred(N,4,class_num,h,w), centerness(N,1,class_num,h,w))...)
        pred_out = self.head(features)
        if isinstance(pred_out, dict):
            pred_out = list(pred_out.values())
        
        dtype = pred_out[0][0].dtype
        device = pred_out[0][0].device
        feature_sizes = [p[0].size()[-2:] for p in pred_out]

        feature_mesh = self.generate_meshgrids(feature_sizes, dtype, device)

        if not self.training:
            pred_class = [p[0] for p in pred_out]
            pred_bbox = [p[1] for p in pred_out]
            pred_centerness = [p[2] for p in pred_out]
            image_shapes = inputs['image_sizes']
            results = self.post_detection(pred_class, pred_bbox, pred_centerness, feature_mesh, image_shapes)
            return results
        else:
            # shapes are (N,class_num), (N,4), (N)
            pred_class, pred_bbox, pred_centerness = self.combine_and_permute_predictions(pred_out)
            boxes = [t['boxes'] for t in targets]
            labels = [t['labels'] for t in targets]

            labels_targets_per_level, boxes_targets_per_level = self.generate_targets(boxes, labels, feature_mesh, dtype, device)

            class_targets = torch.cat(labels_targets_per_level, dim=0)
            bbox_targets = torch.cat(boxes_targets_per_level, dim=0)

            batch_size = pred_out[0][0].size(0)

            loss_class, loss_bbox, loss_centerness = self.compute_loss(
                pred_class, pred_bbox, pred_centerness, feature_mesh, batch_size, class_targets, bbox_targets, centerness_targets=None)

            losses = {
                "loss_cls": loss_class,
                "loss_box_reg": loss_bbox,
                "loss_centerness":loss_centerness
            }
            return losses

    def compute_loss(self, pred_class, pred_bbox, pred_centerness, feature_mesh, batch_size, class_targets, bbox_targets, centerness_targets ):

        # find the positive samples
        class_targets_one_hot = torch.zeros((class_targets.size(0),self.num_class), device=pred_class.device, dtype=pred_class.dtype)
        pos_ind_mask = (class_targets != BG_LABLE)
        pos_labels = class_targets[pos_ind_mask]
        pos_num = pos_ind_mask.sum()
        class_targets_one_hot[pos_ind_mask,pos_labels-1] = 1

        flatten_mesh = torch.cat(
            [mesh.repeat(batch_size, 1) for mesh in feature_mesh])

        # bbox targets need to be decoded
        pred_bbox_pos = pred_bbox[pos_ind_mask]
        #print('bbox targets shape',bbox_targets.shape)
        #print('ind mask shape',pos_ind_mask.shape)
        bbox_targets_pos = bbox_targets[pos_ind_mask]
        mesh_pos = flatten_mesh[pos_ind_mask]
        pred_bbox_decode = distance2bbox(pred_bbox_pos, mesh_pos)
        bbox_targets_decode = distance2bbox(bbox_targets_pos, mesh_pos)

        pred_centerness_pos = pred_centerness[pos_ind_mask]
        centerness_targets = self.generate_centerness_target(bbox_targets_pos)

        loss_class = self.loss_cls(pred_class, class_targets_one_hot, average_factor=pos_num) 
        loss_box = self.loss_bbox(pred_bbox_decode, bbox_targets_decode)
        loss_centerness = self.loss_centerness(pred_centerness_pos, centerness_targets)
        #print(loss_centerness)
        return loss_class, loss_box, loss_centerness

    @torch.no_grad()
    def generate_targets(self, boxes, labels, feature_mesh, dtype, device):
        '''Generate targets for fpn based fcos detection
           Parameters:
                boxes(List[Tensors(Mx4)]): targets boxes for each image
                labels(List(Tensors)): labels for each image 
           Return:
                boxes targets(List(Tensors(Nx4xHxW)))
                labels targets(List(Tensors(Nx1xHxW)))
        
        '''
        # shape of feature mash: (w*h,2)
        concat_mesh = torch.cat(feature_mesh, dim=0)
        mesh_per_level = [m.shape[0] for m in feature_mesh]

        boxes_targets_list, labels_targets_list = multi_apply(self.generate_single_image_target, boxes, labels, mesh=concat_mesh, mesh_per_level=mesh_per_level, dtype=dtype, device=device)

        boxes_targets_list = [b.split(mesh_per_level) for b in boxes_targets_list]
        labels_targets_list = [l.split(mesh_per_level) for l in labels_targets_list]

        boxes_targets_per_level = [torch.cat([b[i] for b in boxes_targets_list], dim=0) for i in range(len(mesh_per_level))]
        labels_targets_per_level = [torch.cat([b[i] for b in labels_targets_list], dim=0) for i in range(len(mesh_per_level))]
        return labels_targets_per_level, boxes_targets_per_level


    def generate_single_image_target(self, boxes, labels, mesh, mesh_per_level, dtype, device):
        '''Generate targets for single image in FCOS
            Parameters:
                boxes (Tensor(Nx4)): targets boxes
                labels(Tensor(N)): target labels
                mesh(Tensor(Mx2)): the mesh points from all feature map level
            Return:
                tuple(class_target(Mxclass_num), box_target(Mx4))
                or
                tuple(class_target, box_target, centerness_target)
        '''
        left = mesh[:,0][:,None] - boxes[None][...,0] # shape (M,N)
        top = mesh[:,1][:,None] - boxes[None][...,1]
        right = boxes[None][...,2] - mesh[:,0][:,None]
        bottom = boxes[None][...,3] - mesh[:,1][:,None]
        bbox_targets = torch.stack((left, top, right, bottom), dim=-1) # (M,N,4)

        mesh_num = bbox_targets.shape[0]
        regress_ranges = torch.tensor(self.regress_ranges, device=device, dtype=dtype) # (5,2)
        #print(bbox_targets.shape)

        if self.center_sampling:
            raise NotImplementedError
        else:
            inside_bbox_mask = bbox_targets.min(dim=-1)[0] > 0 # (M,N)
            #print('mask size', inside_bbox_mask.shape)
        
        areas = (boxes[...,2]-boxes[...,0]) * (boxes[...,3]-boxes[...,1]) # (N,)
        areas = areas[None].repeat(mesh_num, 1) #(M,N)
        #print('areas shape', areas.shape)

        # the boxes should be assigned to the corresponding regress range 
        max_size = bbox_targets.max(dim=-1)[0] #(M,N)
        #print('max size:', max_size.shape)
        #max_size = max_size[:,None] #(M,N,1)
        #print(max_size)
        #max_mask = torch.bitwise_and(max_size >=regress_ranges[:,0], max_size <regress_ranges[:,1]) #(N,5)
        #print('max mask shape',max_mask.shape)
        #max_ind = torch.where(max_mask)[1]
        regress_range_mask = torch.zeros_like(inside_bbox_mask)
        start=0
        for i, num in enumerate(mesh_per_level):
            end = start+num
            #print(f'start:{start}, end:{end}')
            regress_range_mask[start:end] = torch.bitwise_and(max_size[start:end]>=regress_ranges[i,0], max_size[start:end]<regress_ranges[i,1] )
            start = end

        # each mesh points can only be assigned to the smallest box 
        # if more than one box is assigned
        areas[inside_bbox_mask==0] = INF
        areas[regress_range_mask==0] = INF
        area_min, min_ind = areas.min(-1)

        label_targets = labels[min_ind]
        label_targets[area_min==INF] = BG_LABLE # set background label
        bbox_targets = bbox_targets[range(mesh_num), min_ind]

        return bbox_targets, label_targets
        #print(label_targets.shape)
        #print(bbox_targets.shape)

    def generate_meshgrids(self, feature_sizes, dtype, device):
        meshgrids_multi_level = []
        for feature_size, stride in zip(feature_sizes, self.strides):
            meshs = self.generate_single_meshgrids(feature_size, stride, dtype, device)
            meshgrids_multi_level.append(meshs)
        return meshgrids_multi_level

    def generate_single_meshgrids(self,feature_size, stride, dtype, device):
        '''
        meshgrids(shape:(W*H)x2): the meshgrids 
        '''
        h,w = feature_size
        x_range = torch.arange(0, w, device=device).to(dtype)
        y_range = torch.arange(0, h, device=device).to(dtype)
        y,x = torch.meshgrid(y_range, x_range)
        x = x*stride
        y = y*stride
        meshgrids = torch.stack((x.flatten(), y.flatten()), dim=-1) + stride//2
        return meshgrids
    
    def generate_centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.
        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)
        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


    def post_detection(self, pred_class, pred_bbox, pred_centerness, feature_mesh, image_shapes ):
        # pre nms for each feature layers in each image
        # crop the boxes so they are inside the image
        # delete very small boxes
        # post nms for all the proposals for each image
        # pred_class shape: N * Num_anchor_all * C
        cfg = self.test_cfg

        boxes_all_level = []
        scoers_all_level = []
        labels_all_level = []
        p_class_all_level = []

        batch_size = pred_class[0].size(0)
        for p_class, p_bbox, p_centerness, mesh, in zip(pred_class, pred_bbox, pred_centerness, feature_mesh ):
            p_class = torch.sigmoid_(p_class)
            p_centerness = torch.sigmoid_(p_centerness)

            p_class = p_class.permute(0,2,3,1).reshape(batch_size, -1, self.num_class)
            p_bbox = p_bbox.permute(0,2,3,1).reshape(batch_size, -1, 4)
            p_centerness = p_centerness.permute(0,2,3,1).reshape(batch_size, -1)

            mesh = mesh.expand(batch_size,-1,2)

            nms_pre = min(cfg['nms_pre'],p_class.size(1))

            if nms_pre>0:
                max_score, labels = (p_class*p_centerness[...,None]).max(-1)
                max_score_topk, topk_inds = max_score.topk(nms_pre)
                #print('max score shape',max_score.shape)
                #print('lables shape',labels.shape)
                #print('topk inds shape',topk_inds.shape)
                #print('p class shape', p_class.shape)
                #print('lables', labels[0,:5])

                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()

                batch_inds_class = torch.arange(batch_size).view(
                    -1, 1).expand_as(labels).long()
                x,y = torch.meshgrid(torch.arange(batch_size), torch.arange(labels.size(1)))

                mesh = mesh[batch_inds, topk_inds, :]
                p_bbox = p_bbox[batch_inds, topk_inds, :]
                p_class = p_class[batch_inds_class, labels]
                p_class = p_class[x.flatten(), y.flatten(), labels.flatten()].reshape(batch_size, -1)
                p_class = p_class[batch_inds, topk_inds]
                labels = labels[batch_inds, topk_inds]
                score = max_score_topk
                #print('p box shape',p_bbox.shape)
                #scores = scores[batch_inds, topk_inds, :]
                #centerness = centerness[batch_inds, topk_inds]
            p_bbox = distance2bbox(p_bbox, mesh, image_shapes)

            boxes_all_level.append(p_bbox)
            scoers_all_level.append(score)
            labels_all_level.append(labels)
            p_class_all_level.append(p_class)
        boxes_all = torch.cat(boxes_all_level, dim=1)
        scores_all = torch.cat(scoers_all_level, dim=1)
        labels_all = torch.cat(labels_all_level, dim=1)
        p_class_all = torch.cat(p_class_all_level, dim=1)

        boxes_out = []
        labels_out = []
        scores_out = []

        for i in range(batch_size):
            boxes_im = boxes_all[i]
            scores_im = scores_all[i]
            labels_im = labels_all[i]
            p_class_im = p_class_all[i]

            # only keep the prediction with higher score
            keep = p_class_im > cfg['score_thr']
            boxes_im = boxes_im[keep]
            scores_im = scores_im[keep]
            labels_im = labels_im[keep]

            # remove super small boxes
            keep = self.remove_small_boxes(boxes_im, min_size=cfg['min_bbox_size'])
            boxes_im = boxes_im[keep]
            scores_im = scores_im[keep]
            labels_im = labels_im[keep]

            # nms
            iou_thresh = cfg['iou_threshold']
            keep = batched_nms(boxes_im, scores_im, labels_im, iou_threshold=iou_thresh)
            keep = keep[:cfg['max_per_img']]
            boxes_im = boxes_im[keep]
            scores_im = scores_im[keep]
            labels_im = labels_im[keep]

            boxes_out.append(boxes_im)
            scores_out.append(scores_im)
            labels_out.append(labels_im+1)

        results = dict()
        results['boxes'] = boxes_out
        results['scores'] = scores_out
        results['labels'] = labels_out
        return results



        pred_class = torch.sigmoid_(pred_class)

        class_num = pred_class.shape[-1]

        batch_size, num_anchors_all, C = pred_class.shape

        boxes_all = []
        scores_all = []
        labels_all = []
        for i in range(batch_size):
            pred_class_image = pred_class[i]
            pred_bbox_image = pred_bbox[i]
            image_shape = image_shapes[i]

            # crop the box inside the image
            if not self.no_post_clip:
                pred_bbox_image = self.crop_boxes(pred_bbox_image, image_shape)

            boxes_image = []
            scores_image = []
            labels_image = []
            for class_ind in range(class_num):
                pred_class_image_the_class= pred_class_image[:,class_ind]

                # remove detection with low score
                keep = pred_class_image_the_class > self.score_thresh
                scores_class = pred_class_image_the_class[keep]
                boxes_class = pred_bbox_image[keep]

                # remove too small detections
                keep = self.remove_small_boxes(boxes_class, min_size=1e-2)
                boxes_class = boxes_class[keep]
                scores_class = scores_class[keep]

                # doing non maximum surpress
                keep = nms(boxes_class, scores_class, self.nms_thresh)

                # get topk score boxes for the level
                keep = keep[:self.get_post_nms_top_n()]
                boxes_class = boxes_class[keep]
                scores_class = scores_class[keep]
                labels_class = torch.full_like(scores_class, class_ind, dtype=int)

                boxes_image.append(boxes_class)
                scores_image.append(scores_class)
                labels_image.append(labels_class)
            
            boxes_image = torch.cat(boxes_image,dim=0)
            scores_image = torch.cat(scores_image,dim=0)
            labels_image = torch.cat(labels_image,dim=0)

            
            boxes_all.append(boxes_image)
            scores_all.append(scores_image)
            labels_all.append(labels_image+1)

        results = dict()
        results['boxes'] = boxes_all
        results['scores'] = scores_all
        results['labels'] = labels_all
        return results

    def remove_small_boxes(self, boxes, min_size):
        area = box_area(boxes)
        keep = area > min_size
        return keep

    def crop_boxes(self, boxes, image_size):
        # boxes: N * 4 tensor, x1, y1, x2, y2 format
        # image_size: height, width
        height, width = image_size
        boxes[...,0] = boxes[...,0].clamp(min=0, max=width)
        boxes[...,1] = boxes[...,1].clamp(min=0, max=height)
        boxes[...,2] = boxes[...,2].clamp(min=0, max=width)
        boxes[...,3] = boxes[...,3].clamp(min=0, max=height)
        return boxes


    def combine_and_permute_predictions(self, predictions):
        #pred_out: List(tuple(class_pred(N,class_num,h,w), bbox_pred(N,4,class_num,h,w), centerness(N,1,class_num,h,w))...)
        pred_class_all = []
        pred_bbox_deltas_all = []
        pred_centerness_all = []
        # for each feature map, potentially have multiple batch, do the process
        for pred_class, pred_bbox_deltas, pred_centerness in predictions:
            # the shape of original pred_class: N * C * H * W
            # the shape of original pred_bbox_deltas: N * 4 * H * W
            # target shape should be same with anchor NxHxW * 4 

            # keep the batch and last C, the output should be NxHxW * C
            pred_class = permute_and_flatten(pred_class)
            pred_bbox_deltas = permute_and_flatten(pred_bbox_deltas)
            pred_centerness = permute_and_flatten(pred_centerness).reshape(-1)

            pred_class_all.append(pred_class)
            pred_bbox_deltas_all.append(pred_bbox_deltas)
            pred_centerness_all.append(pred_centerness)
        pred_class = torch.cat(pred_class_all, dim=0)
        pred_bbox_deltas = torch.cat(pred_bbox_deltas_all, dim=0)
        pred_centerness = torch.cat(pred_centerness_all)
        return pred_class, pred_bbox_deltas, pred_centerness



            
def permute_and_flatten(pred ):
    C = pred.shape[1]
    pred = pred.permute(0, 2, 3, 1)
    pred = pred.reshape(-1, C)
    return pred

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def distance2bbox(box_distance, mesh, max_shape=None):
    boxes = torch.empty_like(box_distance)
    boxes[...,0] = mesh[...,0] - box_distance[...,0]
    boxes[...,1] = mesh[...,1] - box_distance[...,1]
    boxes[...,2] = mesh[...,0] + box_distance[...,2]
    boxes[...,3] = mesh[...,1] + box_distance[...,3]

    if max_shape is not None:
        if not isinstance(max_shape, torch.Tensor):
            max_shape = boxes.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(boxes)
        if max_shape.ndim == 2:
            assert boxes.ndim == 3
            assert max_shape.size(0) == boxes.size(0)

        min_xy = boxes.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        boxes = torch.where(boxes < min_xy, min_xy, boxes)
        boxes = torch.where(boxes > max_xy, max_xy, boxes)

    return boxes