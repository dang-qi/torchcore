import torch
import math
from torch import nn
from torch.functional import meshgrid
from ..heads import MultiConvHead
from ..losses import FocalLossSigmoid, SmoothL1Loss
from collections import OrderedDict
from torchvision.ops import batched_nms

class FoveaHead(nn.Module):
    def __init__(self, class_num, in_feature_num, loss=None, sigma=0.4, eta=2, nms_pre=1000):
        super().__init__()
        self.strides = None
        self.sigma = sigma
        self.eta = eta
        self.class_num = class_num
        self.nms_pre = nms_pre
        self.cls_head = MultiConvHead(out_channel=class_num, in_channel=in_feature_num, head_conv_channel=256, middle_layer_num=4, init='focal_loss' )
        self.box_head = MultiConvHead(out_channel=4, in_channel=in_feature_num, head_conv_channel=256, middle_layer_num=4, init='gaussion' )
        self.set_r_interval()
        if loss is None:
            self.class_loss = FocalLossSigmoid(alpha=0.4, beta=4, gamma=1.5)
            self.bbox_loss = nn.SmoothL1Loss(beta=0.11)
        else:
            self.class_loss = loss['class_loss']
            self.bbox_loss = loss['bbox_loss']
        

    def forward(self, features, inputs, targets=None):
        assert isinstance(features, dict)
        #for k,v in features.items():
        #    print(k, v.shape)
        if self.strides is None:
            self.strides = self.get_strides(inputs, features)
        
        bbox_pred = OrderedDict()
        class_pred = OrderedDict()
        for k, v in features.items():
            bbox_pred[k] = self.box_head(v)
            class_pred[k] = self.cls_head(v)
            
        if self.training:
            bbox_pred_shape = [v.shape for v in bbox_pred.values()]
            class_pred_shape = [v.shape for v in class_pred.values()]
            device = list(bbox_pred.values())[0].device
            dtype = list(bbox_pred.values())[0].dtype
            bbox_targets, class_targets = self.multi_scale_target(targets, self.strides, bbox_pred_shape, class_pred_shape, device, dtype)

            # flatten the pred and targets and only select the one that can will be used for loss calculation
            bbox_pred_flatten, class_pred_flatten, bbox_targets_flatten, class_targets_flatten = \
                self.flatten_and_select(bbox_pred, class_pred, bbox_targets, class_targets)

            class_loss = 0
            bbox_loss = 0
            class_loss += self.class_loss(class_pred_flatten, class_targets_flatten)
            bbox_loss += self.bbox_loss(bbox_pred_flatten, bbox_targets_flatten)
            return {'class_loss': class_loss, 'bbox_loss': bbox_loss}
        else:
            result = self.decode_pred(bbox_pred, class_pred, inputs['image_sizes'])
            return result

    def decode_pred(self, bbox_pred, class_pred, image_shape):
        batch_size = list(bbox_pred.values())[0].shape[0]
        device = list(bbox_pred.values())[0].device
        boxes_batch = [[] for i in range(batch_size)]
        labels_batch = [[] for i in range(batch_size)]
        scores_batch = [[] for i in range(batch_size)]
        for b_pred_layer, c_pred_layer, r_l, s in zip(bbox_pred.values(), class_pred.values(), self.r_l, self.strides):
            B, C, H, W = c_pred_layer.shape
            c_pred_layer = c_pred_layer.permute(0,2,3,1).reshape(batch_size,-1, self.class_num)
            b_pred_layer = b_pred_layer.permute(0,2,3,1).reshape(batch_size,-1, 4)
            #c_flatten = c_pred_layer.reshape(-1, self.class_num)
            _, pred_num, _ = c_pred_layer.shape
            pred_num_batch = pred_num * batch_size
            max_scores, inds = c_pred_layer.max(dim=2)
            inds_flatten = inds.flatten()+torch.arange(0,pred_num_batch*self.class_num, self.class_num, device=device)
            if max_scores.shape[-1] > self.nms_pre and self.nms_pre>0:
                scores_topk, topk_ind = max_scores.topk(self.nms_pre)
                topk_ind_flatten = (topk_ind+torch.arange(0,pred_num_batch,pred_num, device=device).view(-1,1)).flatten()
                out_inds = inds_flatten[topk_ind_flatten] // self.class_num
                out_cat = inds_flatten[topk_ind_flatten] % self.class_num
            else:
                scores_topk = max_scores
                out_inds = inds_flatten // self.class_num
                out_cat = inds_flatten % self.class_num
            out_cat = out_cat.reshape(batch_size,-1)
            b_pred_layer = b_pred_layer.reshape(-1,4)[out_inds]

            x_grid = out_inds % W
            y_grid = (out_inds // W) % H
            #z = out_inds // (W*H)
            x1 = (s*(x_grid+0.5) - r_l*torch.exp(b_pred_layer[:,0])).reshape(batch_size, -1)
            y1 = (s*(y_grid+0.5) - r_l*torch.exp(b_pred_layer[:,1])).reshape(batch_size, -1)
            x2 = (s*(x_grid+0.5) + r_l*torch.exp(b_pred_layer[:,2])).reshape(batch_size, -1)
            y2 = (s*(y_grid+0.5) + r_l*torch.exp(b_pred_layer[:,3])).reshape(batch_size, -1)
            for b in range(batch_size):
                x1[b].clamp(0, image_shape[b][1]-1)
                y1[b].clamp(0, image_shape[b][0]-1)
                x2[b].clamp(0, image_shape[b][1]-1)
                y2[b].clamp(0, image_shape[b][0]-1)
                boxes = torch.stack((x1[b], y1[b], x2[b], y2[b]), dim=-1)
                labels = out_cat[b]
                scores = scores_topk[b]
                boxes_batch[b].append(boxes)
                labels_batch[b].append(labels)
                scores_batch[b].append(scores)

        # Do NMS for each image for boxes from different layers
        boxes_batch = [torch.cat(boxes_im) for boxes_im in boxes_batch]
        labels_batch = [torch.cat(labels_im) for labels_im in labels_batch]
        scores_batch = [torch.cat(scores_im) for scores_im in scores_batch]
        keep_ind = [batched_nms(boxes, scores, labels, iou_threshold=0.5) for boxes, scores, labels in zip(boxes_batch, scores_batch, labels_batch)]
        boxes_batch = [boxes_im[keep] for boxes_im, keep in zip(boxes_batch, keep_ind)]
        scores_batch = [scores_im[keep] for scores_im, keep in zip(scores_batch, keep_ind)]
        labels_batch = [labels_im[keep]+1 for labels_im, keep in zip(labels_batch, keep_ind)]

        result = {}
        result['boxes'] = boxes_batch
        result['labels'] = labels_batch
        result['scores'] = scores_batch
        return result



                
    def flatten_and_select(self, bbox_pred, class_pred, bbox_targets, class_targets):
        bbox_pred_flatten = [b_pred.permute(0,2,3,1).reshape(-1,4) for b_pred in bbox_pred.values()]
        class_pred_flatten = [c_pred.permute(0,2,3,1).reshape(-1,self.class_num) for c_pred in class_pred.values()]
        bbox_pred_flatten = torch.cat(bbox_pred_flatten)
        class_pred_flatten = torch.cat(class_pred_flatten)
        bbox_target_flatten = torch.cat([b_t.reshape(-1,4) for b_t in bbox_targets])
        class_targets_flatten = torch.cat([c_t.flatten() for c_t in class_targets])
        pos_ind = class_targets_flatten > 0
        bbox_pred_flatten = bbox_pred_flatten[pos_ind]
        bbox_target_flatten = bbox_target_flatten[pos_ind]
        return bbox_pred_flatten, class_pred_flatten, bbox_target_flatten, class_targets_flatten



    @torch.no_grad()
    def multi_scale_target(self, targets, strides, bbox_pred_shape, class_pred_shape, device, dtype):
        meshgrids = self.get_meshgrid(class_pred_shape, device, dtype)
        bbox_targets = [torch.zeros((bbox_out_shape[0], bbox_out_shape[2], bbox_out_shape[3], 4), dtype=dtype, device=device) \
                                        for bbox_out_shape in bbox_pred_shape]
        #print([temp.shape for temp in bbox_targets])
        class_targets = [torch.zeros((class_out_shape[0], class_out_shape[2], class_out_shape[3]), dtype=dtype, device=device) \
                                        for class_out_shape in class_pred_shape]
        inds = [self.split_boxes_by_size(target['boxes']) for target in targets]

        batch_size = len(targets)
        layer_num = len(strides)

        for i in range(batch_size):
            for j in range(layer_num):
                ind = inds[i][j]
                boxes = targets[i]['boxes'][ind]
                labels = targets[i]['labels'][ind]
                if len(boxes) == 0:
                    continue
                bbox_targets[j][i], class_targets[j][i] = self.draw_single_map(boxes, labels, bbox_targets[j][i], class_targets[j][i], strides[j], meshgrids[j], self.r_l[j] )
        return bbox_targets, class_targets

    def get_meshgrid(self, class_pred_shape, device, dtype):
        meshgrids = []
        for _, _, h, w in class_pred_shape:
            x_range = torch.arange(w, dtype=dtype, device=device)+0.5
            y_range = torch.arange(h, dtype=dtype, device=device)+0.5
            y,x = torch.meshgrid(y_range, x_range)
            meshgrids.append((y,x))
        return meshgrids

    def draw_single_map(self, box, label, box_map, class_map, stride, meshgrids, r_l):
        # draw the bigger boxes first then small boxes
        ind = self.sort_by_area(box)
        box = box[ind]
        label = label[ind]

        box_new = box / stride
        cx = (box_new[:,0] + box_new[:,2]) / 2
        cy = (box_new[:,1] + box_new[:,3]) / 2
        w = box_new[:,2] - box_new[:,0]
        h = box_new[:,3] - box_new[:,1]

        map_h, map_w, _ = box_map.shape

        x1 = torch.ceil(cx-0.5*self.sigma*w).long().clamp(0,map_w-1)
        x2 = torch.floor(cx+0.5*self.sigma*w).long().clamp(0,map_w-1)
        y1 = torch.ceil(cy-0.5*self.sigma*h).long().clamp(0,map_h-1)
        y2 = torch.floor(cy+0.5*self.sigma*h).long().clamp(0,map_h-1)

        x1_gt, y1_gt, x2_gt, y2_gt = box[:,0], box[:,1], box[:,2], box[:,3]

        y_mesh, x_mesh = meshgrids
        for x11, x22, y11, y22, label1, bx1, by1, bx2, by2 in \
                zip(x1, x2, y1, y2, label, x1_gt, y1_gt, x2_gt, y2_gt):
            class_map[y11:y22+1, x11:x22+1] = label1
            box_map[ y11:y22+1, x11:x22+1, 0] = stride*x_mesh[y11:y22+1, x11:x22+1] - bx1
            box_map[ y11:y22+1, x11:x22+1, 1] = stride*y_mesh[y11:y22+1, x11:x22+1] - by1
            box_map[ y11:y22+1, x11:x22+1, 2] = bx2 - stride*x_mesh[y11:y22+1, x11:x22+1] 
            box_map[ y11:y22+1, x11:x22+1, 3] = by2 - stride*y_mesh[y11:y22+1, x11:x22+1]
        box_map = torch.clamp((box_map/r_l), 1/16, 16)
        box_map = torch.log(box_map)
        return box_map, class_map

    def sort_by_area(self, boxes):
        box_area = (boxes[:, 2]-boxes[:, 0]) * (boxes[:,3]-boxes[:,1])
        _, ind = torch.sort(-1*box_area)
        return ind
        

    @torch.no_grad()
    def split_boxes_by_size(self, boxes):
        sqrt_area = torch.sqrt_((boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0]))
        inds = [torch.logical_and(sqrt_area.ge_(x[0]), sqrt_area.le_(x[1])) for x in self.r_interval]
        return inds

    def set_r_interval(self):
        r = [2^x for x in range(5,10)]
        r_interval = [[x/self.eta, x*self.eta] for x in r]
        r_interval[0][0] = 1
        r_interval[-1][1] = 10000 # just a very big value
        self.r_l = r
        self.r_interval = r_interval


    def get_strides(self, inputs, features):
        if isinstance(features, dict):
            features = list(features.values())
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in features])
        image_size = inputs['data'].shape[-2:]
        strides = tuple(math.pow(2,math.ceil(math.log2((image_size[0] / g[0] + image_size[1] / g[1])/2))) for g in grid_sizes)
        return strides