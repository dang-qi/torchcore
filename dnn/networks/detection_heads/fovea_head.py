import torch
import math
from torch import nn
from torch.functional import meshgrid
from ..heads import MultiConvHead

class FoveaHead(nn.Module):
    def __init__(self, class_num, in_feature_num, sigma=0.4, eta=2):
        super().__init__()
        self.strides = None
        self.sigma = sigma
        self.eta = eta
        self.class_num = class_num
        self.cls_head = MultiConvHead(out_channel=class_num, in_channel=in_feature_num, head_conv_channel=256, middle_layer_num=4 )
        self.box_head = MultiConvHead(out_channel=4, in_channel=in_feature_num, head_conv_channel=256, middle_layer_num=4 )
        self.set_r_interval()
        

    def forward(self, features, inputs, targets=None):
        assert isinstance(features, dict)
        for k,v in features.items():
            print(k, v.shape)
        if self.strides is None:
            self.strides = self.get_strides(inputs, features)
        
        bbox_pred = {}
        class_pred = {}
        for k, v in features.items():
            bbox_pred[k] = self.box_head(v)
            class_pred[k] = self.cls_head(v)
            
        if self.training:
            bbox_pred_shape = [v.shape for v in bbox_pred.values()]
            class_pred_shape = [v.shape[-2:] for v in class_pred.values()]
            device = list(bbox_pred)[0].device
            dtype = list(bbox_pred)[0].dtype
            bbox_targets, class_targets = self.multi_scale_target(targets, self.strides, bbox_pred_shape, class_pred_shape, device, dtype)


    @torch.no_grad()
    def multi_scale_target(self, targets, strides, bbox_pred_shape, class_pred_shape, device, dtype):
        meshgrids = self.get_meshgrid(class_pred_shape, device, dtype)
        bbox_targets = [torch.zeros(bbox_out_shape, dtype=dtype, device=device) \
                                        for bbox_out_shape in bbox_pred_shape]
        class_targets = [torch.zeros(class_out_shape, dtype=dtype, device=device) \
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
        for h, w in class_pred_shape:
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

        _, map_h, map_w = box_map.shape

        x1 = torch.ceil(cx-0.5*self.sigma*w).long().clamp(0,map_w-1)
        x2 = torch.floor(cx+0.5*self.sigma*w).long().clamp(0,map_w-1)
        y1 = torch.ceil(cy-0.5*self.sigma*h).long().clamp(0,map_h-1)
        y2 = torch.floor(cy+0.5*self.sigma*h).long().clamp(0,map_h-1)

        x1_gt, y1_gt, x2_gt, y2_gt = box[:,0], box[:,1], box[:,2], box[:,3]

        for x11, x22, y11, y22, label1, bx1, by1, bx2, by2, (y_mesh, x_mesh) in \
                zip(x1, x2, y1, y2, label, x1_gt, y1_gt, x2_gt, y2_gt, meshgrids):
            class_map[y11:y22+1, x11:x22+1] = label1
            box_map[0, y11:y22+1, x11:x22+1] = stride*x_mesh[y11:y22+1, x11:x22+1] - bx1
            box_map[1, y11:y22+1, x11:x22+1] = stride*y_mesh[y11:y22+1, x11:x22+1] - by1
            box_map[2, y11:y22+1, x11:x22+1] = bx2 - stride*x_mesh[y11:y22+1, x11:x22+1] 
            box_map[3, y11:y22+1, x11:x22+1] = by2 - stride*y_mesh[y11:y22+1, x11:x22+1]
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
        inds = [sqrt_area>=x[0] & sqrt_area<=x[1] for x in self.r_interval]
        return inds

    def set_r_interval(self):
        r = [2^x for x in range(5,10)]
        r_interval = [[x/self.eta, x*self.eta] for x in r]
        r_interval[0][0] = 0
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