import math
import torch
from .anchor_box_coder import AnchorBoxesCoder

class BBoxesCoder(AnchorBoxesCoder):
    def __init__(self, box_code_clip=math.log(1000. / 16), weight=[1.0, 1.0, 1.0, 1.0]):
        self.box_code_clip = box_code_clip
        self.weight = weight

    def decode(self, codes, anchors):
        # codes shape N * HxWxA * 4 (tx, ty, tw, th)
        # anchors shape N * HxWxA * 4 (x1, y1, x2, y2)
        codes_cat = codes.view(-1, 4)
        anchors_cat = torch.cat(anchors, dim=0)
        #print('code cat shape:', codes_cat.shape)
        #print('anchor cat shape:',anchors_cat.shape)
        boxes = self.decode_once(codes_cat, anchors_cat)
        return boxes

    def decode_once(self, code, anchor):
        wx = self.weight[0]
        wy = self.weight[1]
        ww = self.weight[2]
        wh = self.weight[3]

        anchor_width = anchor[...,2] - anchor[..., 0]
        anchor_height = anchor[...,3] - anchor[..., 1]
        anchor_xc = anchor[..., 0] + anchor_width / 2
        anchor_yc = anchor[..., 1] + anchor_height / 2

        code_x = code[...,0] / wx
        code_y = code[...,1] / wy
        code_w = code[...,2] / ww
        code_h = code[...,3] / wh

        if self.box_code_clip is not None:
            code_w = code_w.clamp_max(self.box_code_clip)
            code_h = code_h.clamp_max(self.box_code_clip)

        xc = code_x * anchor_width + anchor_xc
        yc = code_y * anchor_height + anchor_yc
        width = anchor_width * torch.exp(code_w)
        height = anchor_height * torch.exp(code_h)

        x1 = xc - width / 2
        y1 = yc - height / 2
        x2 = xc + width / 2
        y2 = yc + height / 2

        boxes = torch.stack((x1, y1, x2, y2), dim=1)
            #pred_boxes.append(boxes)
        return boxes

    def encode(self, pred_boxes, pos_boxes):
        out = []
        for anchor, target_box in zip(pred_boxes, pos_boxes):
            out.append(self.encode_once(anchor, target_box))

        return out

    def encode_once(self, pos_anchors, pos_boxes ):
        wx = self.weight[0]
        wy = self.weight[1]
        ww = self.weight[2]
        wh = self.weight[3]

        w_anchor = pos_anchors[...,2] - pos_anchors[...,0]
        h_anchor = pos_anchors[...,3] - pos_anchors[...,1]
        xc_anchor = pos_anchors[..., 0] + w_anchor*0.5
        yc_anchor = pos_anchors[..., 1] + h_anchor*0.5

        w_boxes = pos_boxes[...,2] - pos_boxes[...,0]
        h_boxes = pos_boxes[...,3] - pos_boxes[...,1]
        xc_boxes = pos_boxes[..., 0] + w_boxes*0.5
        yc_boxes = pos_boxes[..., 1] + h_boxes*0.5

        tx = wx * ( xc_boxes - xc_anchor) / w_anchor
        ty = wy * ( yc_boxes - yc_anchor) / h_anchor
        tw = ww * torch.log(w_boxes / w_anchor)
        th = wh * torch.log(h_boxes / h_anchor)

        targets = torch.stack((tx,ty,tw,th), dim=1)
        return targets




