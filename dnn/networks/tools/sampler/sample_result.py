class SampleResult():
    def __init__(self,
                 pos_ind,
                 neg_ind,
                 gt_boxes,
                 anchor_boxes,
                 match_result,
                 ) -> None:
        self.pos_ind = pos_ind
        self.neg_ind = neg_ind
        self.pos_boxes = anchor_boxes[pos_ind]
        self.neg_boxes = anchor_boxes[neg_ind]
        gt_box_ind = match_result.matched_ind[pos_ind]
        self.pos_gt_boxes = gt_boxes[gt_box_ind]
        if match_result.labels is not None:
            self.pos_labels = match_result.labels[pos_ind]
        else:
            self.pos_labels = None
