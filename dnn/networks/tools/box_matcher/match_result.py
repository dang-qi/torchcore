class MatchResult():
    NEGATIVE_MATCH = -2
    IGNORE_MATCH = -1
    def __init__(self, gt_nums, matched_ind, max_iou, labels=None) -> None:
        '''
            gt_nums: ground truth box number
            matched_ind: matched ind of anchor boxes, -2 means negative match, -1 means ignored match, others are 0-based index of the matched gt boxes
            max_iou: the max iou of the gt boxes
            labels: the category label of assigned anchor box
        '''
        self.gt_nums = gt_nums
        self.matched_ind = matched_ind
        self.max_iou = max_iou
        self.labels = labels
