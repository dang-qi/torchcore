import torch
import numpy as np
import clip
from .build import COLLATE_REG

@COLLATE_REG.register()
class CollateHMClassification():
    def __init__(self) -> None:
        pass

    def __call__(self, batch):
        inputs, targets = [list(s) for s in zip(*batch)]
        ims = torch.tensor(np.stack([im['data'] for im in inputs]))
        label_text = [i['label_text'] for i in inputs]
        description = [i['description'] for i in inputs]
        label_num = [i['label_num'] for i in inputs]
        im_id = [i['id'] for i in inputs]
        inputs = {}
        inputs['data'] = ims
        inputs['label_num'] = label_num
        inputs['label_text'] = clip.tokenize(label_text)
        inputs['description'] = clip.tokenize(description,truncate=True)
        inputs['id'] = im_id

        return inputs, targets