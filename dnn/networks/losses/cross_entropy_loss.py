from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from .build import LOSS_REG

LOSS_REG.register(CrossEntropyLoss, 'CrossEntropyLoss')
LOSS_REG.register(BCEWithLogitsLoss, 'BCEWithLogitsLoss')