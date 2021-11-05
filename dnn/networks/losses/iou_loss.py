from mmdet.models.losses import IoULoss, GIoULoss
from .build import LOSS_REG

LOSS_REG.register(IoULoss)
LOSS_REG.register(GIoULoss)