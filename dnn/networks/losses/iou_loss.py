from mmdet.models.losses import IoULoss
from .build import LOSS_REG

LOSS_REG.register(IoULoss)