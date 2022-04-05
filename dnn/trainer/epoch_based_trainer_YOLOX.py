
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import tqdm
import math
import torch.distributed as dist
import numpy as np
import time
from datetime import datetime

from .epoch_based_trainer import EpochBasedTrainer
from .build import TRAINER_REG
from ...dist.all_reduce_norm import all_reduce_norm

@TRAINER_REG.register()
class YOLOXEpochBasedTrainer(EpochBasedTrainer):
    def __init__(self, model, trainset, max_epoch, tag='', rank=0,world_size=1, log_print_iter=1000, log_save_iter=50, testset=None, optimizer=None, scheduler=None, clip_gradient=None, evaluator=None, accumulation_step=1, path_config=None, log_with_tensorboard=False, log_api_token=None, log_memory=True,use_amp=False, ema_cfg=None, eval_epoch_interval=1, save_epoch_interval=1, num_last_epoch=15):
        super().__init__(model, trainset,max_epoch=max_epoch, tag=tag, rank=rank, world_size=world_size, log_print_iter=log_print_iter, log_save_iter=log_save_iter, testset=testset, optimizer=optimizer, scheduler=scheduler, clip_gradient=clip_gradient, evaluator=evaluator, accumulation_step=accumulation_step, path_config=path_config, log_with_tensorboard=log_with_tensorboard, log_api_token=log_api_token,
        log_memory=log_memory, use_amp=use_amp, ema_cfg=ema_cfg,eval_epoch_interval=eval_epoch_interval, save_epoch_interval=save_epoch_interval)


        self._num_last_epoch = num_last_epoch
    
        if log_api_token is not None:
            self.init_log_api()


    def before_train_epoch(self):
        self.reset_trainset() # self._epoch += 1
        if self.is_main_process():
            print("{} Epoch {}/{}".format(datetime.now(),self._epoch,self._max_epoch))
        # TODO fix it with nicer way
        if self._epoch==self._max_epoch-self._num_last_epoch:
            if isinstance(self._model, DDP):
                self._model.module.det_head.use_l1 = True
            else:
                self._model.det_head.use_l1 = True

            if hasattr(self._trainset.dataset,'update_ignore_transform_keys'):
                self._trainset.dataset.update_ignore_transform_keys(['Mosaic', 'RandomAffine', 'MixUp'])
                print('data aug is disabled')
            if hasattr(self._trainset, 'persistent_workers'
                       ) and self._trainset.persistent_workers is True:
                self._trainset._DataLoader__initialized = False
                self._trainset._iterator = None
                self._restart_dataloader = True
                print('stop resistent worker')
            self.eval_epoch_inteval = 1
        else:
            if self._restart_dataloader:
                self._trainset._DataLoader__initialized = True
                self._restart_dataloader = False
                print('resume resistent worker')
