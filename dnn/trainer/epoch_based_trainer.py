
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import tqdm
import math
import torch.distributed as dist
import numpy as np
import time
from datetime import datetime

from .base_trainer import BaseTrainer
from .build import TRAINER_REG
from ...dist.all_reduce_norm import all_reduce_norm

@TRAINER_REG.register()
class EpochBasedTrainer(BaseTrainer):
    def __init__(self, model, trainset, max_epoch, tag='', rank=0,world_size=1, log_print_iter=1000, log_save_iter=50, testset=None, optimizer=None, scheduler=None, clip_gradient=None, evaluator=None, accumulation_step=1, path_config=None, log_with_tensorboard=False, log_api_token=None, log_memory=True,use_amp=False, ema_cfg=None, eval_epoch_interval=1, save_epoch_interval=1, num_last_epoch=15):
        super().__init__(model, trainset, tag=tag, rank=rank, world_size=world_size, log_print_iter=log_print_iter, log_save_iter=log_save_iter, testset=testset, optimizer=optimizer, scheduler=scheduler, clip_gradient=clip_gradient, evaluator=evaluator, accumulation_step=accumulation_step, path_config=path_config, log_with_tensorboard=log_with_tensorboard, log_api_token=log_api_token,
        log_memory=log_memory, use_amp=use_amp, ema_cfg=ema_cfg)
        self._max_epoch = max_epoch
        self._max_step = max_epoch*len(trainset)
        self._start_step=0
        self._epoch = 0
        self._end_epoch = max_epoch
        self.eval_epoch_inteval = eval_epoch_interval
        self.save_epoch_interval = save_epoch_interval
        self._num_last_epoch = num_last_epoch
    
        if log_api_token is not None:
            self.init_log_api()

    def train(self):
        self.train_epoch()

    def before_train(self):
        self._model.train()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if hasattr(self, '_scheduler'): 
            if hasattr(self._scheduler, 'update_milestone_from_epoch_to_iter'):
                dataset_len = len(self._trainset)
                self._scheduler.update_milestone_from_epoch_to_iter(dataset_len)
            if hasattr(self._scheduler, 'update_iter_per_epoch'):
                dataset_len = len(self._trainset)
                print('dataloader length is {}. The lr scheduler is updated.'.format(dataset_len))
                self._scheduler.update_iter_per_epoch(dataset_len)
        self._restart_dataloader = False

    def before_train_epoch(self):
        self.reset_trainset() # self._epoch += 1
        if self.is_main_process():
            print("{} Epoch {}/{}".format(datetime.now(),self._epoch,self._max_epoch))
        # TODO fix it with nicer way
        if self._epoch==self._max_epoch-self._num_last_epoch:
            self._model.use_l1 = True
            if hasattr(self._trainset.dataset,'update_ignore_transform_keys'):
                self._trainset.dataset.update_ignore_transform_keys(['Mosaic', 'RandomAffine', 'MixUp'])
                print('data aug is disabled')
            if hasattr(self._trainset, 'persistent_workers'
                       ) and self._trainset.persistent_workers is True:
                self._trainset._DataLoader__initialized = False
                self._trainset._iterator = None
                self._restart_dataloader = True
                print('stop resistent worker')
            self.eval_epoch_inteval = 100
        else:
            if self._restart_dataloader:
                self._trainset._DataLoader__initialized = True
                self._restart_dataloader = False
                print('resume resistent worker')

    def after_train_epoch(self):
        if hasattr(self, '_scheduler'):
            self._scheduler.step_epoch(cur_iter=self._step, cur_epoch=self._epoch)
        if self.is_main_process():
            self.save_last_training(self._path_config.checkpoint_path_tmp.format('epoch_'+str(self._epoch)))

        all_reduce_norm(self._model)
        
        if self._testset is not None and self._epoch%self.eval_epoch_inteval == 0 :
            if self.distributed:
                dist.barrier()
            #if not self.distributed or self.rank==0:
            if self.is_main_process():
                self.validate()
            if self.distributed:
                dist.barrier()
        if self._epoch%self.save_epoch_interval == 0:
            if self.is_main_process():
                self.save_training(self._path_config.checkpoint_path_tmp.format('epoch_'+str(self._epoch)))


    def train_epoch( self ):
        self.before_train()

        for i in range( self._epoch+1, self._max_epoch+1 ):
            self.before_train_epoch()
            self._train_epoch()
            self.after_train_epoch()

    def _train_epoch( self ):
        if not self._model.training:
            self._model.train()

        self._optimizer.zero_grad()
        data_iter = iter(self._trainset)
        for idx in tqdm.tqdm(range(len(self._trainset)), desc='Training',dynamic_ncols=True):
            iter_start_time = time.time()
            inputs, targets = next(data_iter)
            self._step += 1

            inputs = self._set_device( inputs )
            targets = self._set_device( targets )
            data_end_time = time.time()
            #print('input device:', inputs[0].device)
            self.before_train_iter()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss_dict = self._model(inputs, targets)

            loss_sum, loss_log = self._parse_loss_dict(loss_dict)

            loss_sum_num = loss_sum.item()
            if not math.isfinite(loss_sum):
                self._optimizer.zero_grad(set_to_none=True)
                print('wrong targets:',targets)
                print("Loss is {}, skip this batch".format(loss_sum_num))
                print(loss_dict)
                continue
                #sys.exit(1)
            self.loss_logger.update(loss_log)

            # Computing gradient and do SGD step
            # Scales the loss, and calls backward()
            # to create scaled gradients
            self.scaler.scale(loss_sum).backward()

            if self._clip_gradient is not None:
                self.clip_gradient()
                #torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_gradient)

            if idx % self._accumulation_step == 0:
                # Unscales gradients and calls
                # or skips optimizer.step()
                self.scaler.step(self._optimizer)
                # Updates the scale for next iteration
                self.scaler.update()
                self._optimizer.zero_grad()
            #loss_values.append( loss_sum.cpu().detach().numpy() )
            iter_end_time = time.time()

            #if idx%self._log_print_iter == 0:
                #if self.rank == 0 or not self.distributed:
            if self._step % self._log_save_iter == 1:
                average_losses = self.loss_logger.get_last_average()
                self.loss_logger.clear()
                self.data_time = data_end_time-iter_start_time
                self.iter_time = iter_end_time-data_end_time
                self.update_log('data_time', self.data_time)
                self.update_log('iter_time', self.iter_time)
                self.save_log(average_losses)
                if self._step % self._log_print_iter == 1:
                    self.print_log(average_losses)

            if self.use_ema:
                self.ema.update(self._model)


            if hasattr(self, '_scheduler'):
                #print('step: {}, epoch: {}'.format(self._step, self._epoch))
                self._scheduler.step_iter(cur_iter=self._step, cur_epoch=self._epoch)
            #if idx > 20:
            #    break
            


    def resume_training(self, path, device, to_print=True):
        if isinstance(self._model, DDP):
            dist.barrier()
        checkpoint = torch.load(path, map_location=device)
        self._epoch = checkpoint['epoch']
        self._step = checkpoint['step']
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.move_optimizer_to_device(self._optimizer, device)
        if 'scheduler' in checkpoint:
            self._scheduler.load_state_dict(checkpoint['scheduler'])
        if to_print:
            print('Chekpoint has been loaded from {}'.format(path))