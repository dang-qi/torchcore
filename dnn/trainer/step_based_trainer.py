from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import torch
import tqdm
import math
import torch.distributed as dist
from ...mlblogapi.mlblogapi import MLBlogAPI

from .base_trainer import BaseTrainer
from .build import TRAINER_REG

@TRAINER_REG.register()
class StepBasedTrainer(BaseTrainer):
    def __init__(self, model, trainset, max_step, tag='', rank=0, world_size=1, log_print_iter=1000, log_save_iter=50, testset=None, optimizer=None, scheduler=None, clip_gradient=None, evaluator=None, accumulation_step=1, path_config=None, log_with_tensorboard=False, log_api_token=None, empty_cache_iter=None, log_memory=True, use_amp=False, ema_cfg=None, eval_step_interval=10000, save_step_interval=10000 ):
        super().__init__(model, trainset, tag=tag, rank=rank, world_size=world_size, log_print_iter=log_print_iter, log_save_iter=log_save_iter, testset=testset, optimizer=optimizer, scheduler=scheduler, clip_gradient=clip_gradient, evaluator=evaluator, accumulation_step=accumulation_step,path_config=path_config, log_with_tensorboard=log_with_tensorboard, log_api_token=log_api_token, empty_cache_iter=empty_cache_iter,
        log_memory=log_memory, use_amp=use_amp, ema_cfg=ema_cfg)
        self._max_step = max_step
        self._max_epoch = math.ceil(self._max_step/len(trainset))
        self._start_step=1
        self._end_step = max_step
        self.eval_step_interval = eval_step_interval
        self.save_step_interval = save_step_interval
        self._log_api_token = log_api_token
        self._iter_per_epoch = len(trainset)
        if log_api_token is not None:
            self.init_log_api()
    
    def train(self):
        self.train_step()

    def before_train(self):
        self._model.train()
        if hasattr(self, '_scheduler'): 
            if hasattr(self._scheduler, 'update_milestone_from_epoch_to_iter'):
                dataset_len = len(self._trainset)
                self._scheduler.update_milestone_from_epoch_to_iter(dataset_len)
            if hasattr(self._scheduler, 'update_iter_per_epoch'):
                dataset_len = len(self._trainset)
                print('dataloader length is {}. The lr scheduler is updated.'.format(dataset_len))
                self._scheduler.update_iter_per_epoch(dataset_len)
        self._restart_dataloader = False

    def after_train_epoch(self):
        if hasattr(self, '_scheduler'):
            self._scheduler.step_epoch(cur_iter=self._step, cur_epoch=self._epoch)

    def before_train_epoch(self):
        self.reset_trainset() # self._epoch +=1

    def is_start_epoch(self):
        return self._step % self._iter_per_epoch == 0

    def is_end_epoch(self):
        # it is the same with start since the self._step is updated after start_epoch
        return self._step % self._iter_per_epoch == 0

    def before_train_iter(self):
        self._step += 1

    def after_train_iter(self):
        if self._testset is not None and (self._step%self.eval_step_interval == 0 or self._step==self._end_step):
            if self.distributed:
                dist.barrier()
            #if not self.distributed or self.rank==0:
            if self.is_main_process():
                self.validate()
            if self.distributed:
                dist.barrier()

        if self.is_main_process():
            if self._step % self.save_step_interval == 0 or self._step==self._end_step:
                self.save_training(self._path_config.checkpoint_path_tmp.format('step_'+str(self._step)))

        if hasattr(self, '_scheduler'):
            self._scheduler.step_iter(cur_iter=self._step, cur_epoch=self._epoch)



    def train_step( self ):
        self.before_train()

        self._optimizer.zero_grad()
        for step in tqdm.tqdm(range(self._start_step, self._end_step+1),desc='Training', initial=self._start_step, dynamic_ncols=True):        
            if self.is_start_epoch():
                self.before_train_epoch()

            self.before_train_iter() # before_train_epoch is inside
            self._train_iter()
            self.after_train_iter()

            if self.is_end_epoch():
                self.after_train_epoch()

    def _train_iter( self ):
        if not self._model.training:
            self._model.train()

        data_start_time = time.time()
        #try:
        #    data = next(self.train_set_iter)
        #except StopIteration:
        data = next(self.train_set_iter)
        data_end_time = time.time()
        inputs, targets = data

        inputs = self._set_device( inputs )
        targets = self._set_device( targets )

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_dict = self._model(inputs, targets)

        loss_sum, loss_log = self._parse_loss_dict(loss_dict)
        #loss_sum=0
        #for single_loss in loss_dict:
        #    loss_sum = loss_sum + (loss_dict[single_loss]/self._accumulation_step)

        loss_sum_num = loss_sum.item()
        if not math.isfinite(loss_sum_num):
            #print("Loss is {}, stopping training".format(loss_sum))
            self._optimizer.zero_grad(set_to_none=True)
            print('wrong targets:',targets)
            print("Loss is {}, skip this batch".format(loss_sum_num))
            print(loss_dict)
            raise ValueError('wrong loss')
            return
            sys._exit(1)
        self.loss_logger.update(loss_log)

        # Computing gradient and do SGD step
        self._scaler.scale(loss_sum).backward()
        #loss_sum.backward()

        if self._clip_gradient is not None:
            self.clip_gradient()
            #torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_gradient)

        if self._step % self._accumulation_step == 0:
            self._scaler.step(self._optimizer)
            # Updates the scale for next iteration
            self._scaler.update()
            self._optimizer.zero_grad()

        iter_end_time = time.time()
        if self._empty_cache_iter is not None:
            if self._step % self._empty_cache_iter == 0:
                torch.cuda.empty_cache()

        if self._step % self._log_save_iter == 1:
            average_losses = self.loss_logger.get_last_average()
            self.loss_logger.clear()
            self.data_time = data_end_time-data_start_time
            self.iter_time = iter_end_time - data_end_time
            self.update_log('data_time', self.data_time)
            self.update_log('iter_time', self.iter_time)
            self.save_log(average_losses)
            if self._step % self._log_print_iter == 1:
                self.print_log(average_losses)
            
        if self.use_ema:
                self.ema.update(self._model)


        if hasattr(self, '_scheduler'):
            self._scheduler.step_iter(cur_iter=self._step, cur_epoch=self._epoch)


    def resume_training(self, path=None, new_lr=None, to_print=True):
        if path is None:
            path = self._path_config.checkpoint_path_tmp.format('last')
        device = self._device
        if isinstance(self._model, DDP):
            dist.barrier()
        checkpoint = torch.load(path, map_location=device)
        self._epoch = checkpoint['epoch']
        self._step = checkpoint['step']
        if isinstance(self._model, DDP):
            self._model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self._model.load_state_dict(checkpoint['model_state_dict'])
        # TODO: check here later
        ## use new_lr to update schduler state dict in case the linear lr is adapted
        #if new_lr is not None:
        #    self.update_optimizer_base_lr_dict(checkpoint['optimizer_state_dict'], new_lr)
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.move_optimizer_to_device(self._optimizer, device)
        #if new_lr is not None:
        #    self.update_scheduler_base_lr_dict(checkpoint['scheduler'], new_lr)
        #    #print('The base_lrs of scheduler is updated as {}'.format(checkpoint['scheduler']['base_lrs']))
        if 'scheduler' in checkpoint:
            self._scheduler.load_state_dict(checkpoint['scheduler'])
            #print('new lr is {}'.format(self._scheduler.base_lrs))

        if 'scaler' in checkpoint:
            self._scaler.load_state_dict(checkpoint['scaler'])
        if to_print:
            print('Chekpoint has been loaded from {}'.format(path))