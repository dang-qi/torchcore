from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import tqdm
import math
import torch.distributed as dist
from ...mlblogapi.mlblogapi import MLBlogAPI

from .base_trainer import BaseTrainer
from .build import TRAINER_REG

@TRAINER_REG.register()
class StepBasedTrainer(BaseTrainer):
    def __init__(self, model, trainset, max_step, tag='', rank=0, world_size=1, log_print_iter=1000, log_save_iter=50, testset=None, optimizer=None, scheduler=None, clip_gradient=None, evaluator=None, accumulation_step=1, path_config=None, log_with_tensorboard=False, log_api_token=None, empty_cache_iter=None, log_memory=True, eval_step_interval=10000, save_step_interval=10000 ):
        super().__init__(model, trainset, tag=tag, rank=rank, world_size=world_size, log_print_iter=log_print_iter, log_save_iter=log_save_iter, testset=testset, optimizer=optimizer, scheduler=scheduler, clip_gradient=clip_gradient, evaluator=evaluator, accumulation_step=accumulation_step,path_config=path_config, log_with_tensorboard=log_with_tensorboard, log_api_token=log_api_token, empty_cache_iter=empty_cache_iter,
        log_memory=log_memory)
        self._max_step = max_step
        self._max_epoch = math.ceil(self._max_step/len(trainset))
        self._start_step=1
        self._end_step = max_step
        self.eval_step_interval = eval_step_interval
        self.save_step_interval = save_step_interval
        self._log_api_token = log_api_token
        if log_api_token is not None:
            self.init_log_api()
    
    def train(self):
        self.train_step()



    def train_step( self ):
        self._model.train()
        #if hasattr(self._cfg,'freeze_bn'):
        #    if self._cfg.freeze_bn:
        #        if isinstance(self._model, DDP):
        #            self._model.module.freeze_bn()
        #        else:
        #            self._model.freeze_bn()

        self._optimizer.zero_grad()
        for step in tqdm.tqdm(range(self._start_step, self._end_step+1),desc='Training', initial=self._start_step, dynamic_ncols=True):
            self._step = step
            self._train_step()

            if self._testset is not None and step%self.eval_step_interval == 0 :
                if self.distributed:
                    dist.barrier()
                #if not self.distributed or self.rank==0:
                if self.is_main_process():
                    self.validate()
                if self.distributed:
                    dist.barrier()

            if self.is_main_process():
                if step % self.save_step_interval == 0 or step==self._end_step:
                    self.save_training(self._path_config.checkpoint_path_tmp.format('step_'+str(self._step)))

            if hasattr(self, '_scheduler'):
                self._scheduler.step()

    def _train_step( self ):
        if not self._model.training:
            self._model.train()
            #if hasattr(self._cfg,'freeze_bn'):
            #    if self._cfg.freeze_bn:
            #        if isinstance(self._model, DDP):
            #            self._model.module.freeze_bn()
            #        else:
            #            self._model.freeze_bn()

        try:
            data = next(self.train_set_iter)
        except StopIteration:
            self.reset_trainset()
            data = next(self.train_set_iter)
        inputs, targets = data

        inputs = self._set_device( inputs )
        targets = self._set_device( targets )

        loss_dict = self._model(inputs, targets)
        #try:
        #    loss_dict = self._model(inputs, targets)
        #except RuntimeError as e1:
        #    if 'out of memory' in str(e1):
        #        print('out of memory 11111')
        #        torch.cuda.empty_cache()
        #        print('torch.cuda.empty_cache(),then try again')
        #        try:
        #            loss_dict = self._model(inputs, targets)
        #        except RuntimeError as e2:
        #            if 'out of memory' in str(e2):
        #                print('out of memory 22222')
        #                print('len(targets)',len(targets))
        #                print('skip this iteration')
        #                for target in targets:
        #                    print(target)
        #            else:
        #                raise e2
        #    else:
        #        raise e1

        # add the losses for each part
        #loss_sum = sum(loss for loss in loss_dict.values())
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
            return
            sys._exit(1)
        self.loss_logger.update(loss_log)

        # Computing gradient and do SGD step
        loss_sum.backward()

        if self._clip_gradient is not None:
            self.clip_gradient()
            #torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_gradient)

        if self._step % self._accumulation_step == 0:
            self._optimizer.step()
            self._optimizer.zero_grad()

        if self._empty_cache_iter is not None:
            if self._step % self._empty_cache_iter == 0:
                torch.cuda.empty_cache()

        if self._step % self._log_save_iter == 0:
            average_losses = self.loss_logger.get_last_average()
            self.loss_logger.clear()
            self.save_log(average_losses)
            if self._step % self._log_print_iter == 0:
                self.print_log(average_losses)
            
    #def save_training(self, path, to_print=True):
    #    if isinstance(self._model, DDP):
    #        if self.is_main_process():
    #            state_dict = self._model.state_dict()
    #        else:
    #            return
    #    elif isinstance(self._model, torch.nn.DataParallel):
    #        state_dict = self._model.module.state_dict()
    #    else:
    #        state_dict =self._model.state_dict()
    #    torch.save({
    #        'step': self._step,
    #        'model_state_dict': state_dict,
    #        'optimizer_state_dict': self._optimizer.state_dict(),
    #        'scheduler':self._scheduler.state_dict(),
    #    }, path)
    #    folder = os.path.dirname(path)
    #    if self._path_config is not None:
    #        last_path = self._path_config.checkpoint_path_tmp.format('last')
    #    else:
    #        last_path = os.path.join(folder,'{}_last.pth'.format(self._tag))
    #    torch.save({
    #        'step': self._step,
    #        'model_state_dict': state_dict,
    #        'optimizer_state_dict': self._optimizer.state_dict(),
    #        'scheduler':self._scheduler.state_dict(),
    #    }, last_path)

    #    if to_print:
    #        print('The checkpoint has been saved to {}'.format(path))

    def resume_training(self, path=None, to_print=True):
        if isinstance(self._model, DDP):
            dist.barrier()
        device =self._device
        if path is None:
            path=self._path_config.checkpoint_path_tmp.format('last')
        checkpoint = torch.load(path, map_location=device)
        self._start_step = checkpoint['step']
        self._epoch = checkpoint['epoch']
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.move_optimizer_to_device(self._optimizer, device)
        if 'scheduler' in checkpoint:
            self._scheduler.load_state_dict(checkpoint['scheduler'])
        if to_print:
            print('Chekpoint has been loaded from {} at epoch{} and step{}'.format(path, self._epoch, self._start_step))