from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import tqdm
import math
import torch.distributed as dist

from .base_trainer import BaseTrainer
from .build import TRAINER_REG

@TRAINER_REG.register()
class StepBasedTrainer(BaseTrainer):
    def __init__(self, model, trainset, max_step, tag='', rank=0, log_print_iter=1000, log_save_iter=50, testset=None, optimizer=None, scheduler=None, clip_gradient=None, evaluator=None, accumulation_step=1, path_config=None, log_with_tensorboard=False, eval_step_interval=10000, save_step_interval=10000 ):
        super().__init__(model, trainset, tag=tag, rank=rank, log_print_iter=log_print_iter, log_save_iter=log_save_iter, testset=testset, optimizer=optimizer, scheduler=scheduler, clip_gradient=clip_gradient, evaluator=evaluator, accumulation_step=accumulation_step,path_config=path_config, log_with_tensorboard=log_with_tensorboard)
        self._max_step = max_step
        self._start_step=1
        self._end_step = max_step
        self.eval_step_interval = eval_step_interval
        self.save_step_interval = save_step_interval
    
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

        # add the losses for each part
        #loss_sum = sum(loss for loss in loss_dict.values())
        loss_sum=0
        for single_loss in loss_dict:
            loss_sum = loss_sum + (loss_dict[single_loss]/self._accumulation_step)

        loss_sum_num = loss_sum.item()
        if not math.isfinite(loss_sum_num):
            #print("Loss is {}, stopping training".format(loss_sum))
            self._optimizer.zero_grad(set_to_none=True)
            print('wrong targets:',targets)
            print("Loss is {}, skip this batch".format(loss_sum_num))
            print(loss_dict)
            return
            sys._exit(1)
        self.loss_logger.update(loss_dict)

        # Computing gradient and do SGD step
        loss_sum.backward()

        if self._clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_gradient)

        if self._step % self._accumulation_step == 0:
            self._optimizer.step()
            self._optimizer.zero_grad()

        if self.is_main_process():
            if self._step % self.log_print_iter == 1:
                if not self.log_with_tensorboard:
                    self._logger.info('{} '.format(self._step))
                loss_str = ''
                average_losses = self.loss_logger.get_last_average()
                for loss in average_losses:
                    loss_num = average_losses[loss]
                    if not self.log_with_tensorboard:
                        self._logger.info('{} '.format(loss_num))
                    else:
                        self._logger.add_scalars('loss',{})
                    loss_str += (' {} loss:{}, '.format(loss, loss_num))
                print(loss_str[:-2])
                average_losses = {}
                if not self.log_with_tensorboard:
                    self._logger.info('\n')
            
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
            print('Chekpoint has been loaded from {}'.format(path))