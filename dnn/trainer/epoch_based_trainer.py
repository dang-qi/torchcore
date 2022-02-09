
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import tqdm
import math
import torch.distributed as dist
import numpy as np

from .base_trainer import BaseTrainer
from .build import TRAINER_REG

@TRAINER_REG.register()
class EpochBasedTrainer(BaseTrainer):
    def __init__(self, model, trainset, max_epoch, tag='', rank=0,world_size=1, log_print_iter=1000, log_save_iter=50, testset=None, optimizer=None, scheduler=None, clip_gradient=None, evaluator=None, accumulation_step=1, path_config=None, log_with_tensorboard=False, log_api_token=None, log_memory=True, eval_epoch_interval=1, save_epoch_interval=1):
        super().__init__(model, trainset, tag=tag, rank=rank, world_size=world_size, log_print_iter=log_print_iter, log_save_iter=log_save_iter, testset=testset, optimizer=optimizer, scheduler=scheduler, clip_gradient=clip_gradient, evaluator=evaluator, accumulation_step=accumulation_step, path_config=path_config, log_with_tensorboard=log_with_tensorboard, log_api_token=log_api_token,
        log_memory=log_memory)
        self._max_epoch = max_epoch
        self._max_step = max_epoch*len(trainset)
        self._start_step=0
        self._epoch = 0
        self._end_epoch = max_epoch
        self.eval_epoch_inteval = eval_epoch_interval
        self.save_step_interval = save_epoch_interval
    
        if log_api_token is not None:
            self.init_log_api()

    def train(self):
        self.train_epoch()

    def train_epoch( self ):
        #if self._testset is not None :
        #    self._validate()
        self._model.train()
        #if hasattr(self._cfg,'freeze_bn'):
        #    if self._cfg.freeze_bn:
        #        if isinstance(self._model, DDP):
        #            self._model.module.freeze_bn()
        #        else:
        #            self._model.freeze_bn()
        if hasattr(self, '_scheduler') and hasattr(self._scheduler, 'update_milestone_from_epoch_to_iter'):
            dataset_len = len(self._trainset)
            self._scheduler.update_milestone_from_epoch_to_iter(dataset_len)


        for i in range( self._epoch+1, self._max_epoch+1 ):
            #if self._train_sampler is not None:
            #    self._train_sampler.set_epoch(i)
            self.reset_trainset()
            if self.is_main_process():
                print("Epoch %d/%d" % (i,self._max_epoch))
            #self._model.train()
            self._train_epoch()

            if self._testset is not None and i%1 == 0 :
                if self.distributed:
                    dist.barrier()
                #if not self.distributed or self.rank==0:
                if self.is_main_process():
                    self.validate()
                if self.distributed:
                    dist.barrier()
            self._epoch = i
            if self.is_main_process():
                self.save_training(self._path_config.checkpoint_path_tmp.format('epoch_'+str(self._epoch)))

    def _train_epoch( self ):
        if not self._model.training:
            self._model.train()
            #if hasattr(self._cfg,'freeze_bn'):
            #    if self._cfg.freeze_bn:
            #        if isinstance(self._model, DDP):
            #            self._model.module.freeze_bn()
            #        else:
            #            self._model.freeze_bn()

        self._optimizer.zero_grad()
        for idx, (inputs, targets) in enumerate(tqdm.tqdm(self._trainset, desc='Training',dynamic_ncols=True)):
            #print('inputs:', inputs)
            #print('targets:', targets)
            self._step += 1

            inputs = self._set_device( inputs )
            targets = self._set_device( targets )
            #print('input device:', inputs[0].device)

            loss_dict = self._model(inputs, targets)

            loss_sum, loss_log = self._parse_loss_dict(loss_dict)
            # add the losses for each part
            #loss_sum = sum(loss for loss in loss_dict.values())
            #loss_sum=0
            #for single_loss in loss_dict:
            #    loss_sum = loss_sum + (loss_dict[single_loss]/self._accumulation_step)
            #    #if self.rank==0 or not self.distributed:

            loss_sum_num = loss_sum.item()
            if not math.isfinite(loss_sum):
                #print("Loss is {}, stopping training".format(loss_sum))
                self._optimizer.zero_grad(set_to_none=True)
                print('wrong targets:',targets)
                print("Loss is {}, skip this batch".format(loss_sum_num))
                print(loss_dict)
                continue
                #sys.exit(1)
            self.loss_logger.update(loss_log)

            # Computing gradient and do SGD step
            loss_sum.backward()

            if self._clip_gradient is not None:
                self.clip_gradient()
                #torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_gradient)

            if idx % self._accumulation_step == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()
            #loss_values.append( loss_sum.cpu().detach().numpy() )

            #if idx%self._log_print_iter == 0:
                #if self.rank == 0 or not self.distributed:
            if self._step % self._log_save_iter == 1:
                average_losses = self.loss_logger.get_last_average()
                self.loss_logger.clear()
                self.save_log(average_losses)
                if self._step % self._log_print_iter == 1:
                    self.print_log(average_losses)

            if hasattr(self, '_scheduler'):
                self._scheduler.step()
            #if idx > 20:
            #    break
            
        #print('Average loss : ', np.mean(loss_values))

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
    #        'epoch': self._epoch,
    #        'step': self._step,
    #        'model_state_dict': state_dict,
    #        'optimizer_state_dict': self._optimizer.state_dict(),
    #        'scheduler':self._scheduler.state_dict(),
    #    }, path)

    #    folder = os.path.dirname(path)
    #    if self._path_config is not None:
    #        last_name = self._path_config.checkpoint_path_tmp.format('last')
    #    else:
    #        last_name = '{}_last.pth'.format(self._tag)
    #    torch.save({
    #        'epoch': self._epoch,
    #        'step': self._step,
    #        'model_state_dict': state_dict,
    #        'optimizer_state_dict': self._optimizer.state_dict(),
    #        'scheduler':self._scheduler.state_dict(),
    #    }, os.path.join(folder, last_name))

    #    if to_print:
    #        print('The checkpoint has been saved to {}'.format(path))

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