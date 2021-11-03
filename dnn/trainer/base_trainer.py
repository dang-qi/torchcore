
import torch
import numpy as np
import os
import datetime
import copy
import json
import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from ...util.logging import LossLogger, Logger

from ..optimizer.build import build_optimizer, build_lr_scheduler

from ...evaluation.build import build_evaluator

from ...data.datasets.build import build_dataloader


class BaseTrainer :
    def __init__( self, model, trainset, tag='', rank=0, log_print_iter=1000, log_save_iter=50, testset=None, optimizer=None, scheduler=None, clip_gradient=None, evaluator=None, accumulation_step=1, path_config=None, log_with_tensorboard=False ):
        device = torch.device( rank )
        self._device = device
        self._model=model
        self._optimizer = build_optimizer(model, optimizer)
        self._scheduler = build_lr_scheduler(self._optimizer, scheduler)
        self._tag = tag
        if isinstance(trainset,dict):
            self._trainset = build_dataloader(trainset)
        else:
            self._trainset = trainset
        if isinstance(testset,dict):
            self._testset = build_dataloader(testset)
        else:
            self._testset = testset
        self._accumulation_step=accumulation_step
        self.rank = rank
        self._path_config=path_config

        if isinstance(self._model, DDP):
            self.distributed = True
        else:
            self.distributed = False
        self._log_with_tensorboard=log_with_tensorboard

        self._epoch = 0
        self._step = 0

        self._log_print_iter = log_print_iter
        self._log_save_iter = log_save_iter

        self.loss_logger = LossLogger()

        if trainset is not None:
            self.train_set_iter = iter(self._trainset)
        self._clip_gradient = clip_gradient
        if evaluator is not None:
            self.evaluator = build_evaluator(evaluator)

        self._set_optimizer()
        self._set_scheduler()
        #if cfg.resume:
        #    path = os.path.join(os.path.dirname(cfg.path.CHECKPOINT), '{}_last.pth'.format(self._tag)) 
        #    self.resume_training(path, device)

        if self.is_main_process():
            self.init_logger()
        

    def _set_optimizer( self ):
        if isinstance(self._optimizer, dict):
            optimizer_cfg = self._optimizer.copy()
            self._optimizer = build_optimizer(optimizer_cfg)
        elif not isinstance(self._optimizer, Optimizer):
            raise ValueError('optimizer must be dict or torch.optim.Optimizer')

    def _set_scheduler(self):
        if isinstance(self._scheduler, dict):
            scheduler_cfg = self._scheduler.copy()
            scheduler_cfg = copy.deepcopy(self._scheduler)
            self._scheduler = build_lr_scheduler(self._optimizer, scheduler_cfg)
        elif not isinstance(self._scheduler, _LRScheduler):
            raise ValueError('scheduler must be dict or torch.optim.lr_scheduler._LRScheduler')

    def _set_device( self, blobs ):
        if type(blobs) == list:
            for i in range(len(blobs)):
                blobs[i] = self._set_device(blobs[i])
        elif type(blobs) == dict:
            for key, data in blobs.items():
                blobs[key] = self._set_device(data)
        elif torch.is_tensor(blobs):
            blobs = blobs.to(self._device, non_blocking=True)
        return blobs

    def is_main_process(self):
        if self.rank == 0 or self.rank is None:
            return True
        else:
            return False

    def validate(self):
        if isinstance(self._model, DDP):
            if not self.is_main_process():
                print('one process resturn')
                return
        self._validate()

    def _validate( self ):
        if isinstance(self._model, DDP):
            test_model = self._model.module
        else:
            test_model = self._model
        print('start to validate')
        self._model.eval()

        results = []
        with torch.no_grad() :
            for idx,(inputs, targets) in enumerate(tqdm.tqdm(self._testset, 'evaluating', dynamic_ncols=True)):
            #for idx,(inputs, targets) in enumerate(self._testset):
                inputs = self._set_device( inputs )
                output = test_model( inputs)
                batch_size = len(output['boxes'])
                #for i, im in enumerate(output):
                for i in range(batch_size):
                    if len(output['boxes'][i]) == 0:
                        continue
                    # convert to xywh
                    output['boxes'][i][:,2] -= output['boxes'][i][:,0]
                    output['boxes'][i][:,3] -= output['boxes'][i][:,1]
                    for j in range(len(output['boxes'][i])):
                        results.append({'image_id':int(targets[i]['image_id']), 
                                        'category_id':output['labels'][i][j].cpu().numpy().tolist(), 
                                        'bbox':output['boxes'][i][j].cpu().numpy().tolist(), 
                                        'score':output['scores'][i][j].cpu().numpy().tolist()})
                #if idx == 10:
                #    break
                #output = self._model['net']( inputs, just_embedding=True) # debug
                #bench.update( targets, output )
        result_path = '{}temp_result.json'.format(self._tag)
        with open(result_path,'w') as f:
            json.dump(results,f)
        #self.eval_result(dataset=self._dataset_name)
        self.evaluator.evaluate(result_path)

    def train( self ):
        raise NotImplementedError()

    def load_trained_model(self, model_path):
        state_dict = torch.load(model_path)['state_dict']
        self._model.load_state_dict(state_dict)

    def save_training(self, path, to_print=True):
        if isinstance(self._model, DDP):
            if self.is_main_process():
                state_dict = self._model.state_dict()
            else:
                return
        elif isinstance(self._model, torch.nn.DataParallel):
            state_dict = self._model.module.state_dict()
        else:
            state_dict =self._model.state_dict()

        torch.save({
            'epoch': self._epoch,
            'step': self._step,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler':self._scheduler.state_dict(),
        }, path)

        folder = os.path.dirname(path)
        if self._path_config is not None:
            last_path = self._path_config.checkpoint_path_tmp.format('last')
        else:
            last_path = os.path.join(folder,'{}_last.pth'.format(self._tag))
        torch.save({
            'epoch': self._epoch,
            'step': self._step,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler':self._scheduler.state_dict(),
        }, last_path)

        if to_print:
            print('The checkpoint has been saved to {}'.format(path))

    def resume_training(self, path=None, to_print=True):
        raise NotImplementedError
        if isinstance(self._model, DDP):
            dist.barrier()
        checkpoint = torch.load(path, map_location=device)
        #if self._epoch_based:
        self._epoch = checkpoint['epoch']
        self._step = checkpoint['step']
        #self.start_step = checkpoint['step']
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.move_optimizer_to_device(self._optimizer, device)
        if 'scheduler' in checkpoint:
            self._scheduler.load_state_dict(checkpoint['scheduler'])
        if to_print:
            print('Chekpoint has been loaded from {}'.format(path))
    
    def move_optimizer_to_device(self, optimizer, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def load_training(self, path, to_print=True):
        raise NotImplementedError()

    def reset_trainset(self):
        self._epoch += 1
        if self._trainset.sampler is not None:
            self._trainset.sampler.set_epoch(self._epoch)
        self.train_set_iter = iter(self._trainset)

    def init_logger(self):
        if self._log_with_tensorboard:
            self._writer = SummaryWriter(log_dir=self._path_config.log_dir, comment=self._tag)
        else:
            train_path = self._path_config.log_path
            console_formatter = '{} {{}}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            self._logger = Logger(level='info', file=train_path, console=False, console_formatter=console_formatter)
            print('Loss log path is: {}'.format(train_path))
    
    def print_log(self, average_losses):
        log_str = ''
        #average_losses = self.loss_logger.get_last_average()
        for loss in average_losses:
            loss_num = average_losses[loss]
            log_str += (' {} loss:{}, '.format(loss, loss_num))
        print(log_str[:-2])

    def save_log(self, average_losses):
        #average_losses = self.loss_logger.get_last_average()
        if not self._log_with_tensorboard:
            self._logger.info('{} '.format(self._step))
            for loss in average_losses:
                loss_num = average_losses[loss]
                self._logger.info('{} '.format(loss_num))
            self._logger.info('\n')
        else:
            self._writer.add_scalars('loss', average_losses, global_step=self._step)
            self._writer.add_scalar('lr', self._scheduler.get_last_lr()[0], global_step=self._step)
            self._writer.add_scalar('epoch', self._epoch, global_step=self._step)

        


    def load_checkpoint(self, path, to_print=True):
        state_dict_ = torch.load(path, map_location=self._device)['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        self._model.load_state_dict(state_dict, strict=True )
        #self._epoch = checkpoint['epoch']
        #self._model.load_state_dict(checkpoint['model_state_dict'])
        #self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if to_print:
            print('Chekpoint has been loaded from {}'.format(path))