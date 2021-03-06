import torch
import numpy as np
import os
import copy
import json
import tqdm
from datetime import datetime
import git

from collections import OrderedDict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from ...mlblogapi.mlblogapi import MLBlogAPI
from ...util.logging import LossLogger, Logger

from ..optimizer.build import build_optimizer, build_lr_scheduler

from ...evaluation.build import build_evaluator

from ...data.datasets.build import build_dataloader

from ..utils.ema import ModelEMA

from torchcore.dnn.networks.tools.load_from_mmdetection import mm_result_to_my_result


class BaseTrainer :
    def __init__( self, model, trainset, tag='', rank=0, world_size=1, log_print_iter=1000, log_save_iter=50, testset=None, optimizer=None, scheduler=None, clip_gradient=None, evaluator=None, accumulation_step=1, path_config=None, log_with_tensorboard=False, log_api_token=None, empty_cache_iter=None, log_memory=False, use_amp=False, ema_cfg=None ):
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
        self._world_size = world_size
        self._path_config=path_config
        self._empty_cache_iter= empty_cache_iter
        self._log_memory=log_memory
        self.use_amp = use_amp
        self._scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.use_ema = ema_cfg is not None
        if self.use_ema:
            self.ema = ModelEMA(model, **ema_cfg)

        if isinstance(self._model, DDP):
            self.distributed = True
        else:
            self.distributed = False
        self._log_with_tensorboard=log_with_tensorboard
        self._log_api_token = log_api_token

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

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_train_epoch(self):
        pass

    def after_train_epoch(self):
        pass
        
    def before_train_iter(self):
        pass

    def after_train_iter(self):
        pass

    def before_eval(self):
        pass

    def after_eval(self):
        pass

    def before_eval_epoch(self):
        pass

    def after_eval_epoch(self):
        pass
        
    def before_eval_iter(self):
        pass

    def after_eval_iter(self):
        pass

    def init_log_api(self):
        self._ml_log = MLBlogAPI(self._log_api_token)
        self._ml_log.setup(nepoch=self._max_epoch,epoch_size=len(self._trainset))

    def _set_optimizer( self ):
        if isinstance(self._optimizer, dict):
            optimizer_cfg = self._optimizer.copy()
            self._optimizer = build_optimizer(self._model,optimizer_cfg,)
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

    def validate_mm(self, mm_input=False):
        self._model.eval()
        if isinstance(self._model, DDP):
            test_model = self._model.module
        else:
            test_model = self._model
        print('start to validate with mm model')

        results = []
        with torch.no_grad() :
            for idx,(data) in enumerate(tqdm.tqdm(self._testset, 'evaluating', dynamic_ncols=True)):
            #for idx,(inputs, targets) in enumerate(self._testset):
                if mm_input:
                    mm_result = test_model(return_loss=False, **data )
                    targets = [{'image_id':int(im_meta.data[0][0]['ori_filename'].split('.')[0]) for im_meta in data['img_metas']}]
                else:
                    inputs,targets = data
                    inputs = self._set_device( inputs )
                    img_meta = [{'img_shape':imsize,'scale_factor':imscale,'flip':False} for imsize,imscale in zip(inputs['image_sizes'],inputs['image_sizes'])]
                    for meta in img_meta:
                        meta['pad_shape'] = (inputs['data'].shape[2],inputs['data'].shape[3],inputs['data'].shape[1])
                    #gt_boxes = [t['boxes'] for t in targets]
                    #gt_labels = [t['labels']-1 for t in targets]
                    #the losses of five layers
                    mm_result=test_model([inputs['data']],[img_meta],return_loss=False)
                output = mm_result_to_my_result(mm_result)
                #output = test_model( inputs)
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
                                        'category_id':output['labels'][i][j].tolist(), 
                                        'bbox':output['boxes'][i][j].tolist(), 
                                        'score':output['scores'][i][j].tolist()})
                #if idx == 10:
                #    break
                #output = self._model['net']( inputs, just_embedding=True) # debug
                #bench.update( targets, output )
        if hasattr(self._testset.dataset, 'cast_result_id'):
            self._testset.dataset.cast_result_id(results)
        result_path = '{}temp_result.json'.format(self._tag)
        with open(result_path,'w') as f:
            json.dump(results,f)
        #self.eval_result(dataset=self._dataset_name)
        self.evaluator.evaluate(result_path)

    def _validate( self ):
        if self.use_ema:
            print('Use ema model to test!')
            test_model = self.ema.ema
        else:
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
        if hasattr(self._testset.dataset, 'cast_result_id'):
            self._testset.dataset.cast_result_id(results)
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

    def state_dict_for_save(self):
        if self.use_ema:
            return self.ema.ema.state_dict()
        if isinstance(self._model, (DDP, torch.nn.DataParallel)):
            return self._model.module.state_dict()
        else:
            return self._model.state_dict()

    def save_last_training(self, path):
        if not self.is_main_process():
            return
        state_dict = self.state_dict_for_save()
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
            'scaler':self._scaler.state_dict(),
        }, last_path)

    def save_training(self, path, to_print=True):
        if not self.is_main_process():
            return
        state_dict = self.state_dict_for_save()

        torch.save({
            'epoch': self._epoch,
            'step': self._step,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler':self._scheduler.state_dict(),
            'scaler':self._scaler.state_dict(),
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
            'scaler':self._scaler.state_dict(),
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
            if self.distributed:
                self._trainset.sampler.set_epoch(self._epoch)
        self.train_set_iter = iter(self._trainset)

    def clip_gradient(self):
        # Unscales the gradients of optimizer's assigned params in-place
        self._scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_gradient)

    def init_logger(self):
        if self._log_with_tensorboard:
            self._writer = SummaryWriter(log_dir=self._path_config.log_dir, comment=self._tag)
        else:
            train_path = self._path_config.log_path
            console_formatter = '{} {{}}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            self._logger = Logger(level='info', file=train_path, console=False, console_formatter=console_formatter)
            print('Loss log path is: {}'.format(train_path))
        
    def print_log(self, log_dict):
            log_str = []
            log_str.append('iter: {}'.format(self._step))
            log_str.append('epoch: {}'.format(self._epoch))
            log_str.append('lr: {:.4e}'.format(self._scheduler.get_last_lr()[0]))
            if hasattr(self, 'data_time'):
                log_str.append('data_time: {:.4f}'.format(self.data_time))
            if hasattr(self, 'iter_time'):
                log_str.append('iter_time: {:.4f}'.format(self.iter_time))
            for k,v in log_dict.items():
                log_str.append('{}: {}'.format(k, v))
            log_str = self.log_memory(log_str)
            if self.is_main_process():
                print(datetime.now(),'   ',', '.join(log_str))
                #print(torch.cuda.memory_stats(self._device))
                #print(torch.cuda.memory_summary(self._device))
                #print('max:',torch.cuda.max_memory_allocated(self._device))
                #print('current:',torch.cuda.memory_allocated(self._device))

    def log_memory(self, log_str ):
        if self._log_memory:
            mem_mb=self.get_max_mem()
            log_str.append('memory:{}MB'.format(mem_mb))
        return log_str

    def get_max_mem(self):
        mem = torch.cuda.max_memory_allocated(device=self._device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                            dtype=torch.int,
                            device=self._device)
        #if self._world_size > 1:
        #    dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def update_log(self, scalar_name, scalar):
        if self.is_main_process():
            if not self._log_with_tensorboard:
                self._logger.info('step:{}, {}: {}'.format(self._step, scalar_name,scalar))
                self._logger.info('\n')
            else:
                self._writer.add_scalar(scalar_name, scalar, global_step=self._step)

    def save_log(self, average_losses):
        #average_losses = self.loss_logger.get_last_average()
        if self.is_main_process():
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
                if self._log_memory:
                    self._writer.add_scalar('mem', self.get_max_mem(), global_step=self._step)

            if self._log_api_token is not None:
                loss_sum = sum(average_losses.values())
                self._ml_log.update(self._epoch, self._step, loss_sum)

        
    # from mmdet detector/base.py
    def _parse_loss_dict(self,losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                    if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                        f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                        ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


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
    
    def get_hash(self, short=True):
        sha_all = []
        short_sha_all = []
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        short_sha = repo.git.rev_parse(sha, short=4)
        sha_all.append(sha)
        short_sha_all.append(short_sha)
        for m in repo.submodules:
            sub_sha = repo.submodule(m.name).module().head.object.hexsha
            short_sub_sha = repo.git.rev_parse(sub_sha, short=4)
            sha_all.append(sub_sha)
            short_sha_all.append(short_sub_sha)
        self.short_git_hash = '_'.join(short_sha_all)
        self.git_hash =  '_'.join(sha_all)

        
