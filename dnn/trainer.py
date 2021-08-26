import torch
import sys
import torch.optim as optim
from torch import nn
import numpy as np
import os
import progressbar
import tqdm
import json
import datetime
import math
import copy

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..tools.logger import Logger
from ..evaluation import COCOEvaluator


class BaseTrainer :
    def __init__( self, cfg, model, device, trainset, testset=None, optimizer=None, scheduler=None ):
        self._cfg = cfg
        self._device = device
        self._optimizer = optimizer
        self._model = model

        self._trainset = trainset
        self._testset = testset
        self._epoch = 0
        self._scheduler = scheduler
        self._niter = cfg.n_iter
        #self._trainset_feeder = data_feeder( trainset )

        #if testset is not None :
        #    self._testset_feeder = data_feeder( testset )

        if trainset is not None:
            self._set_optimizer()
        if self._scheduler is not None:
            self._set_scheduler()
        

    #def _set_optimizer( self ):
    #    params = self._cfg.dnn.OPTIMIZER
    #    self._optimizer = optim.SGD( self._model['net'].parameters(),
    #                                 lr=params['lr'],
    #                                 momentum=params.get('momentum',0.9),
    #                                 weight_decay=params.get('weight_decay',1e-4))

    #    self._scheduler = optim.lr_scheduler.StepLR( self._optimizer,
    #                                                step_size=params['decay_step'],
    #                                                gamma=params['decay_rate'] )

    #    self._niter = self._cfg.dnn.NITER

    def _set_optimizer( self ):
        if self._optimizer is None:
            params = self._cfg.optimizer
        if params['type'] == 'SGD':
            self._optimizer = optim.SGD( self._model.parameters(),
                                        lr=params['lr'],
                                        momentum=params.get('momentum',0.9),
                                        weight_decay=params.get('weight_decay',0))
        elif params['type'] == 'Adam':
            self._optimizer = optim.Adam(self._model.parameters(),
                                         lr = params['lr'],
                                         betas=params.get('betas',(0.9, 0.999)),
                                         eps = params.get('eps', 1e-8)
                                         )
        else:
            raise ValueError('Optimiser type wrong, {} is not a valid optimizer type!')

    def _set_scheduler(self):
        raise NotImplementedError
        #self._scheduler = optim.lr_scheduler.MultiStepLR( self._optimizer,
        #                                            milestones=params['decay_steps'],
        #                                            gamma=params['decay_rate'] )
        #self._niter = self._cfg.dnn.NITER

    def _set_device( self, blobs ):
        for n,d in blobs.items() :
            blobs[n] = d.to( self._device )
        return blobs

    def _train( self ):
        self._model.train()

        widgets = [ progressbar.Percentage(), ' ', progressbar.ETA(), ' ',
                    '(',progressbar.DynamicMessage('loss'),')' ]
        bar = progressbar.ProgressBar(widgets=widgets,max_value=len(self._trainset)).start()

        loss_values = []

        self._optimizer.zero_grad()
        for idx in range( len(self._trainset) ):
            inputs, targets = self._trainset.next()

            inputs = self._set_device( inputs )
            targets = self._set_device( targets )

            nets, outputs = self._model['net']( inputs, targets )
            loss = self._model['loss']( inputs, nets, outputs, targets )

            loss_values.append( loss.cpu().detach().numpy() )

            # Computing gradient and do SGD step
            #self._optimizer.zero_grad()
            loss.backward()
            #self._optimizer.step()
            if (idx+1)%self._accumulate_step == 0:
                # every 10 iterations of batches of size 10
                self._optimizer.step()
                self._optimizer.zero_grad()


            bar.update(idx+1,loss=loss.item())
        bar.finish()

        print('Average loss : ', np.mean(loss_values))

    def validate( self ):
        self._model['net'].eval()

        bench = self._testset.benchmark()

        with torch.no_grad() :
            for idx in range( len(self._testset) ):
                inputs, targets = self._testset.next()
                inputs = self._set_device( inputs )
                output = self._model['net']( inputs )
                bench.update( targets, output )

        bench.summary()

    def train( self ):
        #if self._testset is not None :
        #    self._validate()

        for i in range( self._epoch, self._niter ):
            print('epoch {} / {}'.format(i+1, self._niter))
            self._train()

            if self._testset is not None :
                self.validate()
            self._epoch = i
            if self._scheduler is not None:
                self._scheduler.step()

    def load_trained_model(self, model_path):
        state_dict = torch.load(model_path)['state_dict']
        self._model.load_state_dict(state_dict)

    def save_training(self, path, to_print=True,epoch_based=True):
        if isinstance(self._model, DDP):
            #if self.rank == 0:
            if self.is_main_process():
                state_dict = self._model.state_dict()
            else:
                return
        elif isinstance(self._model, torch.nn.DataParallel):
            state_dict = self._model.module.state_dict()
        else:
            state_dict =self._model.state_dict()
        if epoch_based:
            torch.save({
                'epoch': self._epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self._optimizer.state_dict(),
                'scheduler':self._scheduler.state_dict(),
                'niter':self._niter
            }, path)
            folder = os.path.dirname(path)
            torch.save({
                'epoch': self._epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self._optimizer.state_dict(),
                'scheduler':self._scheduler.state_dict(),
                'niter':self._niter
            }, os.path.join(folder, '{}_last.pth'.format(self._tag)))
        else: # step based
            torch.save({
                'step': self._step,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self._optimizer.state_dict(),
                'scheduler':self._scheduler.state_dict(),
                'nstep':self._step
            }, path)
            folder = os.path.dirname(path)
            torch.save({
                'step': self._step,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self._optimizer.state_dict(),
                'scheduler':self._scheduler.state_dict(),
                'nstep':self._step
            }, os.path.join(folder, '{}_last.pth'.format(self._tag)))

        if to_print:
            print('The checkpoint has been saved to {}'.format(path))

    def resume_training(self, path, device, to_print=True):
        if isinstance(self._model, DDP):
            dist.barrier()
        checkpoint = torch.load(path, map_location=device)
        if self._epoch_based:
            self._epoch = checkpoint['epoch']
        else:
            self.start_step = checkpoint['step']
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
        checkpoint = torch.load(path)
        self._epoch = checkpoint['epoch']
        self._model['net'] = checkpoint['model_state_dict']
        self._optimizer = checkpoint['optimizer_state_dict']
        self._scheduler = checkpoint['scheduler_state_dict']
        if to_print:
            print('Chekpoint has been loaded from {}'.format(path))

class DistributedTrainer(BaseTrainer):
    def __init__( self, cfg, model, device, trainset, tag='', testset=None, dataset_name=None, train_sampler=None, rank=None, benchmark=None, log_print_iter=1000, evaluator=None, clip_gradient=None, epoch_based=True, eval_step_interval=20000, save_step_interval=20000):
        self._cfg = cfg
        self._device = device
        self._optimizer = None
        self._model = model
        self._tag = tag
        self.log_print_iter = log_print_iter

        if isinstance(self._model, DDP):
            self.distributed = True
        else:
            self.distributed = False

        self.rank = rank
        self._train_sampler = train_sampler

        self._trainset = trainset
        self._testset = testset
        self._benchmark = benchmark
        self._dataset_name = dataset_name
        self._epoch_based = epoch_based
        self.eval_step_interval = eval_step_interval
        self.save_step_interval = save_step_interval
        self._epoch = 0
        self.loss_logger = LossLogger()
        if trainset is not None:
            self.train_set_iter = iter(self._trainset)
        if epoch_based:
            self._niter = cfg.optimizer.n_iter
        else: # step_based
            self.start_step = 1
            self.end_step = cfg.optimizer.total_step
        self._clip_gradient = clip_gradient
        if evaluator is None:
            self.evaluator = COCOEvaluator(dataset_name=dataset_name)
        else:
            self.evaluator = evaluator

        self._set_optimizer()
        self._set_scheduler()
        if cfg.resume:
            path = os.path.join(os.path.dirname(cfg.path.CHECKPOINT), '{}_last.pth'.format(self._tag)) 
            self.resume_training(path, device)

        if self.is_main_process():
            self.init_logger()

    def is_main_process(self):
        if self.rank == 0 or self.rank is None:
            return True
        else:
            return False

    def train(self):
        if self._epoch_based:
            self.train_epoch()
        else:       #step based training schedual
            self.train_step()

    def train_step( self ):

        self._optimizer.zero_grad()
        for step in tqdm.tqdm(range(self.start_step, self.end_step+1),desc='Training'):
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
                if step % self.save_step_interval == 0 or step==self.end_step:
                    self.save_training(self._cfg.path.CHECKPOINT.format(self._step), epoch_based=False)

            if hasattr(self, '_scheduler'):
                self._scheduler.step()

    def train_epoch( self ):
        #if self._testset is not None :
        #    self._validate()

        for i in range( self._epoch+1, self._niter+1 ):
            if self._train_sampler is not None:
                self._train_sampler.set_epoch(i)
            if self.is_main_process():
                print("Epoch %d/%d" % (i,self._niter))
                self._logger.info('epoch {}\n'.format(i))
            self._model.train()
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
                self.save_training(self._cfg.path.CHECKPOINT.format(self._epoch))
            if hasattr(self, '_scheduler'):
                self._scheduler.step()

    def validate( self ):
        if isinstance(self._model, DDP):
            if not self.is_main_process():
                return
        print('start to validate')
        self._model.eval()

        results = []
        with torch.no_grad() :
            for idx,(inputs, targets) in enumerate(tqdm.tqdm(self._testset, 'evaluating')):
            #for idx,(inputs, targets) in enumerate(self._testset):
                inputs = self._set_device( inputs )
                output = self._model.module( inputs)
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

    def eval_result(self, dataset='coco_person'):
        raise NotImplementedError

    def reset_trainset(self):
        self._epoch += 1
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self._epoch)
        self.train_set_iter = iter(self._trainset)

    def _train_step( self ):
        self._model.train()

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
            loss_sum = loss_sum + (loss_dict[single_loss]/self._cfg.accumulation_step)

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

        if self._step % self._cfg.accumulation_step == 0:
            self._optimizer.step()
            self._optimizer.zero_grad()

        if self.is_main_process():
            if self._step % self.log_print_iter == 1:
                self._logger.info('{} '.format(self._step+1))
                loss_str = ''
                average_losses = self.loss_logger.get_last_average()
                for loss in average_losses:
                    loss_num = average_losses[loss]
                    self._logger.info('{} '.format(loss_num))
                    loss_str += (' {} loss:{}, '.format(loss, loss_num))
                print(loss_str[:-2])
                average_losses = {}
                self._logger.info('\n')
            

    def _train_epoch( self ):
        self._model.train()

        loss_values = []
        average_losses = {}

        self._optimizer.zero_grad()
        for idx, (inputs, targets) in enumerate(tqdm.tqdm(self._trainset, desc='Training')):
            #print('inputs:', inputs)
            #print('targets:', targets)

            inputs = self._set_device( inputs )
            targets = self._set_device( targets )
            #print('input device:', inputs[0].device)

            loss_dict = self._model(inputs, targets)

            # add the losses for each part
            #loss_sum = sum(loss for loss in loss_dict.values())
            loss_sum=0
            for single_loss in loss_dict:
                loss_sum = loss_sum + (loss_dict[single_loss]/self._cfg.accumulation_step)
                #if self.rank==0 or not self.distributed:
                if self.is_main_process():
                    if single_loss in average_losses:
                        average_losses[single_loss]+= loss_dict[single_loss].mean()
                    else:
                        average_losses[single_loss] = loss_dict[single_loss].mean()

            loss_sum = loss_sum.mean()
            if not math.isfinite(loss_sum):
                #print("Loss is {}, stopping training".format(loss_sum))
                self._optimizer.zero_grad()
                print('wrong targets:',targets)
                print("Loss is {}, skip this batch".format(loss_sum))
                print(loss_dict)
                continue
                #sys.exit(1)

            # Computing gradient and do SGD step
            loss_sum.backward()

            if self._clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_gradient)

            if idx % self._cfg.accumulation_step == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()
            loss_values.append( loss_sum.cpu().detach().numpy() )

            if idx%self.log_print_iter == 0:
                #if self.rank == 0 or not self.distributed:
                if self.is_main_process():
                    self._logger.info('{} '.format(idx+1))
                    loss_str = ''
                    for loss in average_losses:
                        if idx==0:
                            loss_num = average_losses[loss]
                        else:
                            loss_num = average_losses[loss] / self.log_print_iter
                        self._logger.info('{} '.format(loss_num))
                        loss_str += (' {} loss:{}, '.format(loss, loss_num))
                    print(loss_str[:-2])
                    average_losses = {}
                    self._logger.info('\n')

            #if idx > 20:
            #    break
            
        print('Average loss : ', np.mean(loss_values))

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

    def _set_scheduler(self):
        scheduler = self._cfg.scheduler
        if scheduler.type == 'multi_step':
            self._scheduler = optim.lr_scheduler.MultiStepLR( self._optimizer,
                                                            milestones=scheduler['milestones'],
                                                            gamma=scheduler.get('gamma', 0.1))
        else:
            raise ValueError('unknow scheduler {}'.format(scheduler.type))

    def init_logger(self):
        train_path = self._cfg.path['LOG']

        console_formatter = '{} {{}}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self._logger = Logger(level='info', file=train_path, console=False, console_formatter=console_formatter)

        print('Loss log path is: {}'.format(train_path))


    #def resume_training(self, path, device, to_print=True):
    #    if isinstance(self._model, DDP):
    #        dist.barrier()
    #    checkpoint = torch.load(path, map_location=device)
    #    self._epoch = checkpoint['epoch']
    #    self._model.load_state_dict(checkpoint['model_state_dict'])
    #    self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    self._niter = self._niter
    #    if 'scheduler' in checkpoint:
    #        self._scheduler.load_state_dict(checkpoint['scheduler'])
    #    if to_print:
    #        print('Chekpoint has been loaded from {}'.format(path))

class LossLogger():
    def __init__(self) -> None:
        self.loss = None
        self.loss_average = None
        self.loss_count = 0
        self.loss_average_count = 0

    def get_last_average(self):
        average = {}
        for k,v in self.loss_average.items():
            average[k] = v / self.loss_average_count
        self.loss_average_count = 0
        self.loss_average = None
        return average

    def update(self, loss_dict):
        if self.loss is None:
            self.loss = {}
            for k in loss_dict.keys():
                self.loss[k] = 0

        self.loss_count += 1
        for k,v in loss_dict.items():
            self.loss[k] += v.item()

        if self.loss_average is None:
            self.loss_average = {}
            for k in loss_dict.keys():
                self.loss_average[k] = 0

        self.loss_average_count += 1
        for k,v in loss_dict.items():
            self.loss_average[k] += v.item()



# to make the older version compatible
trainer = BaseTrainer
trainer_dist = DistributedTrainer