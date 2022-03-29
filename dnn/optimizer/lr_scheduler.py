from bisect import bisect_right
import warnings
import torch
from .build import SCHEDULER_REG
from math import cos, pi

@SCHEDULER_REG.register(force=True)
class WarmupLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    '''A lr scheduler with warmup,
       To make it as a base class, we need to rewrite the
       get_regular_lr function. 
       The iter and epoch index all start from 0
    '''
    def __init__(
            self,
            optimizer,
            update_method='epoch',
            warmup=False,
            warmup_iter=0,
            warmup_factor=1.0/3,
            warmup_method='linear',
            warmup_by_epoch=False,
            last_epoch=-1):
        '''update_method: can be epoch, iter, epoch_iter
           warmup_factor: the warmup scale will multiplex
                warmup_factor before multiplex regular lr
           warmup_by_epoch: means the warmup lr updated on epoch based,
                it is also ok if warmup_by_epoch is False while the warmup lasts for severl epochs
        '''
        assert update_method in ['epoch', 'iter', 'epoch_iter']
        assert warmup_method in ['linear', 'constant', 'exp']
        # self.base_lrs has been get from the init
        self.warmup = warmup
        self.warmup_method = warmup_method
        self.warmup_factor = warmup_factor
        self.warmup_by_epoch = warmup_by_epoch
        self.update_method = update_method
        self.warmup_iter = warmup_iter
        self.cur_epoch = 0
        self.cur_iter = 0
        if update_method == 'epoch':
            self.by_epoch = True
            self.by_iter = False
        elif update_method == 'iter':
            self.by_epoch=False
            self.by_iter = True
        #elif update_method == 'epoch_iter':
        #    self.by_epoch=True
        #    self.by_iter = True
        super().__init__(optimizer, last_epoch)
    
    def _calculate_warmup_scale(self, cur_iter):
        # here cur_iter should start from 0
        assert cur_iter>=0
        cur_iter += 1
        if self.warmup_method == 'linear':
            scale = cur_iter/self.warmup_iter
        elif self.warmup_method == 'constant':
            scale = 1.0
        elif self.warmup_method == 'exp':
            scale = pow(cur_iter/self.warmup_iter, 2)
        scale *= self.warmup_factor
        return scale

    def get_warmup_lr(self, regular_lrs):
        if self.warmup_by_epoch:
            scale = self._calculate_warmup_scale(self.cur_epoch)
        else:
            scale = self._calculate_warmup_scale(self.cur_iter)

        warm_lr = [scale*lr for lr in regular_lrs]
        return warm_lr

    def get_lr(self,):
        lr = self.get_regular_lr()
        if self.warmup:
            cur_iter = self.cur_epoch if self.warmup_by_epoch else self.cur_iter
            if cur_iter < self.warmup_iter:
                lr = self.get_warmup_lr(lr, )
        return lr

    def get_regular_lr(self,):
        return self.base_lrs

    def step(self, cur_iter=None, cur_epoch=None):
        if cur_iter is not None:
            self.cur_iter = cur_iter
        if cur_epoch is not None:
            self.cur_epoch = cur_epoch
        super().step()

    def step_iter(self, cur_iter=None, cur_epoch=None):
        if self.by_iter:
            self.step(cur_iter=cur_iter, cur_epoch=cur_epoch)

    def step_epoch(self, cur_iter=None, cur_epoch=None):
        if self.by_epoch:
            self.step(cur_iter=cur_iter, cur_epoch=cur_epoch)

@SCHEDULER_REG.register()
class MultiStepScheduler(WarmupLRScheduler):
    def __init__(self, 
                 optimizer,
                 milestones, 
                 gamma=0.1,
                 update_method='epoch', 
                 warmup=False, 
                 warmup_iter=0, 
                 warmup_factor=1 / 3, 
                 warmup_method='linear', 
                 warmup_by_epoch=False, 
                 last_epoch=-1):
        self.milestones=milestones
        self.gamma = gamma
        super().__init__(optimizer, update_method, warmup, warmup_iter, warmup_factor, warmup_method, warmup_by_epoch, last_epoch)

    def get_regular_lr(self,):
        if self.by_epoch:
            i = self.cur_epoch
        else:
            i = self.by_iter

        return [lr * self.gamma**bisect_right(self.milestones,i)
                for lr in self.base_lrs]

@SCHEDULER_REG.register(force=True)
class CosineAnnealingScheduler(WarmupLRScheduler):
    def __init__(self, 
                optimizer, 
                min_lr=None,
                min_lr_ratio=None,
                update_method='epoch', 
                warmup=False, 
                warmup_iter=0, 
                warmup_factor=1 / 3, 
                warmup_method='exp', 
                warmup_by_epoch=False,
                max_iter=None,
                max_epoch=None,
                last_epoch=-1):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        super().__init__(optimizer, update_method, warmup, warmup_iter, warmup_factor, warmup_method, warmup_by_epoch, last_epoch)

    def get_regular_lr(self):
        self.cur_epoch
        self.cur_iter
        if self.by_epoch:
            factor = self.cur_epoch / self.max_epoch
        else:
            factor = self.cur_iter / self.max_iter
        if self.min_lr is not None:
            min_lr = self.min_lr
            regular_lr = [cos_annealing(min_lr, lr, factor)for lr in self.base_lrs]
        else:
            regular_lr = [cos_annealing(lr*self.min_lr_ratio, lr, factor)for lr in self.base_lrs]
        
        return regular_lr

def cos_annealing(min_lr, max_lr, factor, weight=1):
    '''factor = t_cur/t_i, t_i'''
    return min_lr + 0.5 * (max_lr-min_lr) * (cos(pi*factor)+1)

@SCHEDULER_REG.register(force=True)
class YOLOXScheduler(WarmupLRScheduler):
    '''
        epoch 1-5: just warmup by iter with exp mode, no cosine
        epoch 6-285: cosine iter_based, stoped at min_lr
        epoch 286-300: same min_lr
    '''
    def __init__(
            self, 
            optimizer, 
            iter_per_epoch,
            num_last_epochs=15,
            max_epoch=300,
            min_lr_ratio=0.05,
            warmup=True,
            warmup_iter=0,
            warmup_factor=1,
            warmup_method='linear',
            warmup_by_epoch=False,
            update_method='iter', 
            warmup_epoch=5,
            last_epoch=-1):
        '''set up iter_per_epoch to None and update in trainer'''
        if iter_per_epoch is None:
            # we need to update the iter per epoch in trainer
            self.saved_vars = locals()
            self.saved_vars.pop('self')
            self.has_update_iter_per_epoch = False
            return
        self.num_last_epochs = num_last_epochs
        self.iter_per_epoch = iter_per_epoch
        # calculate the warmup_iter here
        warmup_iter = warmup_epoch*iter_per_epoch
        self.max_epoch = max_epoch
        self.no_change_epoch = max_epoch-num_last_epochs
        self.min_lr_ratio = min_lr_ratio
        self.cos_total_iter = (max_epoch-num_last_epochs-warmup_epoch)*iter_per_epoch

        self.has_update_iter_per_epoch = True
        
        super().__init__(optimizer, update_method, warmup, warmup_iter, warmup_factor, warmup_method, warmup_by_epoch, last_epoch)


    def get_regular_lr(self):
        if self.cur_epoch <=5:
            return self.base_lrs
        elif self.cur_epoch > self.no_change_epoch:
            return [lr*self.min_lr_ratio for lr in self.base_lrs]
        else:
            factor = (self.cur_iter-self.warmup_iter)/self.cos_total_iter
            regular_lr = [cos_annealing(lr*self.min_lr_ratio, lr, factor)for lr in self.base_lrs]
            return regular_lr

    def update_iter_per_epoch(self, iter_per_epoch):
        if self.has_update_iter_per_epoch:
            warnings.warn('the iter_per_epoch has been updated, please check if this is your intended action')
        
        self.num_last_epochs = self.saved_vars.get('num_last_epochs')
        self.iter_per_epoch = iter_per_epoch
        self.warmup_epoch = self.saved_vars.get('warmup_epoch')
        # calculate the warmup_iter here
        warmup_iter = self.warmup_epoch*self.iter_per_epoch
        self.max_epoch = self.saved_vars.get('max_epoch')
        self.no_change_epoch = self.max_epoch-self.num_last_epochs
        self.min_lr_ratio = self.saved_vars.get('min_lr_ratio')
        self.cos_total_iter = (self.max_epoch-self.num_last_epochs-self.warmup_epoch)*iter_per_epoch
        
        super().__init__(self.saved_vars.get('optimizer'), self.saved_vars.get('update_method'), self.saved_vars.get('warmup'), warmup_iter, self.saved_vars.get('warmup_factor'), self.saved_vars.get('warmup_method'), self.saved_vars.get('warmup_by_epoch'), self.saved_vars.get('last_epoch'))
        self.has_update_iter_per_epoch = True
        




### copy from https://github.com/tianzhi0549/FCOS/blob/0eb8ee0b7114a3ca42ad96cd89e0ac63a205461e/fcos_core/solver/lr_scheduler.py 
@SCHEDULER_REG.register(force=True)
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def update_milestone_from_epoch_to_iter(self, dataset_len):
        self.milestones = [m*dataset_len for m in self.milestones]