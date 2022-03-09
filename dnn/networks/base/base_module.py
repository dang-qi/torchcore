import copy
from re import sub
from torch.nn import Module
from ....util.build import build_with_config
from .weight_init import initialize
import warnings
from collections import defaultdict

class BaseModule(Module):
    '''Merge weight init in the module'''
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = copy.deepcopy(init_cfg)
        self.has_initialized = False

    def init_weights(self):
        '''add init log here'''
        if not hasattr(self,'_init_log'):
            self._init_log = defaultdict(dict)

            for n, p in self.named_parameters():
                self._init_log[p]['mean'] = p.data.mean()
                self._init_log[p]['var'] = p.data.var()
                self._init_log[p]['info'] = 'It is not init by init_weight'

            # just want to make sure everything are syced when
            # initilization happens in sub modules
            for submodule in self.modules():
                submodule._init_log = self._init_log

        if not self.has_initialized:
            if self.init_cfg:
                initialize(self, self.init_cfg)
                self.has_initialized=True

            # We should also init the sub modules if needed
            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
        else:
            warnings.warn('Module {} has been initialized before.'.format(self.__class__.__name__))

    def print_info(self, print_statistics=False):
        if hasattr(self, '_init_log'):
            for n, p in self.named_parameters():
                log_string = '{} - {} \n {}'.format(n, p.shape, self._init_log[p]['info'])
                if print_statistics:
                    log_string+='\nmean: {}, var: {}'.format(self._init_log[p]['mean'], self._init_log[p]['var'])
                print(log_string)
