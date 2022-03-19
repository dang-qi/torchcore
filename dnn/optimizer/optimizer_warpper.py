import copy
from .build import OPTIMIZER_REG, build_with_config, OPTIMIZER_COSTRUCTOR_REG
from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

@OPTIMIZER_COSTRUCTOR_REG.register(force=True)
class OptimizerWarpper():
    ''' here we want to build optimizer with customized parameter, either by name or by norm 
    The valid param_cfg keys could be:
        norm_weight_decay: set up the weight decay on norm layers
    '''

    def __init__(self, optimizer_cfg, param_cfg=None) -> None:
        self.optimizer_cfg = copy.deepcopy(optimizer_cfg)
        self.param_cfg = param_cfg
        self.lr = optimizer_cfg.get('lr', None)
        self.weight_decay = optimizer_cfg.get('weight_decay', None)
        if param_cfg is None:
            param_cfg = {}
        self.norm_weight_decay = param_cfg.get('norm_weight_decay',None)
        #self.bias_weight_decay = param_cfg.pop('bias_weight_decay',None)

    def _check_param_cfg(self):
        '''We need to define the available param keys here'''
        valid_list = ['norm_weight_decay','bias_weight_decay']

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no param option is specified, just use the global setting
        if not self.param_cfg:
            optimizer_cfg['params'] = model.parameters()
            return build_with_config(optimizer_cfg, OPTIMIZER_REG)

        # set param-wise lr and weight decay recursively
        params = []
        self.add_params(model, params)
        optimizer_cfg['params'] = params

        return build_with_config(optimizer_cfg, OPTIMIZER_REG)

    def add_params(self,model, params):
        if self.norm_weight_decay is not None:
            norm_name_list = get_norm_param_name(model)
            norm_group = []
            other_group = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name in norm_name_list:
                    norm_group.append(param)
                else:
                    other_group.append(param)
            params.append({'params':norm_group, 'weight_decay':self.norm_weight_decay})

        self.add_default_params(model, params)

    def add_default_params(self, model, params):
        default_group = []
        exist_param = self.get_exist_param(params)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param not in exist_param:
                default_group.append(param)
        if default_group:
            params.append({'params':default_group})

    def get_exist_param(self, params):
        p_all = set()
        for g in params:
            p_all.update(g['params'])
        return p_all


    #def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    #    decay = []
    #    no_decay = []
    #    for name, param in model.named_parameters():
    #        if not param.requires_grad:
    #            continue
    #        if len(param.shape) == 1 or name in skip_list:
    #            no_decay.append(param)
    #        else:
    #            decay.append(param)
    #    return [
    #        {'params': no_decay, 'weight_decay': 0.},
    #        {'params': decay, 'weight_decay': weight_decay}]

def is_norm(module):  
    return isinstance(module,(_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))

def get_norm_param_name(module):
    name_list = []
    for n,c in module.named_modules():
        if is_norm(c):
            name_list.append(n+'.weight')
            name_list.append(n+'.bias')
    return name_list