import torch
import torch.optim as optim
import numpy as np

from ..cfg_helper import cfg_unique_holder as cfguh

def get_optimizer(net, optimizer_name = None, opmgr = None):
    cfg = cfguh().cfg
    if optimizer_name is None:
        optimizer_name = cfg.TRAIN.OPTIMIZER
    
    # all lr are initialized as 0, 
    # because it will be set outside this function
    if opmgr is None:
        parameter_groups = net.parameters()
    else:
        para_num = len([i for i in net.parameters()])
        para_ckt = sum([
            len([i for i in pg['params']]) for pg in opmgr.get_pg(net)])
        if para_num != para_ckt:
            # this check whether the opmgr paragroup include all parameters.
            # TODO: may put a warning here
            raise ValueError
        parameter_groups = opmgr.get_pg(net)

    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            parameter_groups, 
            lr = 0, 
            momentum = cfg.TRAIN.SGD_MOMENTUM, 
            weight_decay = cfg.TRAIN.SGD_WEIGHT_DECAY)

    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            parameter_groups, 
            lr = 0, 
            betas = cfg.TRAIN.ADAM_BETAS,
            eps = cfg.TRAIN.ADAM_EPS,
            weight_decay = cfg.TRAIN.ADAM_WEIGHT_DECAY)
    
    else:
        raise ValueError

    return optimizer

def adjust_lr(op, new_lr, opmgr = None):
    for idx, param_group in enumerate(op.param_groups):
        if opmgr is None:
            param_group['lr'] = new_lr
        else:
            param_group['lr'] = opmgr.get_pglr(idx, new_lr)

class lr_scheduler(object):
    def __init__(self, 
                 types):
        self.lr = []
        for type in types:
            if type[0] == 'constant':
                _, v, n  = type
                lr = [v for i in range(n)]
            elif type[0] == 'ploy':
                _, va, vb, n, pw  = type
                lr = [ vb + (va-vb) * ((1-i/n)**pw) for i in range(n) ]
            elif type[0] == 'linear':
                _, va, vb, n  = type
                lr = [ vb + (va-vb) * (1-i/n) for i in range(n) ]
            else:
                raise ValueError
            self.lr += lr 
        self.lr = np.array(self.lr)

    def __call__(self, i):
        if i < len(self.lr):
            return self.lr[i]
        else:
            return self.lr[-1]





