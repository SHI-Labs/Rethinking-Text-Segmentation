import torch
import torchvision.models
import os.path as osp
from ..cfg_helper import cfg_unique_holder as cfguh
from ..log_service import print_log 
from .utils import get_total_param, get_total_param_sum, freeze

def load_state_dict(net, model_path):
    paras = torch.load(model_path, map_location=torch.device('cpu'))
    new_paras = net.state_dict()
    new_paras.update(paras)
    net.load_state_dict(new_paras)
    return

def save_state_dict(net, path):
    if isinstance(net, (torch.nn.DataParallel,
                        torch.nn.parallel.DistributedDataParallel)):
        torch.save(net.module.state_dict(), path)
    else:
        torch.save(net.state_dict(), path)

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_model(object):
    def __init__(self):
        self.model = {}

    def register(self, modelf, cfgname, kwmap, kwfix):
        self.model[modelf.__name__] = [modelf, cfgname, kwmap, kwfix]

    def __call__(self, name=None, cfgm=None):
        if cfgm is None:
            cfgm = cfguh().cfg.MODEL
        if name is None:
            name = cfgm.MODEL_NAME

        # the register is in each file
        if name == 'resnet':
            from . import resnet
        elif name == 'deeplab':
            from . import deeplab
        elif name == 'hrnet':
            from . import hrnet
        elif name == 'texrnet':
            from . import texrnet

        modelf, cfgname, kwmap, kwfix = self.model[name]
        cfgm = cfgm.__getitem__(cfgname)

        # MODEL_TAGS and PRETRAINED_PTH are two special args
        # FREEZE_BACKBONE_BN not frequently used.
        kw = {'tags' : cfgm.MODEL_TAGS}
        for k1, k2 in kwmap.items():
            if k2 in cfgm.keys():
                kw[k1] = cfgm[k2]
        kw.update(kwfix)
        net = modelf(**kw)

        # load init model
        if cfgm.PRETRAINED_PTH is not None:
            print_log('Load model from {0}.'.format(
                cfgm.PRETRAINED_PTH))
            load_state_dict(
                net, cfgm.PRETRAINED_PTH)

        # display param_num & param_sum
        print_log('Load {} with total {} parameters, {:3f} parameter sum.'.format(
            name, get_total_param(net), get_total_param_sum(net)))

        return net

def register(cfgname, kwmap={}, kwfix={}):
    def wrapper(class_):
        get_model().register(class_, cfgname, kwmap, kwfix)
        return class_
    return wrapper
