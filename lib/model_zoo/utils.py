import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

import matplotlib.pyplot as plt

from .. import nputils
from .. import torchutils
from .. import loss

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_component(object):
    """
    The singleton class that can 
        register a compnent and 
        get small components
    """
    def __init__(self):
        self.component = {}
        self.register('none', None)

        # general convolution
        self.register(
            'conv', nn.Conv2d,
            kwinit={
                'bias'        : False}, )
        self.register(
            'conv1x1', nn.Conv2d, 
            kwmap={
                'kernel_size' : lambda x:1, 
                'padding'     : lambda x:0,},
            kwinit={
                'bias'        : False}, )
        self.register(
            'conv3x3', nn.Conv2d, 
            kwmap={
                'kernel_size' : lambda x:3, 
                'padding'     : lambda x:1,},
            kwinit={
                'bias'        : False}, )
        self.register(
            'conv5x5', nn.Conv2d, 
            kwmap={
                'kernel_size' : lambda x:5, 
                'padding'     : lambda x:3,},
            kwinit={
                'bias'        : False}, )
        self.register(
            'conv3x3d', nn.Conv2d, 
            kwmap={
                'kernel_size' : lambda x:3, 
                'padding'     : lambda x:x['dilation'],},
            kwinit={
                'dilation'    : 1,
                'bias'        : False}, )

        # general bn
        self.register('bn'    , nn.BatchNorm2d)
        self.register('syncbn', nn.SyncBatchNorm)

        # general relu
        self.register('relu'  , nn.ReLU)
        self.register('relu6' , nn.ReLU6)
        self.register(
            'lrelu', nn.LeakyReLU, 
            kwargparse={
                0 : ['negative_slope', float]
            }, )

        # general dropout
        self.register(
            'dropout', nn.Dropout, 
            kwargparse={
                0 : ['p', float]
            }, )
        self.register(
            'dropout2d', nn.Dropout2d, 
            kwargparse={
                0 : ['p', float]
            }, )

    def register(self, 
                 cname, 
                 objf, 
                 kwargparse={}, 
                 kwmap={},
                 kwinit={},):
        self.component[cname] = [objf, kwargparse, kwmap, kwinit]

    def __call__(self, cname):
        return copy.deepcopy(self.component[cname])

def register(cname, 
             kwargparse={}, 
             kwinit={},
             kwmap={}):
    def wrapper(class_):
        get_component().register(
            cname, class_, kwargparse, kwmap, kwinit)
        return class_
    return wrapper

class nn_component(object):
    def __init__(self, 
                 type=None):
        if type is None:
            self.f = None
            return

        type = type.split('|')
        type, para = type[0], type[1:]

        self.f, kwargparse, self.kwmap, self.kwpre = get_component()(type)

        self.kwpost = {}
        for i, parai in enumerate(para):
            fieldname, fieldtype = kwargparse[i]
            self.kwpost[fieldname] = fieldtype(parai)

    def __call__(self, *args, **kwargs):
        """
        The order or priority goes with the following order:
            kwpre -> kwargs(input) -> kwmap -> kwpost
        """
        if self.f is None:
            return identity()
        kw = copy.deepcopy(self.kwpre)
        kw.update(kwargs)
        kwnew = {fn:ff(kw) for fn, ff in self.kwmap.items()}
        kw.update(kwnew)
        kw.update(self.kwpost)
        return self.f(*args, **kw)

def conv_bn_relu(conv_type, bn_type, relu_type):
    return (
        nn_component(conv_type),
        nn_component(bn_type),
        nn_component(relu_type), )

class conv_block(nn.Module):
    """
    Common layers follows the template: 
        [conv+bn+relu] x n + conv<+bn><+relu>
    """
    def __init__(self, 
                 c_n, 
                 para = [3, 1, 1],
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 conv_bias = False,
                 last_conv_bias = True,
                 end_with = 'conv',
                 **kwargs):
        super().__init__()
        self.layer_n = len(c_n)-1
        self.end_with = end_with
        if self.layer_n < 1:
            # need at least the input and output channel number
            raise ValueError
        conv, bn, relu = conv_bn_relu(conv_type, bn_type, relu_type)

        ks, st, pd = para
        # layers except the last
        for i in range(self.layer_n-1):
            ca_n, cb_n = c_n[i], c_n[i+1]
            setattr(
                self, 'conv{}'.format(i),
                conv(ca_n, cb_n, ks, st, pd, bias=conv_bias))
            setattr(
                self, 'bn{}'.format(i),
                bn(cb_n))

        self.relu = relu(inplace=True)

        # last layer
        i = self.layer_n-1
        setattr(
            self, 'conv{}'.format(i),
            conv(c_n[i], c_n[i+1], ks, st, pd, bias=last_conv_bias))
        if end_with == 'conv':
            pass
        elif end_with in ['bn', 'relu']:
            setattr(
                self, 'bn{}'.format(i),
                bn(c_n[i+1])) 

    def forward(self, x, debug=False):
        for i in range(self.layer_n-1):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = getattr(self, 'bn{}'.format(i))(x)
            x = self.relu(x)

        i = self.layer_n-1
        x = getattr(self, 'conv{}'.format(i))(x)

        if self.end_with == 'bn':
            x = getattr(self, 'bn{}'.format(i))(x)
        elif self.end_with == 'relu':
            x = getattr(self, 'bn{}'.format(i))(x)
            x = self.relu(x)
        return x

class semantic_head(nn.Module):
    def __init__(self, 
                 ic_n,
                 sem_n,
                 align_corners=True,
                 ignore_label=None,
                 loss_type='ce',
                 intrain_getpred=False,
                 ineval_output_argmax=True,
                 **kwargs):
        super().__init__()
        self.intrain_getpred = intrain_getpred
        self.ineval_output_argmax = ineval_output_argmax

        self.conv = nn.Conv2d(ic_n, sem_n, 1, 1, 0, bias=True)
        self.lossf = self.get_lossf(
            loss_type,
            ignore_label=ignore_label,
            align_corners=align_corners)

    def forward(self, 
                x,
                gtsem=None,
            ):
        x = self.conv(x)
        if not self.training:
            if self.ineval_output_argmax:
                out = torch.argmax(x, dim=1)
            else:
                out = torch.softmax(x, dim=1)
            return {'predsem':out}
        loss = {
            ni : lossfi(x, gtsem) \
                for ni, lossfi in self.lossf.items()}

        if self.intrain_getpred:
            x = torch.softmax(x, dim=1)
            loss['pred'] = x
        return loss

    def get_lossf(self, 
                  loss_type,
                  **para):
        if isinstance(loss_type, dict):
            return {
                ni : self.get_lossf(parai, **para)['losssem'] \
                    for ni, parai in loss_type.items()}

        loss_para = loss_type.split('|')

        if loss_para[0] == 'ce':
            ig = para['ignore_label']
            ac = para['align_corners']
            lossf = torchutils.interpolate_2d_lossf(
                lossf = nn.CrossEntropyLoss(ignore_index=ig),
                resize_x = False,
                align_corners=ac, )
        elif loss_para[0] == 'ohemce':
            ig = para['ignore_label']
            ac = para['align_corners']
            lossf = torchutils.ohemce_lossf(
                thres = float(loss_para[1]),
                min_kept = int(loss_para[2]),
                ignore_label=ig,
                align_corners=ac, )
        elif loss_para[0] == 'dice':
            ig = para['ignore_label']
            ac = para['align_corners']
            try:
                atc = int(loss_para[1])
            except:
                atc = None
            lossf = torchutils.dice_lossf(
                with_softmax=True,
                at_channel=atc,
                balance_batch=False,
                balance_class=True,
                ignore_label=ig,
                align_corners=ac, )
        elif loss_para[0] == 'trimapce':
            ig = para['ignore_label']
            ac = para['align_corners']
            lossf = torchutils.boundaryce_lossf(
                resize_x = False,
                ignore_label=ig,
                align_corners=ac, )
        else:
            raise ValueError
        return {'losssem' : lossf}

class classification_head(nn.Module):
    """
    A common head that compute ce loss for classification
        (no use loss.loss)
        contains a nn.Linear layer.
    """
    def __init__(self, 
                 ic_n,
                 cls_n,
                 ignore_label=None,
                 **kwargs):
        super().__init__()
        self.fc = nn.Linear(ic_n, cls_n, bias=True)

        if ignore_label is None:
            ignore_label = -100

        self.lossf = nn.CrossEntropyLoss(
            ignore_index=ignore_label)

    def forward(self, 
                x,
                gtcls=None):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if not self.training:
            _, out = torch.topk(x, 5, dim=1, largest=True, sorted=True)
            return {'predt1':out[:, 0], 'predt5':out, 'pred':x}
        else:
            _, out = torch.topk(x, 1, dim=1, largest=True, sorted=True)
            accn = (out.view(-1)==gtcls)
            loss = self.lossf(x, gtcls)
            return {'losscls':loss, 'accnum':accn}

###############
# some helper #
###############

def freeze(net):
    for m in net.modules():
        if isinstance(m, (
                nn.BatchNorm2d, 
                nn.SyncBatchNorm,)):
            # inplace_abn not supported
            m.eval()
    for pi in net.parameters():
        pi.requires_grad = False
    return net

def common_init(m):
    if isinstance(m, (
            nn.Conv2d, 
            nn.ConvTranspose2d,)):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (
            nn.BatchNorm2d, 
            nn.SyncBatchNorm,)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        try:
            import inplace_abn
            if isinstance(m, (
                    inplace_abn.InPlaceABN,)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        except:
            pass

def init_module(module):
    """
    Args:
        module: [nn.module] list or nn.module
            a list of module to be initialized.
    """
    if isinstance(module, (list, tuple)):
        module = list(module)
    else:
        module = [module]

    for mi in module:
        for mii in mi.modules():
            common_init(mii)

def get_total_param(net):
    return sum(p.numel() for p in net.parameters())

def get_total_param_sum(net):
    with torch.no_grad():
        s = sum(p.cpu().detach().numpy().sum().item() for p in net.parameters())
    return s 

def eval_bn(net):
    try:
        netref = net.module()
    except:
        netref = net
    for m in netref.modules():
        if isinstance(m, (
                nn.BatchNorm2d, 
                nn.SyncBatchNorm,)):
            # inplace_abn not supported
            m.eval()
    return net

