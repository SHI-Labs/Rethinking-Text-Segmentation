import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import nputils
from .. import torchutils
from .. import loss 
from .get_model import get_model, register
from .optim_manager import optim_manager

from . import utils

version = 'v10'

class attention(object):
    def __init__(self, align_corners=True):
        self.ac = align_corners
        self.eps = 1e-12

    def __call__(self, x, att):
        bs, c, h, w = x.shape
        if att.shape[-2:] != x.shape[-2:]:
            att = torchutils.interpolate_2d(
                size=x.shape[-2:], mode='bilinear', 
                align_corners=self.ac)(att)

        xt = x.view(bs, c, -1).transpose(1, 2)
        att = att.view(bs, -1, h*w)

        key = torch.bmm(att, xt)
        key = key / (att.sum(dim=-1, keepdim=True)+self.eps)
        # key [bs x sem_n x c] now the normalized attention keys

        qk = torch.bmm(key, x.view(bs, c, -1))    
        qk = torch.softmax(qk, axis=1)
        qk = qk.view(bs, -1, h, w)
        return qk

class semantic_refinement_head(utils.semantic_head):
    def __init__(self, 
                 ic_n,
                 rfn_c_n,
                 sem_n,
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 align_corners=True,
                 ignore_label=None,
                 sem_losstype='ce',
                 rfn_losstype='ce',
                 bias_att_type='fixed',
                 intrain_getpred_from=None,
                 ineval_output_argmax=True,
                 **kwargs):
        super().__init__(
            ic_n = ic_n,
            sem_n = sem_n,
            align_corners = align_corners,
            ignore_label=ignore_label,
            loss_type=sem_losstype,
            ineval_output_argmax=ineval_output_argmax,
        )

        delattr(self, 'conv')
        delattr(self, 'lossf')

        self.ignore_label = ignore_label
        self.align_corners = align_corners
        self.bias_att_type = bias_att_type
        self.intrain_getpred_from = intrain_getpred_from
        self.ineval_output_argmax = ineval_output_argmax
        self.eps = 1e-12

        conv = utils.nn_component('conv')

        self.conv_sem = conv(ic_n, sem_n, 1, 1, 0, bias=False)
        self.bias_sem = torch.nn.Parameter(torch.zeros(sem_n))

        if bias_att_type == 'fixed':
            # each row of bias_att is the bias for each class
            self.bias_att = torch.nn.Parameter(torch.zeros([sem_n, sem_n]))
            self.bias_att.requires_grad = False
        elif bias_att_type == 'learnable':
            self.bias_att = torch.nn.Parameter(torch.zeros([sem_n, sem_n]))
        elif bias_att_type == 'cossim':
            self.bias_att = None
        elif bias_att_type == 'spatial':
            self.conv_batt = conv(ic_n, sem_n, 1, 1, 0, bias=True)
        else:
            raise ValueError
            
        self.conv_ref = utils.conv_block(
            rfn_c_n, [5, 1, 2], 
            conv_type, bn_type, relu_type, 
            conv_bias=False, last_conv_bias=True,
            end_with='relu')
        self.conv_ref_last = conv(
            rfn_c_n[-1], sem_n, 1, 1, 0, bias=True)

        self.lossf_sem = self.get_lossf(
            sem_losstype, 
            ignore_label=ignore_label,
            align_corners=align_corners)

        if isinstance(rfn_losstype, str):
            rfn_losstype = {'lossrfn' : rfn_losstype}
        self.lossf_rfn = self.get_lossf(
            rfn_losstype, 
            ignore_label=ignore_label,
            align_corners=align_corners)

        # disable the grad of bias_att conv2 bn_relu2 and conv3 
        # to keep the init (no weight decay)
        # initially not activated
        self.refinement_on = False
        self.activate_refinement(False)

    def forward(self, 
                x,
                xref,
                gtsem=None,):
        if x.shape[-2:] != xref.shape[-2:]:
            xref = torchutils.interpolate_2d(
                size=x.shape[-2:], mode='bilinear', 
                align_corners=self.align_corners)(xref)

        xcat = torch.cat([xref, x], axis=1)

        bs, c, h, w = x.shape
        x0 = self.conv_sem(x) 
        xsem = x0 + self.bias_sem.view(1, -1, 1, 1)
        
        if self.training and not self.refinement_on:
            losssem = {
                ni : lossfi(xsem, gtsem) \
                    for ni, lossfi in self.lossf_sem.items()}
            lossrfn = {
                ni : torch.zeros(()) \
                    for ni, _ in self.lossf_rfn.items()}
            losssem.update(lossrfn)

            if self.intrain_getpred_from == 'sem':
                xsem = torch.softmax(xsem, dim=1)
                losssem['pred'] = xsem
            elif self.intrain_getpred_from == 'rfn':
                # no rfn computed so use sem
                xsem = torch.softmax(xrfn, dim=1)
                losssem['pred'] = xsem
            elif self.intrain_getpred_from is None:
                pass
            else:
                raise ValueError
            return losssem

        if self.bias_att_type in ['fixed', 'learnable', 'cossim']:
            ma, _ = x0.max(dim=1, keepdim=True)
            ma = ma.detach()
            x1 = torch.exp(x0-ma).view(bs, -1, h*w)
            # this is a replicate of softmax, both channel multi exp(-ma) 
            # won't affect the softmax output but numerically stablize it. 

            if self.bias_att_type in ['fixed', 'learnable']:
                b = torch.exp(self.bias_att)
                b_diag = torch.diagonal(b)
            elif self.bias_att_type == 'cossim':
                sem_n = x0.shape[1]
                b = torch.softmax(xsem, dim=1).view(bs, sem_n, -1)
                bnorm = b.norm(p=2, dim=-1, keepdim=True)+self.eps
                b = b/bnorm
                b = torch.bmm(b, b.transpose(1, 2))
                # clear diagonal
                b = b*(1-torch.eye(sem_n, dtype=torch.float32, device=b.device))
                b = torch.exp(b)
                b_diag = torch.ones(
                    sem_n, dtype=torch.float32,device=b.device)

            # each row of bias_att is the bias for each class
            x1_sum = torch.matmul(b, x1)+self.eps
            x1 = x1 * b_diag.view(1, -1, 1)
            x1 = x1/x1_sum
            # x1 [bs x sem_n x hw] now holds the key attention on each class

        elif self.bias_att_type == 'spatial':
            xbias = self.conv_batt(x)
            x1 = torch.softmax(x0+xbias, dim=1)
            x1 = x1.view(bs, -1, h*w)
        
        x2 = x.view(bs, c, -1).transpose(1, 2)
        x2 = torch.bmm(x1, x2)
        x2 = x2 / (x1.sum(dim=-1, keepdim=True)+self.eps)
        # x2 [bs x sem_n x c] now the normalized attention keys

        x3 = torch.bmm(x2, x.view(bs, c, -1))
        x3 = torch.softmax(x3, axis=1)
        x3 = x3.view(bs, -1, h, w)
        # x3 [bs x sem_n x h x w] is the query-key attention.

        x4 = torch.cat([xcat, x3], dim=1)
        # x4 [bs x c+sem_n x 2D] attention fused with feature. 

        x4 = self.conv_ref(x4)
        xrfn = self.conv_ref_last(x4)

        if self.training:
            losssem = {
                ni : lossfi(xsem, gtsem) \
                    for ni, lossfi in self.lossf_sem.items()}
            lossrfn = {
                ni : lossfi(xrfn, gtsem) \
                    for ni, lossfi in self.lossf_rfn.items()}
            losssem.update(lossrfn)

            if self.intrain_getpred_from == 'sem':
                xsem = torch.softmax(xsem, dim=1)
                losssem['pred'] = xsem
            elif self.intrain_getpred_from == 'rfn':
                xrfn = torch.softmax(xrfn, dim=1)
                losssem['pred'] = xrfn
            elif self.intrain_getpred_from is None:
                pass
            else:
                raise ValueError
            return losssem

        xattk = x1.view(bs, -1, h, w)
        xatt  = x3

        if self.ineval_output_argmax:
            xsem = torch.argmax(xsem, dim=1)
            xrfn = torch.argmax(xrfn, dim=1)
        else:
            xsem = torch.softmax(xsem, dim=1)
            xrfn = torch.softmax(xrfn, dim=1)

        return {
            'att_key'  : xattk,
            'att'      : xatt,
            'predsem'  : xsem,
            'predrfn'  : xrfn,}

    def activate_refinement(self, onoff):
        self.refinement_on = onoff
        if self.bias_att_type == 'learnable':
            self.bias_att.requires_grad = onoff
        elif self.bias_att_type == 'spatial':
            self.conv_batt.weight.requires_grad = onoff
            self.conv_batt.bias.requires_grad = onoff
        for mi in [self.conv_ref, self.conv_ref_last]:
            for pi in mi.parameters():
                pi.requires_grad = onoff

class TexRNet(nn.Module):
    def __init__(self, 
                 bbn_name,
                 bbn, 
                 ic_n,
                 rfn_c_n,
                 sem_n, 
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 align_corners=False, 
                 ignore_label=None,
                 sem_losstype='ce',
                 rfn_losstype='ce',
                 bias_att_type='dynamic',
                 init_bias_att=None,
                 intrain_getpred_from=None,
                 ineval_output_argmax=True,
                 **kwargs):

        super().__init__()

        setattr(self, bbn_name, bbn)

        self.semrfnhead = semantic_refinement_head(
            ic_n, rfn_c_n, sem_n,
            conv_type, bn_type, relu_type,
            align_corners=align_corners,
            ignore_label=ignore_label,
            sem_losstype=sem_losstype,
            rfn_losstype=rfn_losstype,
            bias_att_type=bias_att_type,
            intrain_getpred_from=intrain_getpred_from,
            ineval_output_argmax=ineval_output_argmax,
        )

        self.bbn_name = bbn_name

        # initialize the weight 
        # (aspp and decoder initalized in base class)
        utils.init_module([self.semrfnhead])
        
        if bias_att_type is ['fixed', 'learnable']:
            self.semrfnhead.bias_att.data = torch.eye(sem_n)*init_bias_att

        self.opmgr = getattr(self, self.bbn_name).opmgr
        self.opmgr.inheritant(self.bbn_name)
        self.opmgr.pushback(
            'texrnet', ['semrfnhead']
        )

    def forward(self, 
                x, 
                gtsem=None, ):
        x0 = getattr(self, self.bbn_name)(x)
        o = self.semrfnhead(x0, x, gtsem)
        return o

    def activate_refinement(self):
        self.semrfnhead.activate_refinement(True)

@register(
    'TEXRNET', 
    {
        'ic_n'                  : 'INPUT_CHANNEL_NUM',
        'sem_n'                 : 'SEMANTIC_CLASS_NUM',
        'conv_type'             : 'CONV_TYPE',
        'bn_type'               : 'BN_TYPE',
        'relu_type'             : 'RELU_TYPE',
        'align_corners'         : 'ALIGN_CORNERS',
        'ignore_label'          : 'SEMANTIC_IGNORE_LABEL',
        'sem_losstype'          : 'SEMANTIC_LOSS_TYPE',
        'rfn_losstype'          : 'REFINEMENT_LOSS_TYPE',
        'bias_att_type'         : 'BIAS_ATTENTION_TYPE',
        'init_bias_att'         : 'INIT_BIAS_ATTENTION_WITH',
        'rfn_c_n'               : 'REFINEMENT_CHANNEL_NUM',
        'intrain_getpred_from'  : 'INTRAIN_GETPRED_FROM',
        'ineval_output_argmax'  : 'INEVAL_OUTPUT_ARGMAX', 
    })
def texrnet(tags, **para):
    if 'deeplab' in tags:
        bbn = get_model()('deeplab')
        para['bbn_name']  = 'deeplab'
        para['bbn']       = bbn
    elif 'resnet' in tags:
        bbn = get_model()('resnet')
        para['bbn_name']  = 'resnet'
        para['bbn']       = bbn
    elif 'hrnet' in tags:
        bbn = get_model()('hrnet')
        para['bbn_name']  = 'hrnet'
        para['bbn']       = bbn

    if 'simple' in tags:
        net = TexRNet_Simple(**para)
    else:
        net = TexRNet(**para)
    return net
