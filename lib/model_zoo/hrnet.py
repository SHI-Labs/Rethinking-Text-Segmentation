import sys
# import os
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), '..', '..', 'hrnet_code', 'lib'))
# sys.path.append('/home/james/Spy/hrnet/HRNet-Semantic-Segmentation-HRNet-OCR/lib')
from models import seg_hrnet
from models import seg_hrnet_ocr

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import nputils
from .. import torchutils
from .. import loss 
from .get_model import get_model, register
from .optim_manager import optim_manager

from . import utils

version = 'v0'
"""
v0: the original code from github made by paper author
"""

class HRNet_Base(seg_hrnet.HighResolutionNet):
    def __init__(self, 
                 oc_n,
                 align_corners,
                 ignore_label,
                 stage1_para,
                 stage2_para,
                 stage3_para,
                 stage4_para,
                 final_conv_kernel, 
                 **kwargs):
        from easydict import EasyDict as edict
        config = edict()
        config.MODEL = edict()
        config.MODEL.ALIGN_CORNERS = align_corners
        config.MODEL.EXTRA = {}
        config.MODEL.EXTRA['STAGE1'] = stage1_para
        config.MODEL.EXTRA['STAGE2'] = stage2_para
        config.MODEL.EXTRA['STAGE3'] = stage3_para
        config.MODEL.EXTRA['STAGE4'] = stage4_para
        config.MODEL.EXTRA['FINAL_CONV_KERNEL'] = final_conv_kernel
        config.DATASET = edict()
        config.DATASET.NUM_CLASSES = 1 # dummy
        super().__init__(config)

        self.opmgr = optim_manager(
            group = {'hrnet': 'self'},
            order = ['hrnet'],
        )

        last_inp_channels = self.last_layer[0].in_channels
        # BatchNorm2d = nn.SyncBatchNorm
        BatchNorm2d = nn.BatchNorm2d
        relu_inplace = True
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=oc_n,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(oc_n, momentum=0.1),
            nn.ReLU(inplace=relu_inplace),
        )

    def forward(self, x):
        x = super().forward(x)
        return x

class HRNet(seg_hrnet.HighResolutionNet):
    def __init__(self, 
                 cls_n,
                 align_corners,
                 ignore_label,
                 stage1_para,
                 stage2_para,
                 stage3_para,
                 stage4_para,
                 final_conv_kernel, 
                 loss_type='ce',
                 intrain_getpred=False,
                 ineval_output_argmax=False,
                 **kwargs):
        from easydict import EasyDict as edict
        config = edict()
        config.MODEL = edict()
        config.MODEL.ALIGN_CORNERS = align_corners
        config.MODEL.EXTRA = {}
        config.MODEL.EXTRA['STAGE1'] = stage1_para
        config.MODEL.EXTRA['STAGE2'] = stage2_para
        config.MODEL.EXTRA['STAGE3'] = stage3_para
        config.MODEL.EXTRA['STAGE4'] = stage4_para
        config.MODEL.EXTRA['FINAL_CONV_KERNEL'] = final_conv_kernel
        config.DATASET = edict()
        config.DATASET.NUM_CLASSES = cls_n
        super().__init__(config)

        self.semhead = utils.semantic_head_noconv(
            align_corners = align_corners,
            ignore_label = ignore_label,
            loss_type = loss_type,
            intrain_getpred = intrain_getpred,
            ineval_output_argmax = ineval_output_argmax,
        )

        self.opmgr = optim_manager(
            group = {'hrnet': 'self'},
            order = ['hrnet'],
        )

    def forward(self, x, gtsem=None):
        x = super().forward(x)
        o = self.semhead(x, gtsem)
        return o

class HRNet_Ocr(seg_hrnet_ocr.HighResolutionNet):
    def __init__(self, 
                 cls_n,
                 align_corners,
                 ignore_label,
                 stage1_para,
                 stage2_para,
                 stage3_para,
                 stage4_para,
                 final_conv_kernel, 
                 loss_type='ce',
                 intrain_getpred=False,
                 ineval_output_argmax=False,
                 ocr_mc_n = 512,
                 ocr_keyc_n = 256,
                 ocr_dropout_rate = 0.05,
                 ocr_scale = 1,
                 **kwargs):
        from easydict import EasyDict as edict
        config = edict()
        config.MODEL = edict()
        config.MODEL.ALIGN_CORNERS = align_corners
        config.MODEL.EXTRA = {}
        config.MODEL.EXTRA['STAGE1'] = stage1_para
        config.MODEL.EXTRA['STAGE2'] = stage2_para
        config.MODEL.EXTRA['STAGE3'] = stage3_para
        config.MODEL.EXTRA['STAGE4'] = stage4_para
        config.MODEL.EXTRA['FINAL_CONV_KERNEL'] = final_conv_kernel
        config.DATASET = edict()
        config.DATASET.NUM_CLASSES = cls_n

        # OCR special
        config.MODEL.OCR = edict()
        config.MODEL.OCR.MID_CHANNELS = ocr_mc_n
        config.MODEL.OCR.KEY_CHANNELS = ocr_keyc_n
        config.MODEL.OCR.DROPOUT = ocr_dropout_rate
        config.MODEL.OCR.SCALE = ocr_scale

        super().__init__(config)

        self.auxhead = utils.semantic_head_noconv(
            align_corners = align_corners,
            ignore_label = ignore_label,
            loss_type = loss_type,
            intrain_getpred = False,
            ineval_output_argmax = False,
        ) # the aux head (not main)

        self.semhead = utils.semantic_head_noconv(
            align_corners = align_corners,
            ignore_label = ignore_label,
            loss_type = loss_type,
            intrain_getpred = intrain_getpred,
            ineval_output_argmax = ineval_output_argmax,
        )

        self.opmgr = optim_manager(
            group = {'hrnet': 'self'},
            order = ['hrnet'],
        )

    def forward(self, x, gtsem=None):
        x = super().forward(x)
        o = self.semhead(x[1], gtsem)
        if self.training:
            o['lossaux'] = self.auxhead(x[0], gtsem)['losssem']            
        return o


@register(
    'HRNET', 
    {
        'oc_n'                 : 'OUTPUT_CHANNEL_NUM',
        'cls_n'                : 'CLASS_NUM',
        'align_corners'        : 'ALIGN_CORNERS',
        'ignore_label'         : 'IGNORE_LABEL',
        'stage1_para'          : 'STAGE1_PARA',
        'stage2_para'          : 'STAGE2_PARA',
        'stage3_para'          : 'STAGE3_PARA',
        'stage4_para'          : 'STAGE4_PARA',
        'final_conv_kernel'    : 'FINAL_CONV_KERNEL',
        'loss_type'            : 'LOSS_TYPE',
        'intrain_getpred'      : 'INTRAIN_GETPRED', 
        'ineval_output_argmax' : 'INEVAL_OUTPUT_ARGMAX', 
    })
def hrnet(tags, **para):
    if 'base' in tags:
        net = HRNet_Base(**para)
    elif 'ocr' in tags:
        net = HRNet_Ocr(**para)
    else:
        net = HRNet(**para)
    return net
