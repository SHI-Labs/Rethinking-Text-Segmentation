import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import nputils
from .. import torchutils
from .. import loss 
from .get_model import get_model, register
from .optim_manager import optim_manager

from . import utils

version = 'v33'

class ASPP(nn.Module):
    def __init__(self, 
                 ic_n, 
                 c_n, 
                 oc_n, 
                 dilation_n,
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 dropout_type='dropout|0.5',
                 with_gap=True,
                 **kwargs):
        super().__init__()

        conv, bn, relu = utils.conv_bn_relu(conv_type, bn_type, relu_type)
        dropout = utils.nn_component(dropout_type)

        d1, d2, d3 = dilation_n
        self.conv1 = nn.Sequential(
            conv(ic_n, c_n, 1, 1, padding=0, dilation=1),
            bn(c_n),
            relu(inplace=True))
        self.conv2 = nn.Sequential(
            conv(ic_n, c_n, 3, 1, padding=d1, dilation=d1),
            bn(c_n),
            relu(inplace=True))
        self.conv3 = nn.Sequential(
            conv(ic_n, c_n, 3, 1, padding=d2, dilation=d2),
            bn(c_n),
            relu(inplace=True))
        self.conv4 = nn.Sequential(
            conv(ic_n, c_n, 3, 1, padding=d3, dilation=d3),
            bn(c_n),
            relu(inplace=True))
        if with_gap:
            self.conv5 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                conv(ic_n, c_n, 1, 1, padding=0, dilation=1),
                bn(c_n),
                relu(inplace=True))
            total_layers=5
        else:
            total_layers=4

        self.bottleneck = nn.Sequential(
            conv(c_n*total_layers, oc_n, 1, 1, 0),
            bn(oc_n),
            relu(inplace=True),)
        self.dropout = dropout()
        self.with_gap = with_gap

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gap:
            feat5 = F.interpolate(
                self.conv5(x), size=(h, w), mode='nearest')
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        else:
            out = torch.cat((feat1, feat2, feat3, feat4), dim=1)
        out = self.bottleneck(out)
        out = self.dropout(out)
        return out

class Decoder(nn.Module):
    def __init__(self, 
                 bic_n, 
                 xic_n, 
                 oc_n,
                 align_corners=False,
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 dropout2_type='dropout|0.5',
                 dropout3_type='dropout|0.1',
                 **kwargs):
        super().__init__()

        conv, bn, relu = utils.conv_bn_relu(conv_type, bn_type, relu_type)
        dropout2 = utils.nn_component(dropout2_type)
        dropout3 = utils.nn_component(dropout3_type)

        self.conv1 = conv(bic_n, 48, 1, 1, 0)
        self.bn1   = bn(48)
        self.relu  = relu(inplace=True)

        self.conv2 = conv(xic_n+48, 256, 3, 1, 1)
        self.bn2   = bn(256)
        self.dropout2 = dropout2()

        self.conv3 = conv(256, oc_n, 3, 1, 1)
        self.bn3   = bn(oc_n)
        self.dropout3 = dropout3()

        self.align_corners = align_corners

    def forward(self, 
                b, 
                x):
        b = self.relu(self.bn1(self.conv1(b)))
        x = torchutils.interpolate_2d(
            size = b.shape[2:], mode='bilinear', 
            align_corners=self.align_corners)(x)
        x = torch.cat([x, b], dim=1)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        return x

class DeepLabv3p_Base(nn.Module):
    def __init__(self, 
                 bbn_name,
                 bbn, 
                 oc_n, 
                 aspp_ic_n,
                 aspp_dilation_n,
                 decoder_bic_n,
                 aspp_dropout_type='dropout|0.5',
                 aspp_with_gap=True,
                 decoder_dropout2_type='dropout|0.5',
                 decoder_dropout3_type='dropout|0.1',
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 align_corners=False, 
                 **kwargs):
        super().__init__()

        setattr(self, bbn_name, bbn)
        self.aspp = ASPP(
            aspp_ic_n, 256, 256, aspp_dilation_n, 
            conv_type=conv_type,
            bn_type=bn_type,
            relu_type=relu_type,
            dropout_type=aspp_dropout_type,
            with_gap=aspp_with_gap)
        self.decoder = Decoder(
            decoder_bic_n, 256, oc_n,
            align_corners=align_corners,
            conv_type=conv_type,
            bn_type=bn_type,
            relu_type=relu_type,
            dropout2_type=decoder_dropout2_type,
            dropout3_type=decoder_dropout3_type,)

        self.bbn_name = bbn_name

        # initialize the weight
        utils.init_module([self.aspp, self.decoder])

        # prepare opmgr
        self.opmgr = getattr(self, self.bbn_name).opmgr
        self.opmgr.inheritant(self.bbn_name)
        self.opmgr.pushback(
            'deeplab', ['aspp', 'decoder']
        )

    def forward(self, x):
        xs = getattr(self, self.bbn_name)(x)
        b, x = xs[0], xs[-1]
        x = self.aspp(x)
        x = self.decoder(b, x)
        return x

class DeepLabv3p(DeepLabv3p_Base):
    def __init__(self, 
                 bbn_name,
                 bbn, 
                 class_n, 
                 aspp_ic_n,
                 aspp_dilation_n,
                 decoder_bic_n,
                 aspp_dropout_type='dropout|0.5',
                 aspp_with_gap=True,
                 decoder_dropout2_type='dropout|0.5',
                 decoder_dropout3_type='dropout|0.1',
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 align_corners=False, 
                 ignore_label=None,
                 loss_type='ce',
                 intrain_getpred=False,
                 ineval_output_argmax=True,
                 **kwargs):

        super().__init__(
            bbn_name,
            bbn, 
            256, 
            aspp_ic_n,
            aspp_dilation_n,
            decoder_bic_n,
            aspp_dropout_type,
            aspp_with_gap,
            decoder_dropout2_type,
            decoder_dropout3_type,
            conv_type,
            bn_type,
            relu_type,
            align_corners,
        )

        self.semhead = utils.semantic_head(
            256, class_n,
            align_corners=align_corners,
            ignore_label=ignore_label,
            loss_type=loss_type,
            ineval_output_argmax=ineval_output_argmax,
            intrain_getpred = intrain_getpred,
        )
        
        # initialize the weight 
        # (aspp and decoder initalized in base class)
        utils.init_module([self.semhead])

        # prepare opmgr
        module_name = self.opmgr.popback()
        module_name.append('semhead')
        self.opmgr.pushback('deeplab', module_name)

    def forward(self, 
                x, 
                gtsem=None,
            ):
        x = super().forward(x)
        o = self.semhead(x, gtsem)
        return o

@register(
    'DEEPLAB', 
    {
        # base
        'freeze_backbone_bn'    : 'FREEZE_BACKBONE_BN',
        'oc_n'                  : 'OUTPUT_CHANNEL_NUM',
        'conv_type'             : 'CONV_TYPE',
        'bn_type'               : 'BN_TYPE',
        'relu_type'             : 'RELU_TYPE',
        'aspp_dropout_type'     : 'ASPP_DROPOUT_TYPE',
        'aspp_with_gap'         : 'ASPP_WITH_GAP',
        'decoder_dropout2_type' : 'DECODER_DROPOUT2_TYPE',
        'decoder_dropout3_type' : 'DECODER_DROPOUT3_TYPE',
        'align_corners'         : 'INTERPOLATE_ALIGN_CORNERS',
        # non_base
        'ignore_label'          : 'SEMANTIC_IGNORE_LABEL',
        'class_n'               : 'SEMANTIC_CLASS_NUM',
        'loss_type'             : 'LOSS_TYPE',
        'intrain_getpred'       : 'INTRAIN_GETPRED', 
        'ineval_output_argmax'  : 'INEVAL_OUTPUT_ARGMAX', 
    })
def deeplab(tags, **para):
    if 'resnet' in tags:
        bbn = get_model()('resnet')
        para['bbn_name']  = 'resnet'
        para['bbn']       = bbn
        para['aspp_ic_n'] = 512*bbn.block_expansion
        para['decoder_bic_n'] = 64*bbn.block_expansion

        if 'os16' in tags:
            para['aspp_dilation_n'] = [6, 12, 18]
        elif 'os8' in tags:
            para['aspp_dilation_n'] = [12, 24, 36]
        else:
            raise ValueError

    try:
        freezebn = para.pop('freeze_backbone_bn')
    except:
        freezebn = False
    if freezebn:
        for m in bbn.modules():
            if isinstance(m, nn.BatchNorm2d):
                for i in m.parameters():
                    i.requires_grad = False

    if 'v3+' in tags:
        if 'base' in tags:
            net = DeepLabv3p_Base(**para)
        else:
            net = DeepLabv3p(**para)

    return net
