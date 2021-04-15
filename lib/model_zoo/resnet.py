import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import nputils
from .. import torchutils
from .get_model import get_model, register
from .optim_manager import optim_manager
from . import utils

VERSION = 'v12'

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None,
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu'):
        super().__init__()

        conv, bn, relu = utils.conv_bn_relu(conv_type, bn_type, relu_type)
        if bn_type in ['bn', 'syncbn', 'none']:
            bn_default = bn
            relu_default = relu
        else:
            bn_default   = utils.nn_component('bn')
            relu_default = utils.nn_component(relu_type)

        self.conv1 = conv(inplanes, planes, 3, stride, 1)
        self.bn1   = bn(planes)
        self.conv2 = conv(inplanes, planes, 3, 1, 1)
        self.bn2   = bn_default(planes)
        self.relu  = relu(inplace=True)
        self.relu_default = relu_default(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu_default(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None,
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu'):
        super().__init__()

        conv, bn, relu = utils.conv_bn_relu(conv_type, bn_type, relu_type)
        if bn_type in ['bn', 'syncbn', 'none']:
            bn_default = bn
            relu_default = relu
        else:
            bn_default   = utils.nn_component('bn')
            relu_default = utils.nn_component(relu_type)

        self.conv1 = conv(inplanes, planes, 1, 1, 0)
        self.bn1   = bn(planes)
        self.conv2 = conv(planes, planes, 3, stride, 1)
        self.bn2   = bn(planes)
        self.conv3 = conv(planes, planes*self.expansion, 1, 1, 0)
        self.bn3   = bn_default(planes*self.expansion)
        self.relu  = relu(inplace=True)
        self.relu_default = relu_default(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu_default(out)
        return out

class ResNet_Base(nn.Module):
    def __init__(self, 
                 block, 
                 layer_n, 
                 ic_n=3,
                 use_maxpool=True,
                 output_layer_type='last',
                 zero_init_residual=False,
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 **kwargs):
        super(ResNet_Base, self).__init__()
        self.inplanes = 64
        self.block_expansion = block.expansion

        conv, bn, relu = utils.conv_bn_relu(conv_type, bn_type, relu_type)

        self.conv1 = conv(ic_n, 64, 7, 2, 3)
        self.bn1   = bn(64)
        self.relu  = relu(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,  64, layer_n[0],           
            conv_type=conv_type,
            bn_type=bn_type, relu_type=relu_type)
        self.layer2 = self._make_layer(
            block, 128, layer_n[1], stride=2, 
            conv_type=conv_type,
            bn_type=bn_type, relu_type=relu_type)
        self.layer3 = self._make_layer(
            block, 256, layer_n[2], stride=2, 
            conv_type=conv_type,
            bn_type=bn_type, relu_type=relu_type)
        self.layer4 = self._make_layer(
            block, 512, layer_n[3], stride=2, 
            conv_type=conv_type,
            bn_type=bn_type, relu_type=relu_type)

        self.use_maxpool = use_maxpool
        self.output_layer_type = output_layer_type

        for m in self.modules():
            utils.common_init(m)

        if zero_init_residual and (bn_type in ['bn', 'syncbn']):
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.opmgr = optim_manager(
            group = {'resnet': 'self'},
            order = ['resnet'],
        )

    def _make_layer(self, 
                    block, 
                    planes, 
                    blocks, 
                    stride=1,
                    conv_type='conv',
                    bn_type='bn',
                    relu_type='relu'):
        downsample = None
        conv = utils.nn_component(conv_type)
        if bn_type in ['bn', 'syncbn', 'none']:
            bn = utils.nn_component(bn_type)

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes, 
                    planes*block.expansion, 
                    1, stride, 0),
                bn(planes*block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, 
            conv_type=conv_type, bn_type=bn_type, 
            relu_type=relu_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, 
                conv_type=conv_type, 
                bn_type=bn_type, 
                relu_type=relu_type))

        return nn.Sequential(*layers)

    def forward(self, 
                x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        if self.use_maxpool:
            x = self.maxpool(x0)
        else:
            x = x0
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if self.output_layer_type == '5layers':
            return x0, x1, x2, x3, x4
        elif self.output_layer_type == '4layers':
            return x1, x2, x3, x4
        elif self.output_layer_type == 'last':
            return x4
        else:
            raise ValueError

class ResNet(ResNet_Base):
    def __init__(self,
                 cls_n,
                 block, 
                 layer_n, 
                 ic_n=3,
                 use_maxpool=True,
                 zero_init_residual=False,
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 ignore_label=None,
                 **kwargs):

        super(ResNet, self).__init__(
            block,
            layer_n, 
            ic_n,
            use_maxpool,
            output_layer_type='last',
            zero_init_residual=zero_init_residual,
            conv_type=conv_type,
            bn_type=bn_type,
            relu_type=relu_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = utils.classification_head(
            512*block.expansion, cls_n, 
            ignore_label=ignore_label)

        self.opmgr = optim_manager(
            group = {'resnet': 'self'},
            order = ['resnet'],
        )

    def forward(self, 
                x, 
                gtcls=None):
        x = super().forward(x)
        x = self.avgpool(x)
        o = self.cls_head(x, gtcls)
        return o

############################
#####     deeplab      #####
############################

class BasicBlockD(BasicBlock):
    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 dilation=1, 
                 downsample=None, 
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu'):
        super(BasicBlockD, self).__init__(
            inplanes, planes, stride, downsample, 
            bn_type=bn_type, relu_type=relu_type)
        conv, _, _ = utils.conv_bn_relu(conv_type, bn_type, relu_type)
        self.conv1 = conv(
            inplanes, planes, kernel_size=3, stride=stride, 
            padding=dilation, dilation=dilation)

class BottleneckD(Bottleneck):
    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 dilation=1, 
                 downsample=None, 
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu'):
        super(BottleneckD, self).__init__(
            inplanes, planes, stride, downsample, 
            conv_type = conv_type,
            bn_type=bn_type, relu_type=relu_type)
        conv, _, _ = utils.conv_bn_relu(conv_type, bn_type, relu_type)
        self.conv2 = conv(
            planes, planes, kernel_size=3, stride=stride, 
            padding=dilation, dilation=dilation)

class ResNet_Dilated_Base(ResNet_Base):
    def __init__(self, 
                 block, 
                 layer_n, 
                 stride_n=[1, 2, 2, 2],
                 dilation_n=[1, 1, 1, 1],
                 layer4_multi_grid=[1, 1, 1],
                 ic_n=3, 
                 use_maxpool=True,
                 output_layer_type='4layers',
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 **kwargs):
        super().__init__(
            block, layer_n, ic_n, use_maxpool, output_layer_type,
            zero_init_residual=False, 
            conv_type=conv_type, bn_type=bn_type, relu_type=relu_type)
        
        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #, ceil_mode=True)
        self.layer1 = self._make_layer(
            block,  64, layer_n[0], stride=stride_n[0], 
            dilation=dilation_n[0], 
            conv_type=conv_type, 
            bn_type=bn_type, relu_type=relu_type)
        self.layer2 = self._make_layer(
            block, 128, layer_n[1], stride=stride_n[1], 
            dilation=dilation_n[1], 
            conv_type=conv_type, 
            bn_type=bn_type, relu_type=relu_type)
        self.layer3 = self._make_layer(
            block, 256, layer_n[2], stride=stride_n[2], 
            dilation=dilation_n[2], 
            conv_type=conv_type, 
            bn_type=bn_type, relu_type=relu_type)
        self.layer4 = self._make_layer(
            block, 512, layer_n[3], stride=stride_n[3], dilation=dilation_n[3],
            multi_grid=layer4_multi_grid, 
            conv_type=conv_type, 
            bn_type=bn_type, relu_type=relu_type)
        
        for m in self.modules():
            utils.common_init(m)

        self.opmgr = optim_manager(
            group = {'resnet': 'self'},
            order = ['resnet'],
        )

    def _make_layer(self, 
                    block, 
                    planes, 
                    blocks, 
                    stride=1, 
                    dilation=1, 
                    multi_grid=None,
                    conv_type='conv',
                    bn_type='bn',
                    relu_type='relu'):

        if multi_grid is None:
            multi_grid = [1] * blocks
        if blocks != len(multi_grid):
            raise ValueError

        if (stride == 1) and (dilation == 1) \
                and (self.inplanes == planes*block.expansion):
            downsample = None
        else:
            conv = utils.nn_component(conv_type)
            if bn_type in ['bn', 'syncbn', 'none']:
                bn = utils.nn_component(bn_type)
            else:
                bn = utils.nn_component('bn')
            downsample = nn.Sequential(
                conv(
                    self.inplanes, 
                    planes*block.expansion, 
                    1, stride, 0),
                bn(planes*block.expansion))

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, 
                dilation*multi_grid[0], downsample,
                conv_type=conv_type,
                bn_type=bn_type,
                relu_type=relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, 1, 
                    dilation*multi_grid[i], 
                    downsample=None, 
                    conv_type=conv_type,
                    bn_type=bn_type,
                    relu_type=relu_type))

        return nn.Sequential(*layers)

class ResNet_Cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

        layers.append(block(self.in_planes, planes, stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1, downsample=None))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _make_layer_old(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

@register(
    'RESNET', 
    {
        # base & regular
        'ic_n'              : 'INPUT_CHANNEL_NUM',
        'conv_type'         : 'CONV_TYPE',
        'bn_type'           : 'BN_TYPE',
        'relu_type'         : 'RELU_TYPE',
        'use_maxpool'       : 'USE_MAXPOOL',
        'output_layer_type' : 'OUTPUT_LAYER_TYPE',
        'zero_init_residual': 'ZERO_INIT_RESIDUAL',
        'cls_n'             : 'CLASS_NUM',
        'ignore_label'      : 'IGNORE_LABEL',
        # dilated
        'stride_n'          : 'STRIDE_NUM',
        'dilation_n'        : 'DILATION_NUM',
        'layer4_multi_grid' : 'LAYER4_MULTI_GRID',
    })
def resnet(tags, **para):
    if 'dilated' in tags:
        basicblock = BasicBlockD
        bottleneck = BottleneckD
    else:
        basicblock = BasicBlock
        bottleneck = Bottleneck

    if 'resnet18' in tags:
        para['block']   = basicblock
        para['layer_n'] = [2, 2, 2, 2]
    elif 'resnet34' in tags:
        para['block']   = basicblock
        para['layer_n'] = [3, 4, 6, 3]
    elif 'resnet50' in tags:
        para['block']   = bottleneck
        para['layer_n'] = [3, 4, 6, 3]
    elif 'resnet101' in tags:
        para['block']   = bottleneck
        para['layer_n'] = [3, 4, 23, 3]
    elif 'resnet152' in tags:
        para['block']   = bottleneck
        para['layer_n'] = [3, 8, 36, 3]
    else:
        raise ValueError        

    if 'base' in tags:
        if 'dilated' in tags:
            if 'os16' in tags:
                para['stride_n']   = [1, 2, 2, 1]
                para['dilation_n'] = [1, 1, 1, 2]
                para['layer4_multi_grid'] = [1, 2, 4]
                para['output_layer_type'] = '4layers'
            elif 'os8' in tags:
                para['stride_n']   = [1, 2, 1, 1]
                para['dilation_n'] = [1, 1, 2, 4]
                para['layer4_multi_grid'] = [1, 2, 4]
                para['output_layer_type'] = '4layers'
            else:
                pass
            net = ResNet_Dilated_Base(**para)
        else:
            net = ResNet_Base(**para)
    else:
        net = ResNet(**para)
    return net
