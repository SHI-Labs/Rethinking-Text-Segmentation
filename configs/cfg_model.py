import os
import os.path as osp
import numpy as np
import copy

from easydict import EasyDict as edict

cfg = edict()
cfg.MODEL_NAME = None
# cfg.CONV_TYPE = 'conv'
# cfg.BN_TYPE = 'bn'
# cfg.RELU_TYPE = 'relu'

# resnet
cfg_resnet = copy.deepcopy(cfg)
cfg_resnet.MODEL_NAME = 'resnet'
cfg_resnet.RESNET = edict()
cfg_resnet.RESNET.MODEL_TAGS = None
cfg_resnet.RESNET.PRETRAINED_PTH = None
cfg_resnet.RESNET.BN_TYPE = 'bn'
cfg_resnet.RESNET.RELU_TYPE = 'relu'

# deeplab
cfg_deeplab = copy.deepcopy(cfg)
cfg_deeplab.MODEL_NAME = 'deeplab'
cfg_deeplab.DEEPLAB = edict()
cfg_deeplab.DEEPLAB.MODEL_TAGS = None
cfg_deeplab.DEEPLAB.PRETRAINED_PTH = None
cfg_deeplab.DEEPLAB.FREEZE_BACKBONE_BN = False
cfg_deeplab.DEEPLAB.BN_TYPE = 'bn'
cfg_deeplab.DEEPLAB.RELU_TYPE = 'relu'
# cfg_deeplab.DEEPLAB.ASPP_DROPOUT_TYPE = 'dropout|0.5'
cfg_deeplab.DEEPLAB.ASPP_WITH_GAP = True
# cfg_deeplab.DEEPLAB.DECODER_DROPOUT2_TYPE = 'dropout|0.5'
# cfg_deeplab.DEEPLAB.DECODER_DROPOUT3_TYPE = 'dropout|0.1'
cfg_deeplab.RESNET = cfg_resnet.RESNET

# hrnet
cfg_hrnet = copy.deepcopy(cfg)
cfg_hrnet.MODEL_NAME = 'hrnet'
cfg_hrnet.HRNET = edict()
cfg_hrnet.HRNET.MODEL_TAGS = None
cfg_hrnet.HRNET.PRETRAINED_PTH = None
cfg_hrnet.HRNET.BN_TYPE = 'bn'
cfg_hrnet.HRNET.RELU_TYPE = 'relu'

# texrnet
cfg_texrnet = copy.deepcopy(cfg)
cfg_texrnet.MODEL_NAME = 'texrnet'
cfg_texrnet.TEXRNET = edict()
cfg_texrnet.TEXRNET.MODEL_TAGS = None
cfg_texrnet.TEXRNET.PRETRAINED_PTH = None
cfg_texrnet.RESNET = cfg_resnet.RESNET
cfg_texrnet.DEEPLAB = cfg_deeplab.DEEPLAB
