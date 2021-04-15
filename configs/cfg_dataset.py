import os
import os.path as osp
import numpy as np
import copy

from easydict import EasyDict as edict

cfg = edict()
cfg.DATASET_MODE = None
cfg.LOADER_PIPELINE = []
cfg.LOAD_BACKEND_IMAGE = 'pil'
cfg.LOAD_IS_MC_IMAGE = False
cfg.TRANS_PIPELINE = []
cfg.NUM_WORKERS_PER_GPU = None
cfg.NUM_WORKERS = None
cfg.TRY_SAMPLE = None

##############################
#####      imagenet      #####
##############################

cfg_imagenet = copy.deepcopy(cfg)
cfg_imagenet.DATASET_NAME = 'imagenet'
cfg_imagenet.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data',
    'ImageNet', 'ILSVRC2012'))
cfg_imagenet.CLASS_INFO_JSON = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data',
    'ImageNet', 'addon', 'ILSVRC2012', '1000nids.json'))
cfg_imagenet.IM_MEAN = [0.485, 0.456, 0.406]
cfg_imagenet.IM_STD = [0.229, 0.224, 0.225]
cfg_imagenet.CLASS_NUM = 1000

#############################
#####      textseg      #####
#############################

cfg_textseg = copy.deepcopy(cfg)
cfg_textseg.DATASET_NAME = 'textseg'
cfg_textseg.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'TextSeg'))
cfg_textseg.CLASS_NUM = 2
cfg_textseg.CLASS_NAME = [
    'background', 
    'text']
cfg_textseg.SEGLABEL_IGNORE_LABEL = 999
cfg_textseg.SEMANTIC_PICK_CLASS = 'all'
cfg_textseg.IM_MEAN = [0.485, 0.456, 0.406]
cfg_textseg.IM_STD = [0.229, 0.224, 0.225]
cfg_textseg.LOAD_IS_MC_SEGLABEL = True

##########################
#####    cocotext    #####
##########################

cfg_cocotext = copy.deepcopy(cfg)
cfg_cocotext.DATASET_NAME = 'cocotext'
cfg_cocotext.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'COCO'))
cfg_cocotext.IM_MEAN = [0.485, 0.456, 0.406]
cfg_cocotext.IM_STD = [0.229, 0.224, 0.225]

########################
#####    cocots    #####
########################

cfg_cocots = copy.deepcopy(cfg)
cfg_cocots.DATASET_NAME = 'cocots'
cfg_cocots.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'COCO'))
cfg_cocots.IM_MEAN = [0.485, 0.456, 0.406]
cfg_cocots.IM_STD = [0.229, 0.224, 0.225]
cfg_cocots.CLASS_NUM = 2
cfg_cocots.SEGLABEL_IGNORE_LABEL = 255
cfg_cocots.LOAD_BACKEND_SEGLABEL = 'pil'
cfg_cocots.LOAD_IS_MC_SEGLABEL = False

#####################
#####    mlt    #####
#####################

cfg_mlt = copy.deepcopy(cfg)
cfg_mlt.DATASET_NAME = 'mlt'
cfg_mlt.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'ICDAR17', 'challenge8'))
cfg_mlt.IM_MEAN = [0.485, 0.456, 0.406]
cfg_mlt.IM_STD = [0.229, 0.224, 0.225]
cfg_mlt.CLASS_NUM = 2
cfg_mlt.SEGLABEL_IGNORE_LABEL = 255

#######################
#####   icdar13   #####
#######################

cfg_icdar13 = copy.deepcopy(cfg)
cfg_icdar13.DATASET_NAME = 'icdar13'
cfg_icdar13.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'ICDAR13'))
cfg_icdar13.CLASS_NUM = 2
cfg_icdar13.CLASS_NAME = [
    'background', 
    'text']
cfg_icdar13.SEGLABEL_IGNORE_LABEL = 999
cfg_icdar13.SEMANTIC_PICK_CLASS = 'all'
cfg_icdar13.IM_MEAN = [0.485, 0.456, 0.406]
cfg_icdar13.IM_STD = [0.229, 0.224, 0.225]
cfg_icdar13.LOAD_BACKEND_SEGLABEL = 'pil'
cfg_icdar13.LOAD_IS_MC_SEGLABEL = False
cfg_icdar13.FROM_SOURCE = 'addon'
cfg_icdar13.USE_CACHE = False

#########################
#####   totaltext   #####
#########################

cfg_totaltext = copy.deepcopy(cfg)
cfg_totaltext.DATASET_NAME = 'totaltext'
cfg_totaltext.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'TotalText'))
cfg_totaltext.CLASS_NUM = 2
cfg_totaltext.CLASS_NAME = [
    'background', 
    'text']
cfg_totaltext.SEGLABEL_IGNORE_LABEL = 999 
# dummy, totaltext pixel level anno has no ignore label
cfg_totaltext.IM_MEAN = [0.485, 0.456, 0.406]
cfg_totaltext.IM_STD = [0.229, 0.224, 0.225]

#######################
#####   textssc   #####
#######################
# text semantic segmentation composed

cfg_textssc = copy.deepcopy(cfg)
cfg_textssc.DATASET_NAME = 'textssc'
cfg_textssc.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'TextSSC'))
cfg_textssc.CLASS_NUM = 2
cfg_textssc.CLASS_NAME = [
    'background', 
    'text']
cfg_textssc.SEGLABEL_IGNORE_LABEL = 999 
cfg_textssc.IM_MEAN = [0.485, 0.456, 0.406]
cfg_textssc.IM_STD = [0.229, 0.224, 0.225]
cfg_textssc.LOAD_BACKEND_SEGLABEL = 'pil'
cfg_textssc.LOAD_IS_MC_SEGLABEL = False
