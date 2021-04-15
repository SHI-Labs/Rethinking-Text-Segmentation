import os
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
import torchvision
import PIL
import json
import cv2
import copy
import scipy
import pandas
PIL.Image.MAX_IMAGE_PIXELS = None

from .ds_base import ds_base, register as regdataset
from .ds_loader import pre_loader_checkings, register as regloader
from .ds_transform import TBase, have, register as regtrans
from .ds_formatter import register as regformat

from .. import nputils
from ..cfg_helper import cfg_unique_holder as cfguh
from ..log_service import print_log

@regdataset()
class textssc(ds_base):
    def init_load_info(self, mode):
        cfgd = cfguh().cfg.DATA
        self.root_dir = cfgd.ROOT_DIR

        imdir = []
        segdir = []
        for modei in mode.split('+'):
            dsi, seti = modei.split('_')
            imdir  += [osp.join(self.root_dir, dsi, 'image_'+seti)]
            segdir += [osp.join(self.root_dir, dsi, 'seglabel_'+seti)]

        self.load_info = []

        for imdiri, segdiri in zip(imdir, segdir): 
            for fi in os.listdir(imdiri):
                ftag = fi.split('.')[0]
                info = {
                    'unique_id'     : ftag,
                    'filename'      : fi,
                    'image_path'    : osp.join(imdiri, fi),
                    'seglabel_path' : osp.join(segdiri, ftag+'.png'),
                }
                self.load_info.append(info)

    def get_semantic_classname(self,):
        map = {
            0  : 'background' , 
            1  : 'text'       ,
        }
        return map
