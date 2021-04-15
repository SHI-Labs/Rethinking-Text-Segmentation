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
PIL.Image.MAX_IMAGE_PIXELS = None

from .ds_base import ds_base, register as regdataset
from .ds_loader import pre_loader_checkings, register as regloader
from .ds_transform import TBase, have, register as regtrans
from .ds_formatter import register as regformat

from .. import nputils
from ..cfg_helper import cfg_unique_holder as cfguh
from ..log_service import print_log

@regdataset()
class totaltext(ds_base):
    def init_load_info(self, mode):
        cfgd = cfguh().cfg.DATA
        self.root_dir = cfgd.ROOT_DIR

        im_path = []
        seg_path = []

        for mi in mode.split('+'):
            if mi == 'train':
                im_path  += [osp.join(self.root_dir, 'Images', 'Train')]
                seg_path += [osp.join(self.root_dir, 'groundtruth_pixel', 'Train')]
            elif mi == 'test':
                im_path  += [osp.join(self.root_dir, 'Images', 'Test')]
                seg_path += [osp.join(self.root_dir, 'groundtruth_pixel', 'Test')]
            else:
                raise ValueError

        self.load_info = []
        for impi, segpi in zip(im_path, seg_path):
            for fi in os.listdir(impi):
                uid = fi.split('.')[0]
                self.load_info.append({
                    'unique_id'     : uid,
                    'filename'      : fi,
                    'image_path'    : osp.join(impi, fi),
                    'seglabel_path' : osp.join(segpi, uid+'.jpg'),
                })

    def get_semantic_classname(self,):
        map = {
            0  : 'background' , 
            1  : 'text'       ,
        }
        return map

# ---- loader ----

@regloader()
class TotalText_SeglabelLoader(object):
    def __init__(self):
        pass

    @pre_loader_checkings('seglabel')
    def __call__(self, path, element):
        sem = np.array(PIL.Image.open(path)).astype(int)
        sem = (sem>127).astype(int)
        return sem
