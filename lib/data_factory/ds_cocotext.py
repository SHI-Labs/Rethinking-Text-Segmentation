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

class common(ds_base):
    def init_load_info(self, mode):
        cfgd = cfguh().cfg.DATA
        annofile = osp.join(cfgd.ROOT_DIR, 'coco_text', 'cocotext.v2.json')
        with open(annofile, 'r') as f:
            annoinfo = json.load(f)

        im_list  = [i for _, i in annoinfo['imgs'].items()]
        im_train = list(filter(lambda x:x['set'] == 'train', im_list))
        im_val   = list(filter(lambda x:x['set'] == 'val'  , im_list))
        im_list  = []
            
        for m in mode.split('+'):
            if m == 'train':
                im_list += im_train
            elif m == 'val':
                im_list += im_val
            else:
                raise ValueError

        self.load_info = []

        for im in im_list:
            filename = im['file_name']
            path = filename.split('_')[1]
            imagepath = osp.join(cfgd.ROOT_DIR, path, filename)
            annids = annoinfo['imgToAnns'][str(im['id'])]
            info = {
                'unique_id' : filename.split('.')[0],
                'filename'  : filename,
                'image_path': imagepath,
                'coco_text_anno' : [annoinfo['anns'][str(i)] for i in annids],
            }
            self.load_info.append(info)

@regdataset()
class cocotext(common):
    def init_load_info(self, mode):
        super().init_load_info(mode)

@regdataset()
class cocots(common):
    def init_load_info(self, mode):
        cfgd = cfguh().cfg.DATA
        super().init_load_info(mode)

        cocots_annopath = osp.join(cfgd.ROOT_DIR, 'coco_ts_labels')

        info_new = []
        for info in self.load_info:
            annf = info['filename'].split('.')[0]+'.png'
            segpath = osp.join(cocots_annopath, annf)
            if osp.exists(segpath):
                info = copy.deepcopy(info)
                info['seglabel_path'] = segpath
                info['bbox_path'] = info['coco_text_anno'] # a hack
                info_new.append(info)
        self.load_info = info_new

    def get_semantic_classname(self,):
        map = {
            0  : 'background' , 
            1  : 'text'       ,
        }
        return map
