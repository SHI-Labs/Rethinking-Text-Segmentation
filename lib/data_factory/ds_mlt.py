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
class mlt(ds_base):
    def init_load_info(self, mode):
        cfgd = cfguh().cfg.DATA
        self.root_dir = cfgd.ROOT_DIR

        trainlist = self.get_trainlist()
        vallist   = self.get_vallist()

        trainlist = [i for _, i in trainlist.items()]
        vallist   = [i for _, i in vallist.items()  ]

        self.load_info = []
        for mi in mode.split('+'):
            if mi == 'train':
                self.load_info += trainlist
            elif mi == 'trainseg':
                self.load_info += [
                    i for i in trainlist if i['seglabel_path'] is not None]
            elif mi == 'val':
                self.load_info += vallist
            elif mi == 'valseg':
                self.load_info += [
                    i for i in vallist if i['seglabel_path'] is not None]
            else:
                raise ValueError

    def get_trainlist(self):
        # get train data
        imdir_list = [
            'ch8_training_images_1', 'ch8_training_images_2', 
            'ch8_training_images_3', 'ch8_training_images_4', 
            'ch8_training_images_5', 'ch8_training_images_6', 
            'ch8_training_images_7', 'ch8_training_images_8', ]

        # get image info
        trainlist = {}
        for di in imdir_list:
            for fi in os.listdir(osp.join(self.root_dir, di)):
                fname = fi.split('.')[0]
                fpath = osp.join(self.root_dir, di, fi)
                trainlist[fname] = {
                    'unique_id'  : '00_train_' + fname,
                    'filename'   : fi,
                    'image_path' : fpath}

        # get bpoly label info
        labeldir = 'ch8_training_localization_transcription_gt_v2'
        for fi in trainlist.keys():
            bpoly_path = osp.join(self.root_dir, labeldir, 'gt_'+fi+'.txt')
            if osp.exists(bpoly_path):
                trainlist[fi]['bpoly_path'] = bpoly_path
            else:
                raise ValueError

        # get seglabel info
        labeldir = osp.join('MLT_S_labels', 'training_labels')
        for fi in trainlist.keys():
            seglabel_path = osp.join(self.root_dir, labeldir, fi+'.png')
            if osp.exists(seglabel_path):
                trainlist[fi]['seglabel_path'] = seglabel_path
            else:
                trainlist[fi]['seglabel_path'] = None        
        return trainlist

    def get_vallist(self):
        # get train data
        imdir_list = [
            'ch8_validation_images', ]

        # get image info
        vallist = {}
        for di in imdir_list:
            for fi in os.listdir(osp.join(self.root_dir, di)):
                fname = fi.split('.')[0]
                fpath = osp.join(self.root_dir, di, fi)
                vallist[fname] = {
                    'unique_id'  : '01_val_' + fname,
                    'filename'   : fi,
                    'image_path' : fpath}

        # get bpoly label info
        labeldir = 'ch8_validation_localization_transcription_gt_v2'
        for fi in vallist.keys():
            bpoly_path = osp.join(self.root_dir, labeldir, 'gt_'+fi+'.txt')
            if osp.exists(bpoly_path):
                vallist[fi]['bpoly_path'] = bpoly_path
            else:
                raise ValueError

        # get seglabel info
        labeldir = osp.join('MLT_S_labels', 'validation_labels')
        for fi in vallist.keys():
            seglabel_path = osp.join(self.root_dir, labeldir, fi+'.png')
            if osp.exists(seglabel_path):
                vallist[fi]['seglabel_path'] = seglabel_path
            else:
                vallist[fi]['seglabel_path'] = None        
        return vallist

    def get_semantic_classname(self,):
        map = {
            0  : 'background' , 
            1  : 'text'       ,
        }
        return map

# ---- loader ----

@regloader()
class Mlt_SeglabelLoader(object):
    def __init__(self):
        pass

    @pre_loader_checkings('seglabel')
    def __call__(self, path, element):
        sem = np.array(PIL.Image.open(path)).astype(int) #.convert('RGB'))
        return sem
