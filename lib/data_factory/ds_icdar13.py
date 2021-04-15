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

from .ds_base import ds_base, collate, register as regdataset
from .ds_loader import pre_loader_checkings, register as regloader
from .ds_transform import TBase, have, register as regtrans
from .ds_formatter import register as regformat

from .. import nputils
from ..cfg_helper import cfg_unique_holder as cfguh
from ..log_service import print_log

@regdataset()
class icdar13(ds_base):
    def init_load_info(self, mode):
        from_source = cfguh().cfg.DATA.FROM_SOURCE
        if from_source == 'addon':
            return self.init_load_info_fromaddon(mode)
        elif from_source == 'original':
            return self.init_load_info_fromori(mode)
        else:
            raise ValueError

    def init_load_info_fromaddon(self, mode):
        cfgd = cfguh().cfg.DATA
        self.root_dir = cfgd.ROOT_DIR

        dirinfo = {}
        for m in mode.split('+'):
            # train_bdi, test_bdi : born digital image 
            # train_fst, test_fst : focused scene text
            if m == 'train_bdi':
                raise NotImplementedError
            elif m == 'train_fst':
                dirinfo['01_train_fst'] = {
                    'image' : osp.join(
                        self.root_dir, 
                        'Focused_Scene_Text', 'addon', 'train_image'),
                    'anno' : osp.join(
                        self.root_dir, 
                        'Focused_Scene_Text', 'addon', 'train_label'),
                }
            elif m == 'test_bdi':
                raise NotImplementedError
            elif m == 'test_fst':
                dirinfo['51_test_fst'] = {
                    'image' : osp.join(
                        self.root_dir, 
                        'Focused_Scene_Text', 'addon', 'test_image'),
                    'anno' : osp.join(
                        self.root_dir, 
                        'Focused_Scene_Text', 'addon', 'test_label'),
                }
            else:
                raise ValueError

        self.load_info = []
        for tagi, diri in dirinfo.items():
            dirim, diranno = diri['image'], diri['anno']
            for imi in os.listdir(dirim):
                fn = imi.split('.')[0]
                seglabel_path = osp.join(
                    diranno, '{}_seglabel.png'.format(fn))
                wbbox_path = None # so far not supported
                cbbox_path = osp.join(
                    diranno, '{}_charbbox.json'.format(fn))

                if not osp.exists(cbbox_path):
                    cbbox_path = None

                info = {
                    'unique_id': tagi+'_'+fn, 
                    'filename': fn, 
                    'image_path': osp.join(dirim, imi),
                    'seglabel_path': seglabel_path,
                    'bbox_path': [wbbox_path, cbbox_path],
                } 
                self.load_info.append(info)

        if cfgd.USE_CACHE:
            from .ds_loader import get_loader
            for info in self.load_info:
                get_loader()(cfgd.LOADER_PIPELINE)(info)
                if 'image' in info:
                    info['image_cache'] = info.pop('image')
                if 'seglabel' in info:
                    info['seglabel_cache'] = info.pop('seglabel')
                if 'bbox' in info:
                    info['bbox_cache'] = info.pop('bbox')
                # must pop the info otherwise will be error 
                # at pre_loader_check

    def init_load_info_fromori(self, mode):
        cfgd = cfguh().cfg.DATA
        self.root_dir = cfgd.ROOT_DIR

        dirinfo = {}
        for m in mode.split('+'):
            # train_bdi, test_bdi : born digital image 
            # train_fst, test_fst : focused scene text
            if m == 'train_bdi':
                dirinfo['00_train_bdi'] = \
                    osp.join(self.root_dir, 'Born_Digital_Images', 'train')
            elif m == 'train_fst':
                dirinfo['01_train_fst'] = \
                    osp.join(self.root_dir, 'Focused_Scene_Text', 'train')
            elif m == 'test_bdi':
                dirinfo['50_test_bdi'] = \
                    osp.join(self.root_dir, 'Born_Digital_Images', 'test')
            elif m == 'test_fst':
                dirinfo['51_test_fst'] = \
                    osp.join(self.root_dir, 'Focused_Scene_Text', 'test')
            else:
                raise ValueError

        self.load_info = []
        for tagi, diri in dirinfo.items():
            for imi in os.listdir(osp.join(diri, 'images')):
                fn = imi.split('.')[0]
                possible_gt_name = [
                    'gt_{}.png'.format(fn),
                    '{}_GT.bmp'.format(fn),
                ]
                seglabel_path = None
                for pi in possible_gt_name:
                    pi_full = osp.join(
                        diri, 'GT_Text_Segmentation', pi)
                    if osp.exists(pi_full):
                        seglabel_path = pi_full
                if seglabel_path is None:
                    raise ValueError

                word_bbox_path = osp.join(
                    diri, 'GT_Text_Localization', 'gt_{}.txt'.format(fn))
                letter_bbox_path = osp.join(
                    diri, 'GT_Char_Localization', '{}_GT.txt'.format(fn))
                if not osp.exists(letter_bbox_path):
                    letter_bbox_path = None

                info = {
                    'unique_id': tagi+'_'+fn, 
                    'filename': fn, 
                    'image_path': osp.join(diri, 'images', imi),
                    'seglabel_path': seglabel_path,
                    'bbox_path': [word_bbox_path, letter_bbox_path],
                } 
                self.load_info.append(info)

    def get_semantic_classname(self, cls_n=None):
        if cls_n is None:
            cls_n = cfguh().cfg.DATA.CLASS_NUM
        if cls_n == 2:
            map = {
                0  : 'background' , 
                1  : 'text'       ,
            }
        else:
            raise ValueError
        return map

##########
# loader #
##########

@regloader()
class SeglabelLoader(object):
    def __init__(self):
        pass

    @pre_loader_checkings('seglabel')
    def __call__(self, path, element):
        label = np.array(PIL.Image.open(path))
        return label

@regloader({
    'ignore_label' : 'IGNORE_LABEL',})
class CharBboxLoader(object):
    def __init__(self, ignore_label):
        self.map = {}
        self.map.update({chr(48+i) : i    for i in range(10)})
        self.map.update({chr(65+i) : 10+i for i in range(26)})
        self.map.update({chr(97+i) : 10+i for i in range(26)})
        self.ig = ignore_label

    @pre_loader_checkings('bbox')
    def __call__(self, path, element):
        """
        Outputs:
            data: [n x 6] float nparray,
                formatted as <h1, w1, h2, w2, clsid, insid>
        """
        with open(path[1], 'r') as f:
            bbox_jinfo = json.load(f)

        bbox = []

        for ji in bbox_jinfo:
            bbox.append(ji[0:6])

        if len(bbox)==0:
            return np.zeros([0, 6], dtype=float)
        else:
            return np.array(bbox, dtype=float)

@regloader({
    'ignore_label' : 'IGNORE_LABEL',
    'square_bbox'  : 'LOADER_SQUARE_BBOX',})
class CharBboxSpLoader(object):
    """
    The special character bbox loader that load
        character bbox only from seglabel. 
        a) Each character bbox will be squared(an option).
        b) ignore label instance mask will be ignored.
    """
    def __init__(self, 
                 ignore_label, 
                 square_bbox=True,
                 **kwargs):
        """
        Args:
            square_bbox: bool,
                whether to square the box or not.
        """
        self.ig = ignore_label
        self.square_bbox = square_bbox

    @pre_loader_checkings('bbox')
    def __call__(self, path, element):
        return self.load(path, element)

    def load(self, path, element):
        # load only the cls map from json
        with open(path[1], 'r') as f:
            jsoninfo = json.load(f)
        lcmap = {}
        for bboxi in jsoninfo:
            clsid, insid = bboxi[4:6]
            lcmap[insid] = clsid

        ins = element['seglabel']
        insoh = nputils.one_hot_2d(ignore_label=self.ig)(ins)
        bbox = nputils.bbox_binary(is_coord=False)(insoh)
        bbox = [
            list(bi)+[lcmap[idi], idi] \
                for idi, bi in enumerate(bbox) if (bi.sum()!=0) and (idi>0)]
        # idi>0 remove the background

        # square the bbox
        bbox_squared = []
        for h1, w1, h2, w2, clsi, idi in bbox:
            d = (h2-h1) - (w2-w1)
            if d>0:
                w1 -= d//2
                w2 += d-d//2
            else:
                h1 -= -(d//2)
                h2 += -(d-d//2)
            bbox_squared.append([h1, w1, h2, w2, clsi, idi])

        if (len(bbox) != 0) and (self.square_bbox):
            element['bbox_squared'] = np.array(bbox_squared).astype(int)
            bbox = element['bbox_squared'].astype(float)
        elif (len(bbox) != 0) and (not self.square_bbox):
            element['bbox_squared'] = np.array(bbox_squared).astype(int)
            bbox = np.array(bbox).astype(float)
        else:
            element['bbox_squared'] = np.zeros((0, 6)).astype(int)
            bbox = np.zeros((0, 6)).astype(float)
        return bbox

@regloader({
    'ignore_label' : 'IGNORE_LABEL',})
class Icdar13_SeglabelLoader_Original(object):
    """
    Function to process the original dataset label files.
    """
    def __init__(self, ignore_label):
        self.ig = ignore_label

    @pre_loader_checkings('seglabel')
    def __call__(self, path, element):
        # only red channel is useful
        label = np.array(PIL.Image.open(path).convert('RGB'))
        label = np.transpose(label, (2, 0, 1)).astype(int)
        if 'char_colormap' not in element:
            # the loaded seglabel remains its RGB value
            return label
        else:
            # the loaded seglabel will be remap based on bbox info
            label = label[0]*256*256 + label[1]*256 + label[2]
            cmap = element['char_colormap']
            label = np.vectorize(
                lambda x: cmap[x] if x in cmap else self.ig)(label)
        return label

@regloader({
    'ignore_label' : 'IGNORE_LABEL',})
class Icdar13_CharBboxLoader_Original(object):
    def __init__(self, ignore_label):
        self.map = {}
        self.map.update({chr(48+i) : i    for i in range(10)})
        self.map.update({chr(65+i) : 10+i for i in range(26)})
        self.map.update({chr(97+i) : 10+i for i in range(26)})
        self.ig = ignore_label

    @pre_loader_checkings('bbox')
    def __call__(self, path, element):
        """
        Outputs:
            data: [n x 10] float nparray,
                formatted as <h1, w1, h2, w2, h3, w3, h4, w4, clsid, insid>
        """
        path = path[1]
        if path is None:
            return np.zeros((0, 10), dtype=int)

        with open(path, 'r') as f:
            l = f.readlines()
    
        bbox = []
        bbox_ignore = []
        cmap = {255*256*256 + 255*256 + 255 : 0} # white is bg
        for li in l:
            isignore = False
            if li[0] == '#':
                # the ignore case
                li = li[1:]
                isignore = True

            lsplit = li.strip().split(' ')
            if len(lsplit) != 10:
                continue

            color = [int(i) for i in lsplit[:3]]
            coord = [int(i) for i in lsplit[5:9]]
            char = lsplit[-1].split('"')[1]

            clsid = self.map[char] if char in self.map else 36

            if not isignore:
                insid = len(bbox) + 1
                w1, h1, w2, h2 = coord
                bbox.append([h1, w1, h2, w2, clsid, insid, char])
            else:
                insid = self.ig
                w1, h1, w2, h2 = coord
                bbox_ignore.append([h1, w1, h2, w2, clsid, insid, char])

            r, g, b = color
            cmap[r*256*256 + g*256 + b] = insid

        element['char_colormap'] = cmap

        if 'seglabel' in element:
            seg = element['seglabel']
            if len(seg.shape)==3:
                # unconverted seglabel
                seg = seg[0]*256*256 + seg[1]*256 + seg[2]
                seg = np.vectorize(
                    lambda x: cmap[x] if x in cmap else self.ig)(seg)
                element['seglabel'] = seg

        bbox = bbox + bbox_ignore
        return bbox

##############
# formatters #
##############

@regformat({
    'ignore_label' : 'IGNORE_LABEL', })
class SemanticFormatter(object):
    def __init__(self,
                 ignore_label,
                 **kwargs):
        self.ignore_label = ignore_label

    def __call__(self, element):
        im = element['image']
        seglabel = element['seglabel']
        igmask = seglabel == self.ignore_label
        seglabel[seglabel>0] = 1
        seglabel[igmask] = self.ignore_label
        return im.astype(np.float32), seglabel.astype(int), element['unique_id']

@regformat()
class InstanceFormatter(object):
    def __init__(self, 
                 **kwargs):
        pass

    def __call__(self, element):
        im = element['image']
        seglabel = element['seglabel']
        return im, seglabel, element['unique_id']

@regformat()
class PanopticFormatter(object):
    def __init__(self, 
                 ignore_label, 
                 **kwargs):
        self.SemanticFormatter = SemanticFormatter(ignore_label)
        self.InstanceFormatter = InstanceFormatter()

    def __call__(self, element):
        im, sem, _ = self.SemanticFormatter(element)
        _, ins, _ = self.InstanceFormatter(element)
        pan = np.stack([sem, ins])
        return im, pan, element['unique_id']

@regformat()
class DsmakerFormatter(object):
    def __init__(self,
                 **kwargs):
        pass

    def __call__(self, element):
        seglabel = element['seglabel']
        bbox = element['bbox']
        uid = element['unique_id']
        return seglabel, [bbox], uid

@regformat({
    'ignore_label' : 'IGNORE_LABEL', })
class SemChinsChbbxFormatter(SemanticFormatter):
    """
    This formatter output semantic/instance/bbox
        bbox will be filtered out if 
            a) part of it is outside image 
                (caused by random crop or square)
            b) its clsid is 36 (other)
    """
    def __call__(self, element):
        im, sem, uid = super().__call__(element)
        h, w = im.shape[-2:]
        bbox = element['bbox']
        h1, w1, h2, w2, clsid, _ = bbox.T
        valid = (h1>=0) & (w1>=0) & (h2<=h) & (w2<=w) & (clsid!=36)
        bbox = bbox[valid].astype(np.float32)
        # chins = element['chins'].astype(np.float32)
        # chcls = element['chcls'].astype(int)
        # so far not supported
        return im, sem, [bbox], ['not_supported'], ['not_supported'], uid
