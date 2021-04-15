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
from .ds_transform import CenterCrop
from .ds_formatter import register as regformat

from .. import nputils
from ..cfg_helper import cfg_unique_holder as cfguh
from ..log_service import print_log

import math

@regdataset()
class textseg(ds_base):
    def init_load_info(self, mode):
        cfgd = cfguh().cfg.DATA
        self.root_dir = cfgd.ROOT_DIR

        # get split info
        with open(osp.join(self.root_dir, 'split.json'), 'r') as f:
            split = json.load(f)

        sampletag = []
        for m in mode.split('+'):            
            if m in ['train', 'val', 'test', 'new']:
                sampletag += split[m]
            else:
                raise ValueError

        self.load_info = []
        for tagi in sorted(sampletag):
            info = {
                'unique_id' : tagi, 
                'filename'  : tagi+'.jpg', 
                'image_path': osp.join(
                    self.root_dir, 'image', tagi+'.jpg'),
                'seglabel_path': [
                    osp.join(
                        self.root_dir, 'semantic_label', 
                        tagi+'_maskfg.png'
                    ),
                    (lambda x: x if osp.exists(x) else None)(
                        osp.join(
                            self.root_dir, 'instance_label', 
                            tagi+'_mask.png'
                        ),
                    ),
                ],
                'bbox_path' : osp.join(
                    self.root_dir, 'bpoly_label', 
                    tagi+'_anno.json'
                ),
                'bpoly_path' : osp.join(
                    self.root_dir, 'bpoly_label', 
                    tagi+'_anno.json'
                ),
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
from .ds_loader import NumpyImageLoader

@regloader({
    'ignore_label' : 'IGNORE_LABEL', 
    'der_map_to'   : 'LOADER_DERIVED_CLS_MAP_TO', })
class TextSeg_SemlabelLoader(object):
    def __init__(self, ignore_label, der_map_to='ig'):
        self.ig = ignore_label
        if der_map_to == 'ig':
            self.der = ignore_label
        elif der_map_to == 'bg':
            self.der = 0
        elif der_map_to == 'fg':
            self.der = 1
        elif der_map_to == 'sepcls':
            raise NotImplementedError
            # self.der = 2
        else:
            raise ValueError

    @pre_loader_checkings('seglabel')
    def __call__(self, path, element):
        return self.load(path, element)

    def load(self, path, element):
        psem = path[0]
        sem = np.array(PIL.Image.open(psem)).astype(int)
        sem[sem==100] = 1
        sem[sem==200] = self.der
        sem[sem==255] = self.ig
        return sem

@regloader({
    'ignore_label' : 'IGNORE_LABEL', 
    'der_map_to'   : 'LOADER_DERIVED_CLS_MAP_TO', })
class TextSeg_SeglabelLoader(TextSeg_SemlabelLoader):
    """
    Load both semantic and instance mask
    """
    def __init__(self, ignore_label, der_map_to='ig'):
        super().__init__(ignore_label, der_map_to)

    def load(self, path, element):
        sem = super().load(path, element)
        pins = path[1]
        if pins is not None:
            ins = np.array(PIL.Image.open(pins)).astype(int)
        else:
            ins = np.zeros(sem.shape).astype(int)
        ins[ins==255] = self.ig
        seg = np.stack([sem, ins])
        return seg

@regloader({
    'ignore_label' : 'IGNORE_LABEL',
    'square_bbox'  : 'LOADER_SQUARE_BBOX',})
class CharBboxSpLoader(object):
    """
    The special character bbox loader that load
        character bbox only from instance mask. 
        a) Each character bbox will be squared(an option).
        b) if instance mask id is not consecutive, the output
            bbox (and chins) will skip those ids even it exists
            in json.
        c) ignore label instance mask will be ignored.
    """
    def __init__(self, 
                 ignore_label, 
                 square_bbox=True,
                 **kwargs):
        self.map = {}
        self.map.update({chr(48+i) : i    for i in range(10)})
        self.map.update({chr(65+i) : 10+i for i in range(26)})
        self.map.update({chr(97+i) : 10+i for i in range(26)})
        self.ig = ignore_label
        self.square_bbox = square_bbox

    @pre_loader_checkings('bbox')
    def __call__(self, path, element):
        return self.load(path, element)

    def load(self, path, element):
        # load only the cls map from json
        with open(path, 'r') as f:
            jsoninfo = json.load(f)
        lcmap = {}
        for _, wi in jsoninfo.items():
            try:
                lcmap.update({ci['mask_value']:ci['text'] \
                    for _, ci in wi['char'].items()})
            except:
                pass

        lcmap = {
            li:self.map[ci] \
                if ci in self.map else 36 \
                for li, ci in lcmap.items()}

        ins = element['seglabel'][1]
        insoh = nputils.one_hot_2d(ignore_label=self.ig)(ins)
        bbox = nputils.bbox_binary(is_coord=False)(insoh)
        bbox = [
            list(bi)+[lcmap[idi], idi] \
                for idi, bi in enumerate(bbox) if (bi.sum()!=0) and (idi>0)]

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

#############
# transform #
#############

@regtrans(
    {'size'          : 'RANDOM_RESIZE_CROP_SIZE', 
     'scale'         : 'RANDOM_RESIZE_CROP_SCALE',
     'ratio'         : 'RANDOM_RESIZE_CROP_RATIO',
     'ignore_label'  : 'IGNORE_LABEL',
     'crop_from'     : 'RANDOM_RESIZE_CROP_FROM',})
class TextSeg_RandomResizeCropCharBbox(TBase):
    """
    A special transform that produce element 'chins' and 'chcls'
            from the semantic label from seglabel.
    For TextSeg classification
        a) Get the squared bbox coord.
        b) Find the RandomResize coord for each bbox coord.
        c) Crop each of the bbox from semantic label. 
            Pad the crop when outside the image boundary.
    """
    def __init__(self, 
                 size,
                 scale,
                 ratio,
                 ignore_label, 
                 crop_from='sem',
                 **kwargs):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.int_f = nputils.interpolate_2d(
            size, mode='bilinear', align_corners=False,)
        self.ig = ignore_label
        self.crop_from = crop_from
    
    @have(
        must=['image', 'seglabel'],)
    def __call__(self, element):
        if self.crop_from == 'sem':
            seg = copy.deepcopy(element['seglabel'][0])
        else:
            seg = copy.deepcopy(element['seglabel'][1])

        seg[seg==self.ig] = 0 # bg all ig label
        bbox = copy.deepcopy(element['bbox_squared'])
        bbox = bbox.astype(int)

        chins = []
        chcls = []
        for h1, w1, h2, w2, clsi, idi in bbox:
            off = self.rand(
                element['unique_id'], 'cropsize'+str(idi),
                rand_f=self.find_offset, 
                oricrop=[h1, w1, h2, w2], 
                scale=self.scale,
                ratio=self.ratio)
            if self.crop_from == 'sem':
                chinsi = nputils.crop_2d(off, fill=[0])(
                    seg[np.newaxis])[0]
                chinsi = chinsi.astype(np.float64)
            else: 
                chinsi = nputils.crop_2d(off, fill=[0])(
                    (seg==idi)[np.newaxis])[0]
                chinsi = chinsi.astype(np.float64)

            # debug
            # vis.quick_plot([chinsi])

            chinsi = self.int_f(chinsi)
            chins.append(chinsi)
            chcls.append(clsi)
        
        if len(chins) > 0:
            element['chins'] = np.stack(chins)
            element['chcls'] = np.array(chcls).astype(int)
        else:
            element['chins'] = np.zeros([0] + list(self.size), dtype=np.float64)
            element['chcls'] = np.zeros([0], dtype=int)
        return element

    def find_offset(self,
                    oricrop,
                    scale,
                    ratio):
        """
        Args:
            oricrop: [int, int, int, int],
                original crop offset.
        """
        h1, w1, h2, w2 = oricrop
        h = h2-h1
        w = w2-w1
        off = self.find_offset_inner([h, w], scale, ratio)
        off = [off[0]+h1, off[1]+w1, off[2]+h1, off[3]+w1]
        return off

    def find_offset_inner(self,
                          orisize,
                          scale,
                          ratio):
        """
        This function is very similar to RandomResizeCrop
            in the regular transform. The only part that is
            different is it accept to set offset beyond orisize.
        """
        oh, ow = orisize
        target_area = npr.uniform(*scale)*oh*ow
        # the log scale uniform
        logrt = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(npr.uniform(*logrt))
                
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        w = int(round(math.sqrt(target_area * aspect_ratio)))

        if h<=0:
            h = 1
            w = round(target_area)
            w = 1 if w<=0 else w
        elif w<=0:
            w = 1
            h = round(target_area)
            h = 1 if h<=0 else h

        if h <= oh:
            h1 = npr.randint(0, oh-h+1)
        else:
            h1 = npr.randint(oh-h, 1)

        if w <= ow:
            w1 = npr.randint(0, ow-w+1)
        else:
            w1 = npr.randint(ow-w, 1)

        return [h1, w1, h1+h, w1+w]

    def exec(self, data, element):
        return data, element

@regtrans(
    {'size'          : 'RESIZE_CROP_SIZE', 
     'ignore_label'  : 'IGNORE_LABEL',})
class TextSeg_ResizeCropForCharBbox(TBase):
    """
    A special transform that produce element 'chins' and 'chcls'
            from the semantic label from seglabel.
    For TextSeg classification
        a) Get the bbox coord.
        b) Crop semantic label and resize to the target size.
        c) fill 0 if output bound.
    """
    def __init__(self, 
                 size,
                 ignore_label, 
                 **kwargs):
        super().__init__()
        self.size = size
        self.int_f = nputils.interpolate_2d(
            size, mode='bilinear', align_corners=False,)
        self.ig = ignore_label
    
    @have(
        must=['image', 'seglabel'],)
    def __call__(self, element):
        sem = copy.deepcopy(element['seglabel'][0])
        sem[sem==self.ig] = 0 # bg all ig label
        bbox = copy.deepcopy(element['bbox'])
        sem = sem.astype(np.float64)
        bbox = bbox.astype(int)

        chins = []
        chcls = []
        for h1, w1, h2, w2, clsi, idi in bbox:
            off = [h1, w1, h2, w2]
            chinsi = nputils.crop_2d(off, fill=[0])(sem[np.newaxis])[0]
            chinsi = self.int_f(chinsi)
            chins.append(chinsi)
            chcls.append(clsi)
        
        if len(chins) > 0:
            element['chins'] = np.stack(chins)
            element['chcls'] = np.array(chcls).astype(int)
        else:
            element['chins'] = np.zeros([0] + list(self.size), dtype=np.float64)
            element['chcls'] = np.zeros([0], dtype=int)
        return element

    def exec(self, data, element):
        return data, element

@regtrans(
    {'n'            : 'ADD_RANDOM_BBOX_NUM', 
     'target_clsid' : 'ADD_RANDOM_BBOX_TARGET_CLSID',})
class TextSeg_AddRandomBboxCharBbox(TBase):
    """
    A special transform that add some extra bbox 
            from the semantic label from seglabel.
        a) when image has no bbox, the new bbox is randomly cropped from image 
            that do not exceed the image size.
        b) when image has bbox, the new bbox size is uniformly sampled from the largest
            and smallest bbox, that aspect ratio is the same. Offset is random.
    """
    def __init__(self, 
                 n,
                 target_clsid,
                 **kwargs):
        """
        Args:
            n: int, 
                number of the new bbox.
            target_clsid: int,
                label of the new bbox.
        """
        super().__init__()
        self.n = n
        self.target_clsid = target_clsid
    
    @have(
        must=['image', 'bbox'],)
    def __call__(self, element):
        return element

    def find_offset(self,
                    n,
                    imsize,
                    area_range,
                    ratio_range):
        h, w = imsize
        bbox_new = []
        while len(bbox_new) < n:
            if area_range[0] == area_range[1]:
                target_area = area_range[0]
            else:
                target_area = npr.uniform(*area_range)
 
            if ratio_range[0] == ratio_range[1]:
                target_ratio = ratio_range[0]
            else:
                target_ratio = npr.uniform(*ratio_range)

            nh = int(round(math.sqrt(target_area / target_ratio)))
            nw = int(round(math.sqrt(target_area * target_ratio)))

            if (nh > h) or (nw > w):
                continue

            h1 = npr.randint(0, h-nh+1)
            w1 = npr.randint(0, h-nh+1)
            bbox_new.append([h1, w1, h1+nh, w1+nw, self.target_clsid, -1])
        return bbox_new

    def exec_image(self, data, element):
        return data, element

    def exec_bbox(self, data, element):
        imarea = element['image'].shape[-2] * element['image'].shape[-1]
        if len(data) == 0:
            area_range = (imarea/256, imarea/16)
            ratio_range = (0.5, 1)
        else:
            dh = data[:, 2] - data[:, 0]
            dw = data[:, 3] - data[:, 1]
            area_range = (np.min(dh*dw), np.max(dh*dw))
            ratio_range = (np.min(dh/dw), np.max(dh/dw))

        newbbx = self.rand(
            element['unique_id'], 'newbbx',
            rand_f=self.find_offset, 
            n = self.n,
            imsize = element['image'].shape[-2:],
            area_range = area_range,
            ratio_range = ratio_range)

        # square the bbox for element['bbox_sqaured']
        newbbx_squared = []
        for h1, w1, h2, w2, clsi, idi in newbbx:
            d = (h2-h1) - (w2-w1)
            if d>0:
                w1 -= d//2
                w2 += d-d//2
            else:
                h1 -= -(d//2)
                h2 += -(d-d//2)
            newbbx_squared.append([h1, w1, h2, w2, clsi, idi])

        # update both
        newbbx = np.array(newbbx).astype(float)
        data = np.concatenate([data, newbbx], axis=0)

        newbbx_squared = np.array(newbbx_squared).astype(int)
        element['bbox_squared'] = np.concatenate(
            [element['bbox_squared'], newbbx_squared], axis=0)
        return data, element

##############
# formatters #
##############

@regformat()
class SemanticFormatter(object):
    def __init__(self,
                 **kwargs):
        pass

    def __call__(self, element):
        im = element['image']
        seglabel = element['seglabel']
        if len(seglabel.shape) > 2:
            seglabel = seglabel[0]
        return im.astype(np.float32), seglabel.astype(int), element['unique_id']

@regformat()
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
        chins = element['chins'].astype(np.float32)
        chcls = element['chcls'].astype(int)
        return im, sem, [bbox], [chins], [chcls], uid

@regformat()
class SemChinsChbbxFormatter2(SemanticFormatter):
    """
    This formatter output semantic/instance/bbox
        bbox will be filtered out if 
            a) part of it is outside image 
                (caused by random crop or square)
    """
    def __call__(self, element):
        im, sem, uid = super().__call__(element)
        h, w = im.shape[-2:]
        bbox = element['bbox']
        h1, w1, h2, w2, _, _ = bbox.T
        valid = (h1>=0) & (w1>=0) & (h2<=h) & (w2<=w)
        bbox = bbox[valid].astype(np.float32)
        chins = element['chins'].astype(np.float32)
        chcls = element['chcls'].astype(int)
        return im, sem, [bbox], [chins], [chcls], uid

@regformat()
class ChbbxClsFormatter(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, element):
        chins = element['chins'].astype(np.float32)
        chcls = element['chcls'].astype(int)
        return [chins], [chcls], element['unique_id']
