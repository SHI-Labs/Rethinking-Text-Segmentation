import os.path as osp
import numpy as np
import numpy.random as npr
import PIL
import cv2

import torch
import torchvision
import xml.etree.ElementTree as ET
import json
import copy
import math

from .. import nputils
from .. import torchutils
from ..cfg_helper import cfg_unique_holder as cfguh

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_transform(object):
    def __init__(self):
        self.transform = {}

    def register(self, transf, kwmap, kwfix):
        self.transform[transf.__name__] = [transf, kwmap, kwfix]

    def __call__(self, pipeline=None):
        cfgd = cfguh().cfg.DATA
        if pipeline is None:
            pipeline = cfgd.TRANS_PIPELINE

        trans = []
        for tag in pipeline:
            transf, kwmap, kwfix = self.transform[tag]
            kw = {k1:cfgd[k2] for k1, k2 in kwmap.items()}
            kw.update(kwfix)
            trans.append(transf(**kw))
        if len(trans) == 0:
            return None
        else:
            return compose(trans)

def register(kwmap={}, kwfix={}):
    def wrapper(class_):
        get_transform().register(class_, kwmap, kwfix)
        return class_
    return wrapper

def have(must=[], may=[]):
    def route(self, item, e, d):
        if isinstance(d, np.ndarray):
            dtype = 'nparray'
        elif isinstance(d, torch.Tensor):
            dtype = 'tensor'
        elif isinstance(d, PIL.Image.Image):
            dtype = 'pilimage'
        else:
            raise ValueError

        # find function by order
        f = None
        for attrname in [
                'exec_{}_{}'.format(item, dtype),
                'exec_{}'.format(item),
                'exec_{}'.format(dtype),
                'exec']:
            f = getattr(self, attrname, None)
            if f is not None:
                break
        d, e = f(d, e)
        e[item] = d
        return e

    def wrapper(func):
        def inner(self, e): 
            e['imsize_previous'] = e['imsize_current']            
            imsize_tag_cnt = 0
            imsize_tag = 'imsize_before_' + self.__class__.__name__
            while True:
                if imsize_tag_cnt != 0:
                    tag = imsize_tag + str(imsize_tag_cnt)
                else:
                    tag = imsize_tag
                if not tag in e:
                    e[tag] = e['imsize_current']
                    break
                imsize_tag_cnt += 1
            
            e = func(self, e)
            # must transform list
            for item in must:
                try:
                    d = e[item]
                except:
                    raise ValueError
                if d is None:
                    raise ValueError
                e = route(self, item, e, d)
            # may transform list
            for item in may:
                try:
                    d = e[item]
                except:
                    d = None
                if d is not None:
                    e = route(self, item, e, d)
            return e
        return inner
    return wrapper

class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, element):
        for t in self.transforms:
            element = t(element)
        return element

class TBase(object):
    def __init__(self):
        pass

    def exec(self, data, element):
        raise ValueError

    def rand(self, 
             uid,
             tag, 
             rand_f, 
             *args,
             **kwargs):
        return rand_f(*args, **kwargs)

@register({'mean':'IM_MEAN', 'std':'IM_STD'})
class Normalize(TBase):
    """
    Normalize data using the given mean and std.
    """
    def __init__(self, 
                 mean, 
                 std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    @have(must=['image'])
    def __call__(self, element):
        return element

    def exec_nparray(self, data, element):
        return nputils.normalize_2d(self.mean, self.std)(
            data.astype(np.float32)), element
        
    def exec_tensor(self, data, element):
        return torchvision.transforms.Normalize(
            mean=self.mean, std=self.std)(data), element

@register({'torange':'NORMALIZE_UINT8_TO_RANGE'})
class NormalizeUint8ToRange(TBase):
    """
    Normalize the 0~255 uint8 into the given range
    """
    def __init__(self, 
                 torange=(-1, 1)):
        """
        Args: 
            torange = tuple of two float tells the range to normalized
        """
        super(NormalizeUint8ToRange, self).__init__()
        self.low, self.high = torange
        if self.low > self.high:
            raise ValueError
    
    @have(must=['image'])
    def __call__(self, element):
        return element

    def exec_image_nparray(self, data, element):
        return data.astype(np.float32)*(self.high-self.low)/255 + self.low, element

@register({})
class NormalizeUint8ToZeroOne(TBase):
    """
    Normalize the 0~255 uint8 into 0~1
    """
    def __init__(self):
        super(NormalizeUint8ToZeroOne, self).__init__()
    
    @have(must=['image'])
    def __call__(self, element):
        return element

    def exec_image_nparray(self, data, element):
        return data.astype(np.float32)/255, element

@register(
    {'size'         :'RESCALE_SIZE', 
     'align_corners':'RESCALE_ALIGN_CORNERS'})
class Rescale(TBase):
    def __init__(self, 
                 size,
                 align_corners=True):
        super(Rescale, self).__init__()
        self.size = size
        self.align_corners = align_corners

    @have(
        must=['image'], 
        may=['seglabel', 'mask', 'bpoly', 'bbox'])
    def __call__(self, element):
        element['imsize_current'] = self.size
        return element

    def exec_image_nparray(self, data, element):
        dtype = data.dtype
        data = nputils.interpolate_2d(
            element['imsize_current'], 'bilinear', self.align_corners)(data)
        data = data.astype(dtype)
        return data, element

    def exec_image_pilimage(self, data, element):
        data = data.resize(
            np.array(element['imsize_current'])[::-1], 
            PIL.Image.BILINEAR)
        return data, element
    
    def exec_nparray(self, data, element):
        data = nputils.nearest_interpolate_2d(
            element['imsize_current'], align_corners=self.align_corners)(data)
        return data, element
                
    def exec_bbox_nparray(self, data, element):
        element['bbox_before_Rescale'] = copy.deepcopy(data)
        oh, ow = element['imsize_previous']
        h, w = element['imsize_current']
        extra_cn = data.shape[-1]-4
        return data*np.array([h/oh, w/ow]*2+[1]*extra_cn), element

    def exec_bpoly_nparray(self, data, element):
        element['bpoly_before_Rescale'] = copy.deepcopy(data)
        oh, ow = element['imsize_previous']
        h, w = element['imsize_current']
        return data*np.array([h/oh, w/ow]*4+[1]), element

@register(
    {'range'        :'RANDOM_SCALE_RANGE', 
     'base_size'    :'RANDOM_SCALE_BASE_SIZE',
     'align_corners':'RANDOM_SCALE_ALIGN_CORNERS'})
class RandomScale(Rescale):
    """
    Random scale the data.
    """
    def __init__(self, 
                 range, 
                 base_size=None,
                 align_corners=True):
        super(RandomScale, self).__init__(None, align_corners)
        self.low, self.high = range
        self.base_size = base_size
        self.align_corners = align_corners
        
    @have(
        must=['image'], 
        may=['seglabel', 'mask', 'bpoly', 'bbox'])
    def __call__(self, element):
        scale = self.rand(
            element['unique_id'], 'scale',
            rand_f=npr.uniform, 
            low=self.low, high=self.high)
        if self.base_size is None:
            csize = element['imsize_current']
            element['imsize_current'] = [int(csize[0]*scale), int(csize[1]*scale)]
        else:
            element['imsize_current'] = [
                int(self.base_size[0]*scale), 
                int(self.base_size[1]*scale)]
        return element

@register(
    {'size'         :'RESCALE_ONESIDE_SIZE', 
     'dim'          :'RESCALE_ONESIDE_DIM',
     'align_corners':'RESCALE_ONESIDE_ALIGN_CORNERS'})
class RescaleOneSide(Rescale):
    def __init__(self, 
                 size,
                 dim='shortside', 
                 align_corners=True):
        super(RescaleOneSide, self).__init__(
            None, align_corners)
        self.size = size
        self.dim = dim
        
    @have(
        must=['image'], 
        may =['seglabel', 'mask', 'bpoly', 'bbox'])
    def __call__(self, element):
        ch, cw = element['imsize_current']
        if ((ch<=cw) and (self.dim=='shortside')) \
                or ((ch>cw) and (self.dim=='longside')) \
                or (self.dim=='height'):
            h = self.size
            w = int(h/ch*cw)
        elif ((ch>cw) and (self.dim=='shortside')) \
                or ((ch<=cw) and (self.dim=='longside')) \
                or (self.dim=='width'):
            w = self.size
            h = int(w/cw*ch)
        else:
            raise ValueError
        element['imsize_current'] = [h, w]
        return element

@register(
    {'size_range'   :'RANDOM_SCALE_ONESIDE_RANGE', 
     'dim'          :'RANDOM_SCALE_ONESIDE_DIM',
     'align_corners':'RANDOM_SCALE_ONESIDE_ALIGN_CORNERS'})
class RandomScaleOneSide(Rescale):
    def __init__(self, 
                 size_range,
                 dim='shortside', 
                 align_corners=True):
        super(RandomScaleOneSide, self).__init__(
            size=None)
        self.low, self.high = size_range
        self.dim = dim
        self.align_corners = align_corners
        
    @have(
        must=['image'], 
        may=['seglabel', 'mask', 'bpoly', 'bbox'])
    def __call__(self, element):
        size_oneside = self.rand(
            element['unique_id'], self.dim,
            rand_f=npr.randint, 
            low=self.low, high=self.high+1) 
        # self.high+1 to make it close interval [min, max]
        # self.dim is the string id for the field.
        ch, cw = element['imsize_current']
        if ((ch<=cw) and (self.dim=='shortside')) \
                or ((ch>cw) and (self.dim=='longside')) \
                or (self.dim=='height'):
            h = size_oneside
            w = int(h/ch*cw)
            if w==0:
                # for safety
                w=1
        elif ((ch>cw) and (self.dim=='shortside')) \
                or ((ch<=cw) and (self.dim=='longside')) \
                or (self.dim=='width'):
            w = size_oneside
            h = int(w/cw*ch)
            if h==0:
                h=1
        else:
            raise ValueError
        element['imsize_current'] = [h, w]
        return element

@register(
    {'size':'CENTER_CROP_SIZE', 
     'fill':'CENTER_CROP_FILL',})
class CenterCrop(TBase):
    def __init__(self, 
                 size, 
                 fill):
        super(CenterCrop, self).__init__()
        self.size = size
        self.fill = fill

    @have(
        must=['image'], 
        may =['seglabel', 'mask', 'bpoly', 'bbox'])
    def __call__(self, element):
        return self.find_center_crop_offset(element)

    def find_center_crop_offset(self, element):
        ch, cw = element['imsize_current']
        th, tw = self.size
        oh1 = (ch-th)//2
        ow1 = (cw-tw)//2
        oh2 = oh1 + th
        ow2 = ow1 + tw
        element['Crop_offset'] = (oh1, ow1, oh2, ow2)
        element['imsize_before_CenterCrop'] = \
            element['imsize_current']
        element['imsize_current'] = self.size
        return element

    def exec_helper(self, x, offset, type):
        if len(x.shape) == 2:
            x = x[np.newaxis, :, :].copy()
            shape2d = True
        else:
            shape2d = False        
        y = nputils.crop_2d(
            offset, np.array(self.fill[type]))(x)
        if shape2d:
            y = y[0]
        return y

    def exec_image_nparray(self, data, element):
        data = self.exec_helper(
            data, element['Crop_offset'], 'image')
        return data, element

    def exec_image_pilimage(self, data, element):
        h1, w1, h2, w2 = element['Crop_offset']
        data = data.crop((w1, h1, w2, h2))
        return data, element

    def exec_seglabel_nparray(self, data, element):
        data = self.exec_helper(
            data, element['Crop_offset'], 'seglabel')
        return data, element

    def exec_mask_nparray(self, data, element):
        data = self.exec_helper(
            data, element['Crop_offset'], 'mask')
        return data, element

    def exec_bpoly_nparray(self, data, element):
        h1, w1, _, _ = element['Crop_offset']
        delta = np.array([-h1, -w1]*4 + [0])
        data += delta
        # doesn't check whether the point is inside and outside.
        return data, element

    def exec_bbox_nparray(self, data, element):
        h1, w1, _, _ = element['Crop_offset']
        extra_cn = data.shape[-1] - 4
        delta = np.array([-h1, -w1]*2 + [0]*extra_cn)
        data += delta
        # doesn't check whether the point is inside and outside.
        return data, element

@register(
    {'size'        :'RANDOM_CROP_SIZE', 
     'padding_mode':'RANDOM_CROP_PADDING_MODE',
     'fill'        :'RANDOM_CROP_FILL',})
class RandomCrop(CenterCrop):
    def __init__(self, 
                 size, 
                 padding_mode, 
                 fill):
        super(RandomCrop, self).__init__(size, fill)
        self.padding_mode = padding_mode

    @have(
        must=['image'], 
        may =['seglabel', 'mask', 'bpoly', 'bbox'])
    def __call__(self, element):
        ch, cw = element['imsize_current']
        th, tw = self.size
        if ch >= th:
            oh1 = self.rand(
                element['unique_id'], 'oh1', npr.randint, low=0, high=ch-th+1)
        else:
            if self.padding_mode == 'random':
                oh1 = -self.rand(
                    element['unique_id'], 'oh1', npr.randint, low=0, high=th-ch+1)
            elif self.padding_mode == 'equal':
                oh1 = (ch-th)//2
        
        if cw >= tw:
            ow1 = self.rand(
                element['unique_id'], 'ow1', npr.randint, low=0, high=cw-tw+1)
        else:
            if self.padding_mode == 'random':
                ow1 = -self.rand(
                    element['unique_id'], 'ow1', npr.randint, low=0, high=tw-cw+1)
            elif self.padding_mode == 'equal':
                ow1 = (cw-tw)//2

        element['Crop_offset'] = (oh1, ow1, oh1+th, ow1+tw)
        element['imsize_before_RandomCrop'] = \
            element['imsize_current']
        element['imsize_current'] = self.size
        return element

@register()
class UniformNumpyType(TBase):
    """
    Convert types based on their property
    Also, regularize:
        image -> np.float64
        seglabel -> np.int64
        mask -> np.uint8
        bbox -> np.float64
        bpoly -> np.float64
    """
    def __init__(self):
        super(UniformNumpyType, self).__init__()

    @have(
        must=[], 
        may=['image', 'seglabel', 'mask', 'bbox', 'bpoly'])
    def __call__(self, element):
        return element

    def exec_image_nparray(self, data, element):
        return data.astype(np.float64), element

    def exec_seglabel_nparray(self, data, element):
        return data.astype(np.int64), element

    def exec_mask_nparray(self, data, element):
        return data.astype(np.uint8), element

    def exec_bbox_nparray(self, data, element):
        return data.astype(np.float64), element

    def exec_bpoly_nparray(self, data, element):
        return data.astype(np.float64), element
