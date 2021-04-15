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

from ..cfg_helper import cfg_unique_holder as cfguh
from .. import nputils

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_loader(object):
    def __init__(self):
        self.loader = {}

    def register(self, loadf, kwmap, kwfix):
        self.loader[loadf.__name__] = [loadf, kwmap, kwfix]

    def __call__(self, pipeline=None):
        cfgd = cfguh().cfg.DATA
        if pipeline is None:
            pipeline = cfgd.LOADER_PIPELINE

        loader = []
        for tag in pipeline:
            loadf, kwmap, kwfix = self.loader[tag]
            kw = {k1:cfgd[k2] for k1, k2 in kwmap.items()}
            kw.update(kwfix)
            loader.append(loadf(**kw))
        if len(loader) == 0:
            return None
        else:
            return compose(loader)

class compose(object):
    def __init__(self, loaders):
        self.loaders = loaders

    def __call__(self, element):
        for l in self.loaders:
            element = l(element)
        return element

def register(kwmap={}, kwfix={}):
    def wrapper(class_):
        get_loader().register(class_, kwmap, kwfix)
        return class_
    return wrapper

def pre_loader_checkings(ltype):
    lpath = ltype+'_path'
    # cache feature added on 20201021
    lcache = ltype+'_cache'
    def wrapper(func):
        def inner(self, element):
            if lcache in element:
                # cache feature added on 20201021
                data = element[lcache]
            else:
                if ltype in element:
                    raise ValueError
                if lpath not in element:
                    raise ValueError

                if element[lpath] is None:
                    data = None
                else:
                    data = func(self, element[lpath], element)
            element[ltype] = data

            if ltype == 'image':
                if isinstance(data, np.ndarray):
                    imsize = data.shape[-2:]
                elif isinstance(data, PIL.Image.Image):
                    imsize = data.size[::-1]
                elif data is None:
                    imsize = None
                else:
                    raise ValueError
                element['imsize'] = imsize
                element['imsize_current'] = copy.deepcopy(imsize)
            return element
        return inner
    return wrapper

###########
# general #
###########

@register(
    {'backend':'LOAD_BACKEND_IMAGE',
     'is_mc'  :'LOAD_IS_MC_IMAGE'  ,})
class NumpyImageLoader(object):
    def __init__(self, 
                 backend='pil',
                 is_mc=False,):
        self.backend = backend
        self.is_mc = is_mc

    @pre_loader_checkings('image')
    def __call__(self, path, element):
        return self.load(path)

    def load(self, path):
        if not self.is_mc:
            if self.backend == 'cv2': 
                data = cv2.imread(
                    path, cv2.IMREAD_COLOR)[:, :, ::-1]
            elif self.backend == 'pil':
                data = np.array(PIL.Image.open(
                    path).convert('RGB'))
            else:
                raise ValueError
        else:
            # multichannel should not assume image is
            # defaultly RGB
            datai = []
            for p in path:
                if self.backend == 'cv2':
                    i = cv2.imread(p)[:, :, ::-1]
                elif self.backend == 'pil':
                    i = np.array(PIL.Image.open(p))
                else:
                    raise ValueError
                if len(i.shape) == 2:
                    i = i[:, :, np.newaxis]
                datai.append(i)                
            data = np.concatenate(datai, axis=2)
        return np.transpose(data, (2, 0, 1)).astype(np.uint8)

@register(
    {'backend':'LOAD_BACKEND_IMAGE',
     'is_mc'  :'LOAD_IS_MC_IMAGE'  ,})
class NumpyImageLoaderWithCache(NumpyImageLoader):
    def __init__(self,
                 backend='pil',
                 is_mc=False):
        super().__init__(backend, is_mc)
        self.cache_data = None
        self.cache_path = None

    @pre_loader_checkings('image')
    def __call__(self, path, element):
        if path == self.cache_path:
            return self.cache_data
        else:
            self.cache_data = super().load(path)
        return self.cache_data

@register(
    {'backend':'LOAD_BACKEND_SEGLABEL',
     'is_mc'  :'LOAD_IS_MC_SEGLABEL'  ,})
class NumpySeglabelLoader(object):
    def __init__(self, 
                 backend='pil',
                 is_mc=False):
        self.backend = backend
        self.is_mc = is_mc

    @pre_loader_checkings('seglabel')
    def __call__(self, path, element):
        return self.load(path)

    def load(self, path):
        if not self.is_mc:
            if self.backend == 'cv2': 
                data = cv2.imread(
                    path, cv2.IMREAD_GRAYSCALE)
            elif self.backend == 'pil':
                data = np.array(PIL.Image.open(path))
            else:
                raise ValueError
        else:
            # seglabel doesn't convert to rgb.
            datai = []
            for p in path:
                if self.backend == 'cv2':
                    i = cv2.imread(
                        p, cv2.IMREAD_GRAYSCALE)
                elif self.backend == 'pil':
                    i = np.array(PIL.Image.open(p))
                else:
                    raise ValueError
                if len(i.shape) == 2:
                    i = i[:, :, np.newaxis]
                datai.append(i)                
            data = np.concatenate(datai, axis=2)
            data = np.transpose(data, (2, 0, 1))
            if data.shape[0] == 1:
                data = data[0]                
        return data.astype(np.int32)

@register(
    {'backend':'LOAD_BACKEND_MASK',
     'is_mc'  :'LOAD_IS_MC_MASK'  ,})
class NumpyMaskLoader(NumpySeglabelLoader):
    def __init__(self, 
                 backend='pil',
                 is_mc=False):
        super().__init__(
            backend, is_mc)

    @pre_loader_checkings('mask')
    def __call__(self, path, element):
        data = super().load(path)
        return (data!=0).astype(np.uint8)

