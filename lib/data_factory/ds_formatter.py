import os
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
import cv2
# import scipy.ndimage
from PIL import Image
import copy
import gc
import itertools

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
class get_formatter(object):
    def __init__(self):
        self.formatter = {}

    def register(self, formatf, kwmap, kwfix):
        self.formatter[formatf.__name__] = [formatf, kwmap, kwfix]

    def __call__(self, format_name=None):
        cfgd = cfguh().cfg.DATA
        if format_name is None:
            format_name = cfgd.FORMATTER

        formatf, kwmap, kwfix = self.formatter[format_name]
        kw = {k1:cfgd[k2] for k1, k2 in kwmap.items()}
        kw.update(kwfix)
        return formatf(**kw)

def register(kwmap={}, kwfix={}):
    def wrapper(class_):
        get_formatter().register(class_, kwmap, kwfix)
        return class_
    return wrapper

@register()
class SemanticFormatter(object):
    def __init__(self, 
                 **kwargs):
        pass

    def __call__(self, element):
        im = element['image']
        semlabel = element['seglabel']
        return im.astype(np.float32), semlabel.astype(int), element['unique_id']
