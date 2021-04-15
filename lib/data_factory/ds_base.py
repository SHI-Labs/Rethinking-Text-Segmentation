import os
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
import torchvision
import copy
import itertools

import sys
code_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.append(code_dir)

from .. import nputils
from ..cfg_helper import cfg_unique_holder as cfguh
from ..log_service import print_log

class ds_base(torch.utils.data.Dataset):
    def __init__(self, 
                 mode, 
                 loader = None, 
                 estimator = None, 
                 transforms = None, 
                 formatter = None, 
                 **kwargs):
        self.init_load_info(mode)
        self.loader = loader
        self.transforms = transforms
        self.formatter = formatter

        console_info = '{}: '.format(self.__class__.__name__)
        console_info += 'total {} unique images, '.format(len(self.load_info))

        self.load_info = sorted(self.load_info, key=lambda x:x['unique_id'])
        if estimator is not None:
            self.load_info = estimator(self.load_info)

        console_info += 'total {0} unique images after estimation.'.format(
            len(self.load_info))
        print_log(console_info)

        for idx, info in enumerate(self.load_info):
            info['idx'] = idx

        try:
            trysome = cfguh().cfg.DATA.TRY_SOME_SAMPLE
        except:
            trysome = None

        if trysome is not None:
            if isinstance(trysome, str):
                trysome = [trysome]
            elif isinstance(trysome, (list, tuple)):
                trysome = list(trysome)
            else:
                raise ValueError

            self.load_info = [
                infoi for infoi in self.load_info \
                if osp.splitext(
                    osp.basename(infoi['image_path'])
                )[0] in trysome
            ]
            print_log('try {} samples.'.format(len(self.load_info)))

    def init_load_info(self, mode):
        # implement by sub class
        raise ValueError

    def __len__(self):
        try:
            try_sample = cfguh().cfg.DATA.TRY_SAMPLE
        except:
            try_sample = None
        if try_sample is not None:
            return try_sample
        return len(self.load_info)

    def __getitem__(self, idx):
        element = copy.deepcopy(self.load_info[idx])
        element = self.loader(element)
        if self.transforms is not None:
            element = self.transforms(element)
        if self.formatter is not None:
            return self.formatter(element)
        else:
            return element

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_dataset(object):
    def __init__(self):
        self.dataset = {}

    def register(self, dsf):
        self.dataset[dsf.__name__] = dsf

    def __call__(self, dsname=None):
        if dsname is None:
            dsname = cfguh().cfg.DATA.DATASET_NAME

        # the register is in each file
        if dsname == 'textseg':
            from . import ds_textseg
        elif dsname == 'cocots':
            from . import ds_cocotext
        elif dsname == 'mlt':
            from . import ds_mlt
        elif dsname == 'icdar13':
            from . import ds_icdar13
        elif dsname == 'totaltext':
            from . import ds_totaltext
        elif dsname == 'textssc':
            from . import ds_textssc

        return self.dataset[dsname]

def register():
    def wrapper(class_):
        get_dataset().register(class_)
        return class_
    return wrapper

# some other helpers

class collate(object):
    def __init__(self):
        self.default_collate = torch.utils.data._utils.collate.default_collate

    def __call__(self, batch):
        elem = batch[0]
        if not isinstance(elem, (tuple, list)):
            return self.default_collate(batch)
        
        rv = []
        # transposed
        for i in zip(*batch):
            if isinstance(i[0], list):
                if len(i[0]) != 1:
                    raise ValueError
                try:
                    i = [[self.default_collate(ii).squeeze(0)] for ii in i]
                except:
                    pass
                rvi = list(itertools.chain.from_iterable(i))
                rv.append(rvi) # list concat
            elif i[0] is None:
                rv.append(None)
            else:
                rv.append(self.default_collate(i))
        return rv
