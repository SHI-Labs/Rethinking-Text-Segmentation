import torch.distributed as dist
import torch.multiprocessing as mp

import os
import os.path as osp
import sys
import numpy as np
import copy
import gc
import time

import argparse
from easydict import EasyDict as edict

from lib.model_zoo.texrnet import version as VERSION
from lib.cfg_helper import cfg_unique_holder as cfguh, \
    get_experiment_id, \
    experiment_folder, \
    common_initiates

from configs.cfg_dataset import cfg_textseg, cfg_cocots, cfg_mlt, cfg_icdar13, cfg_totaltext
from configs.cfg_model import cfg_texrnet as cfg_mdel
from configs.cfg_base import cfg_train, cfg_test

from train_utils import \
    set_cfg as set_cfg_train, \
    set_cfg_hrnetw48 as set_cfg_hrnetw48_train, \
    ts, ts_with_classifier, train

from eval_utils import \
    set_cfg as set_cfg_eval, \
    set_cfg_hrnetw48 as set_cfg_hrnetw48_eval, \
    es, eval

cfguh().add_code(osp.basename(__file__))

def common_argparse():

    
    cfg = edict()
    cfg.DEBUG = args.debug
    cfg.DIST_URL = 'tcp://127.0.0.1:{}'.format(args.port)
    is_eval = args.eval
    pth = args.pth
    return cfg, is_eval, pth

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug' , action='store_true', default=False)
    parser.add_argument('--hrnet' , action='store_true', default=False)
    parser.add_argument('--eval'  , action='store_true', default=False)
    parser.add_argument('--pth'   , type=str)
    parser.add_argument('--gpu'   , nargs='+', type=int)
    parser.add_argument('--port'  , type=int, default=11233)
    parser.add_argument('--dsname', type=str, default='textseg')
    parser.add_argument('--trainwithcls', action='store_true', default=False)
    args = parser.parse_args()

    istrain = not args.eval

    if istrain:
        cfg = copy.deepcopy(cfg_train)
    else:
        cfg = copy.deepcopy(cfg_test)

    if istrain:
        cfg.EXPERIMENT_ID = get_experiment_id()
    else:
        cfg.EXPERIMENT_ID = None

    if args.dsname == "textseg":
        cfg_data = cfg_textseg
    elif args.dsname == "cocots":
        cfg_data = cfg_cocots
    elif args.dsname == "mlt":
        cfg_data = cfg_mlt
    elif args.dsname == "icdar13":
        cfg_data = cfg_icdar13
    elif args.dsname == "totaltext":
        cfg_data = cfg_totaltext
    else:
        raise ValueError

    cfg.DEBUG = args.debug
    cfg.DIST_URL = 'tcp://127.0.0.1:{}'.format(args.port)
    if args.gpu is None:
        cfg.GPU_DEVICE = 'all'
    else:
        cfg.GPU_DEVICE = args.gpu

    cfg.MODEL = copy.deepcopy(cfg_mdel)
    cfg.DATA  = copy.deepcopy(cfg_data)
    
    if istrain:
        cfg = set_cfg_train(cfg, dsname=args.dsname)
        if args.hrnet:
            cfg = set_cfg_hrnetw48_train(cfg)
    else:
        cfg = set_cfg_eval(cfg, dsname=args.dsname)
        if args.hrnet:
            cfg = set_cfg_hrnetw48_eval(cfg)
        cfg.MODEL.TEXRNET.PRETRAINED_PTH = args.pth

    if istrain:
        if args.dsname == "textseg":
            cfg.DATA.DATASET_MODE = 'train+val'
        elif args.dsname == "cocots":
            cfg.DATA.DATASET_MODE = 'train'
        elif args.dsname == "mlt":
            cfg.DATA.DATASET_MODE = 'trainseg'
        elif args.dsname == "icdar13":
            cfg.DATA.DATASET_MODE = 'train_fst'
        elif args.dsname == "totaltext":
            cfg.DATA.DATASET_MODE = 'train'
        else:
            raise ValueError
    else:
        if args.dsname == "textseg":
            cfg.DATA.DATASET_MODE = 'test'
        elif args.dsname == "cocots":
            cfg.DATA.DATASET_MODE = 'val'
        elif args.dsname == "mlt":
            cfg.DATA.DATASET_MODE = 'valseg'
        elif args.dsname == "icdar13":
            cfg.DATA.DATASET_MODE = 'test_fst'
        elif args.dsname == "totaltext":
            cfg.DATA.DATASET_MODE = 'test'
        else:
            raise ValueError

    if istrain:
        if args.trainwithcls:
            if args.dsname == 'textseg':
                cfg.DATA.LOADER_PIPELINE = [
                    'NumpyImageLoader', 
                    'TextSeg_SeglabelLoader',
                    'CharBboxSpLoader',]
                cfg.DATA.RANDOM_RESIZE_CROP_SIZE = [32, 32]
                cfg.DATA.RANDOM_RESIZE_CROP_SCALE = [0.8, 1.2]
                cfg.DATA.RANDOM_RESIZE_CROP_RATIO = [3/4, 4/3]
                cfg.DATA.TRANS_PIPELINE = [
                    'UniformNumpyType',
                    'TextSeg_RandomResizeCropCharBbox',
                    'NormalizeUint8ToZeroOne', 
                    'Normalize',
                    'RandomScaleOneSide',
                    'RandomCrop', 
                ]
            elif args.dsname == 'icdar13':
                cfg.DATA.LOADER_PIPELINE = [
                    'NumpyImageLoader', 
                    'SeglabelLoader',
                    'CharBboxSpLoader',]
                cfg.DATA.TRANS_PIPELINE = [
                    'UniformNumpyType',
                    'NormalizeUint8ToZeroOne', 
                    'Normalize',
                    'RandomScaleOneSide',
                    'RandomCrop', 
                ]
            else:
                raise ValueError
            cfg.DATA.FORMATTER = 'SemChinsChbbxFormatter'
            cfg.DATA.LOADER_SQUARE_BBOX = True
            cfg.DATA.RANDOM_RESIZE_CROP_FROM = 'sem'
            cfg.MODEL.TEXRNET.INTRAIN_GETPRED_FROM = 'sem'
            # the one with 93.98% and trained on semantic crops 
            cfg.TRAIN.CLASSIFIER_PATH = osp.join(
                'pretrained', 'init', 'resnet50_textcls.pth',
            )
            cfg.TRAIN.ROI_BBOX_PADDING_TYPE = 'semcrop'
            cfg.TRAIN.ROI_ALIGN_SIZE = [32, 32]
            cfg.TRAIN.UPDATE_CLASSIFIER = False
            cfg.TRAIN.ACTIVATE_CLASSIFIER_FOR_SEGMODEL_AFTER = 0
            cfg.TRAIN.LOSS_WEIGHT = {
                'losssem'   : 1, 
                'lossrfn'   : 0.5,
                'lossrfntri': 0.5,
                'losscls'   : 0.1,
            }

    if istrain:
        if args.hrnet:
            cfg.TRAIN.SIGNATURE = ['texrnet', 'hrnet']
        else:
            cfg.TRAIN.SIGNATURE = ['texrnet', 'deeplab']
        cfg.LOG_DIR = experiment_folder(cfg, isnew=True, sig=cfg.TRAIN.SIGNATURE)
        cfg.LOG_FILE = osp.join(cfg.LOG_DIR, 'train.log')
    else:
        cfg.LOG_DIR = osp.join(cfg.MISC_DIR, 'eval')
        cfg.LOG_FILE = osp.join(cfg.LOG_DIR, 'eval.log')
        cfg.TEST.SUB_DIR = None

    if cfg.DEBUG:
        cfg.EXPERIMENT_ID = 999999999999
        cfg.DATA.NUM_WORKERS_PER_GPU = 0
        cfg.TRAIN.BATCH_SIZE_PER_GPU = 2

    cfg = common_initiates(cfg)

    if istrain:
        if args.trainwithcls:
            exec_ts = ts_with_classifier()
        else:
            exec_ts = ts()
        trainer = train(cfg)
        trainer.register_stage(exec_ts)

        # trainer(0)
        mp.spawn(trainer,
                    args=(),
                    nprocs=cfg.GPU_COUNT,
                    join=True)
    else:
        exec_es = es()
        tester = eval(cfg)
        tester.register_stage(exec_es)

        # tester(0)
        mp.spawn(tester,
                args=(),
                nprocs=cfg.GPU_COUNT,
                join=True)
