import os
import os.path as osp
import shutil
import copy
import time
from easydict import EasyDict as edict
import pprint
import numpy as np
import torch
import matplotlib
import argparse
import torch
from configs.cfg_base import cfg_train, cfg_test
import json

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class cfg_unique_holder(object):
    def __init__(self):
        self.cfg = None
        # this is use to track the main codes.
        self.code = set()
    def save_cfg(self, cfg):
        self.cfg = copy.deepcopy(cfg)
    def add_code(self, code):
        """
        A new main code is reached and 
            its name is added.
        """
        self.code.add(code)

def get_experiment_id():
    time.sleep(0.5)
    return int(time.time()*100)

def set_debug_cfg(cfg, istrain=True):
    if istrain:
        cfg.EXPERIMENT_ID = 999999999999
        cfg.TRAIN.SAVE_INIT_MODEL = False
        cfg.TRAIN.COMMENT = "Debug"
        cfg.LOG_DIR = osp.join(
            cfg.MISC_DIR, 
            '{}_{}'.format(cfg.MODEL.MODEL_NAME, cfg.DATA.DATASET_NAME),
            str(cfg.EXPERIMENT_ID))
        cfg.LOG_FILE = osp.join(cfg.LOG_DIR, 'train.log')
        cfg.TRAIN.BATCH_SIZE = None
        cfg.TRAIN.BATCH_SIZE_PER_GPU = 2
    else:
        cfg.LOG_DIR = cfg.LOG_DIR.replace(cfg.TEST.SUB_DIR, 'debug')
        cfg.TEST.SUB_DIR = 'debug'
        cfg.LOG_FILE = osp.join(
            cfg.LOG_DIR, 'eval.log')
        cfg.TEST.BATCH_SIZE = None
        cfg.TEST.BATCH_SIZE_PER_GPU = 1

    cfg.DATA.NUM_WORKERS = None
    cfg.DATA.NUM_WORKERS_PER_GPU = 0
    cfg.MATPLOTLIB_MODE = 'TKAgg'
    return cfg

def experiment_folder(cfg,
                      isnew=False,
                      sig=['nosig'],
                      mdds_override=None,
                      **kwargs):
    """
    Args:
        cfg: easydict,
            the config easydict
        isnew: bool,
            whether this is a new folder or not
            True, create a path using exid and sig
            False, find a path based on exid and refpath
        sig: [sig1, ...] array of str
            when isnew == True, these are the signature
            put as [exid]_[sig1]_.._[sign]
            signatures after (and include) 'hided' will be
            hided from the name.
        mdds_override: str or None,
            the override folder for [modelname]_[dataset],
            None, no override 
    Returns:
        workdir: str,
            the absolute path to the folder.
    """
    if mdds_override is None:
        refdir = osp.join(
            cfg.MISC_DIR, 
            '{}_{}'.format(
                cfg.MODEL.MODEL_NAME, cfg.DATA.DATASET_NAME),)    
    else:
        refdir = osp.abspath(osp.join(
            cfg.MISC_DIR, mdds_override))

    if isnew:
        try:
            hided_after = sig.index('hided')
        except:
            hided_after = len(sig)
        sig = sig[0:hided_after]
        workdir = '_'.join([str(cfg.EXPERIMENT_ID)] + sig)
        return osp.join(refdir, workdir)
    else: 
        for d in os.listdir(refdir):
            if not d.find(str(cfg.EXPERIMENT_ID))==0:
                continue
            if not osp.isdir(osp.join(refdir, d)):
                continue
            try:
                workdir = osp.join(
                    refdir, d, cfg.TEST.SUB_DIR)
            except:
                workdir = osp.join(
                    refdir, d)
            return workdir                    
        raise ValueError

def get_experiment_folder(exid, 
                          path,
                          full_path=False,
                          **kwargs):
    """
    Args:
        exid: int, 
            experiment ID
        path: path,
            the base folder to search... 
            folder should be like <exid>_....
        full_path: bool,
            whether return the full path or not.
    """    
    for d in os.listdir(path):
        if d.find(str(exid))==0:
            if osp.isdir(osp.join(path, d)):
                if not full_path:
                    return d
                else:
                    return osp.abspath(osp.join(path, d))
    raise ValueError

def set_experiment_folder(exid, 
                          signature,
                          **kwargs):
    """
    Args:
        exid: experiment ID
        signature: string or array of strings tells the tags that append after exid 
            as a experiment folder...
    """ 
    if isinstance(signature, str):
        signature = [signature]
    return '_'.join([str(exid)] + signature)

def hided_sig_to_str(sig):
    """
    Args:
        sig: [] of str,
    Returns:
        out: str
        If sig is [..., 'hided', 'sig1', 'sig2']
            out = hided: sig1_sig2_...
        If sig do not have 'hided'
            out = None
    """
    try:
        hided_after = sig.index('hided')
    except:
        return None

    return 'hided: '+'_'.join(sig[hided_after+1:]) 

def common_initiates(cfg):
    if cfg.GPU_DEVICE != 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            [str(gid) for gid in cfg.GPU_DEVICE]) 
        cfg.GPU_COUNT = len(cfg.GPU_DEVICE)
    else:
        cfg.GPU_COUNT = torch.cuda.device_count()

    if 'TRAIN' in cfg:
        if (cfg.TRAIN.BATCH_SIZE is None) and \
                (cfg.TRAIN.BATCH_SIZE_PER_GPU is None):
            raise ValueError
        elif (cfg.TRAIN.BATCH_SIZE is not None) and \
                (cfg.TRAIN.BATCH_SIZE_PER_GPU is not None):
            if cfg.TRAIN.BATCH_SIZE != \
                    cfg.TRAIN.BATCH_SIZE_PER_GPU * cfg.GPU_COUNT:
                raise ValueError
        elif cfg.TRAIN.BATCH_SIZE is None:
            cfg.TRAIN.BATCH_SIZE = \
                cfg.TRAIN.BATCH_SIZE_PER_GPU * cfg.GPU_COUNT
        else:
            cfg.TRAIN.BATCH_SIZE_PER_GPU = \
                cfg.TRAIN.BATCH_SIZE // cfg.GPU_COUNT
    if 'TEST' in cfg:
        if (cfg.TEST.BATCH_SIZE is None) and \
                (cfg.TEST.BATCH_SIZE_PER_GPU is None):
            raise ValueError
        elif (cfg.TEST.BATCH_SIZE is not None) and \
                (cfg.TEST.BATCH_SIZE_PER_GPU is not None):
            if cfg.TEST.BATCH_SIZE != \
                    cfg.TEST.BATCH_SIZE_PER_GPU * cfg.GPU_COUNT:
                raise ValueError
        elif cfg.TEST.BATCH_SIZE is None:
            cfg.TEST.BATCH_SIZE = \
                cfg.TEST.BATCH_SIZE_PER_GPU * cfg.GPU_COUNT
        else:
            cfg.TEST.BATCH_SIZE_PER_GPU = \
                cfg.TEST.BATCH_SIZE // cfg.GPU_COUNT

    if (cfg.DATA.NUM_WORKERS is None) and \
            (cfg.DATA.NUM_WORKERS_PER_GPU is None):
        raise ValueError
    elif (cfg.DATA.NUM_WORKERS is not None) and \
            (cfg.DATA.NUM_WORKERS_PER_GPU is not None):
        if cfg.DATA.NUM_WORKERS != \
                cfg.DATA.NUM_WORKERS_PER_GPU * cfg.GPU_COUNT:
            raise ValueError
    elif cfg.DATA.NUM_WORKERS is None:
        cfg.DATA.NUM_WORKERS = \
            cfg.DATA.NUM_WORKERS_PER_GPU * cfg.GPU_COUNT
    else:
        cfg.DATA.NUM_WORKERS_PER_GPU = \
            cfg.DATA.NUM_WORKERS // cfg.GPU_COUNT

    cfg.MAIN_CODE_PATH = osp.abspath(osp.join(
        osp.dirname(__file__), '..'))
    cfg.MAIN_CODE = list(cfg_unique_holder().code)

    cfg.TORCH_VERSION = torch.__version__

    pprint.pprint(cfg)
    if cfg.LOG_FILE is not None:
        if not osp.exists(osp.dirname(cfg.LOG_FILE)):
            os.makedirs(osp.dirname(cfg.LOG_FILE))
        with open(cfg.LOG_FILE, 'w') as f:
            pprint.pprint(cfg, f)
    with open(osp.join(cfg.LOG_DIR, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)

    # step3.1 code saving
    if cfg.SAVE_CODE:
        codedir = osp.join(cfg.LOG_DIR, 'code')
        if osp.exists(codedir):
            shutil.rmtree(codedir)
        for d in ['configs', 'lib']:
            fromcodedir = osp.abspath(
                osp.join(cfg.MAIN_CODE_PATH, d))
            tocodedir = osp.join(codedir, d)
            shutil.copytree(
                fromcodedir, tocodedir, 
                ignore=shutil.ignore_patterns('*__pycache__*', '*build*'))
        for codei in cfg.MAIN_CODE:
            shutil.copy(osp.join(cfg.MAIN_CODE_PATH, codei), codedir)

    # step3.2
    if cfg.RND_SEED is None:
        pass
    elif isinstance(cfg.RND_SEED, int):
        np.random.seed(cfg.RND_SEED)
        torch.manual_seed(cfg.RND_SEED)
    else:
        raise ValueError
        
    # step3.3
    if cfg.RND_RECORDING:
        rnduh().reset(osp.join(cfg.LOG_DIR, 'rcache'), None)
        if not isinstance(cfg.RND_SEED, str):
            pass
        elif osp.isfile(cfg.RND_SEED):
            print('[Warning]: RND_SEED is a file but RND_RECORDING is on and disables the file.')
            # raise ValueError

    # step3.4
    try:
        if cfg.MATPLOTLIB_MODE is not None:
            matplotlib.use(cfg.MATPLOTLIB_MODE)
    except:
        pass

    return cfg

def common_argparse(extra_parsing_f=None):
    """
    Outputs:
        cfg: edict,
            'DEBUG'
            'GPU_DEVICE'
            'ISTRAIN'
        exid: [] of int -or- None
            experiment id followed by --eval 
            None so do the regular training
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', nargs='+', type=int)
    parser.add_argument('--gpu', nargs='+', type=int)
    parser.add_argument('--port', type=int)

    if extra_parsing_f is not None:
        args, cfg = extra_parsing_f(parser)
    else:
        args = parser.parse_args()
        cfg = edict()

    cfg.DEBUG = args.debug
    try:
        cfg.GPU_DEVICE = list(args.gpu)
    except:
        pass

    try:
        eval_exid = list(args.eval)
    except:
        eval_exid = None

    try:
        port = int(args.port)
        cfg.DIST_URL = 'tcp://127.0.0.1:{}'.format(port)
    except:
        pass
    
    return cfg, eval_exid
