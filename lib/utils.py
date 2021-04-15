import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# cudnn.enabled = True
# cudnn.benchmark = True
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import os.path as osp
import sys
import numpy as np
import pprint
import timeit
import time
import copy
import matplotlib
import matplotlib.pyplot as plt

from . import nputils
from . import torchutils
from . import loss as myloss

from .cfg_helper import cfg_unique_holder as cfguh

from .data_factory.ds_base import get_dataset, collate
from .data_factory.ds_loader import get_loader
from .data_factory.ds_transform import get_transform
from .data_factory.ds_formatter import get_formatter
from .data_factory.ds_sampler import DistributedSampler

from .model_zoo.get_model import get_model, save_state_dict
from .optimizer.get_optimizer import get_optimizer, adjust_lr, lr_scheduler

from .log_service import print_log, log_manager

class exec_container(object):
    """
    This is the base functor for all types of executions.
        One execution can have multiple stages, 
        but are only allowed to use the same 
        config, network, dataloader. 
    Thus, in most of the cases, one exec_container is one
        training/evaluation/demo...
    If DPP is in use, this functor should be spawn.
    """
    def __init__(self,
                 cfg,
                 **kwargs):
        self.cfg = cfg
        self.registered_stages = []
        self.RANK = None

    def register_stage(self, stage):
        self.registered_stages.append(stage)

    def __call__(self, 
                 RANK,
                 **kwargs):
        """
        Args:
            RANK: int,
                the rank of the stage process.
            If not multi process, please set 0.
        """
        self.RANK = RANK
        cfg = self.cfg
        cfguh().save_cfg(cfg) 
        # broadcast cfg
        dist.init_process_group(
            backend = cfg.DIST_BACKEND,
            init_method = cfg.DIST_URL,
            rank = RANK,
            world_size = cfg.GPU_COUNT,
        )

        # need to set random seed 
        # originally it is in common_init()
        # but it looks like that the seed doesn't transfered to here.
        if isinstance(cfg.RND_SEED, int):
            np.random.seed(cfg.RND_SEED)
            torch.manual_seed(cfg.RND_SEED)

        time_start = timeit.default_timer()

        para = {
            'RANK':RANK,
            'itern_total':0}
        dl_para = self.prepare_dataloader()
        if not isinstance(dl_para, dict):
            raise ValueError
        para.update(dl_para)
        md_para = self.prepare_model()
        if not isinstance(md_para, dict):
            raise ValueError
        para.update(md_para)

        for stage in self.registered_stages:
            stage_para = stage(**para)
            if stage_para is not None:
                para.update(stage_para)

        # save the model
        if RANK == 0:
            # only train will save the model
            if 'TRAIN' in cfg:
                self.save(**para)

            if cfg.RND_RECORDING:
                raise NotImplementedError
                # rnduh().merge_cache(
                #     osp.join(cfg.LOG_DIR, 'random.p'))

        print_log(
            'Total {:.2f} seconds'.format(timeit.default_timer() - time_start))
        self.RANK = None
        dist.destroy_process_group()

    def prepare_dataloader(self):
        """
        Prepare the dataloader from config.
        """
        return {'dataloader' : None}

    def prepare_model(self):
        """
        Prepare the model from config.
        A default prepare_model for training.
            If the desire behaviour is eval. Or the there
            are two models, or there are something special.
            Please override this function.
        """
        cfg = cfguh().cfg
        net = get_model()()
        paras = {}
        istrain = 'TRAIN' in cfg
        if istrain:
            if 'TEST' in cfg:
                raise ValueError

        # save the init model
        if istrain:
            if (cfg.TRAIN.SAVE_INIT_MODEL) and (self.RANK==0):
                output_model_file = osp.join(
                    cfg.LOG_DIR, '{}_{}.pth.init'.format(
                        cfg.EXPERIMENT_ID, cfg.MODEL.MODEL_NAME))
                save_state_dict(net, output_model_file)

        if cfg.CUDA:
            net.to(self.RANK)
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[self.RANK], 
                find_unused_parameters=True)

        # set the correct mode 
        # assign optimizer and opmgr
        # these things has to be assigned after ddp, 
        # or there will be wierd issues.
        if istrain:
            net.train() 

            # if freeze bbn in effective
            try:
                freeze_bbn = cfg.MODEL.FREEZE_BBN
            except:
                freeze_bbn = False

            if freeze_bbn:
                from myTorchLib.model_zoo.utils import freeze
                try:
                    netref = net.module
                except:
                    netref = net
                freeze(getattr(netref, netref.bbn_name))

            if cfg.TRAIN.USE_OPTIM_MANAGER:
                try:
                    opmgr = net.module.opmgr
                except:
                    opmgr = net.opmgr
                opmgr.set_lrscale(cfg.TRAIN.OPTIM_MANAGER_LRSCALE)
            else:
                opmgr = None

            optimizer = get_optimizer(net, opmgr = opmgr)
            compute_lr = lr_scheduler(cfg.TRAIN.LR_TYPE)
            paras.update({
                'net'       : net,
                'optimizer' : optimizer,
                'compute_lr': compute_lr,
                'opmgr'     : opmgr,
            })
        else:
            net.eval()
            paras.update({'net': net})
        return paras

    def save(self, net, **kwargs):
        cfg = cfguh().cfg
        output_model_file = osp.join(
            cfg.LOG_DIR,
            '{}_{}_last.pth'.format(
                cfg.EXPERIMENT_ID, cfg.MODEL.MODEL_NAME))
        print_log('Saving model file {0}'.format(output_model_file))
        save_state_dict(net, output_model_file)

class train(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        dataset = get_dataset()()
        loader = get_loader()()
        # loader_cache = 'CACHE' in cfg.DATA.TRANS_PIPELINE
        # if not loader_cache:
        #     transforms = get_transform()()
        #     cache_trans = None
        # else:
        #     transp = cfg.DATA.TRANS_PIPELINE
        #     i = transp.index('CACHE')
        #     cache_trans = get_transform()(pipeline=transp[0:i])
        #     transforms = get_transform()(pipeline=transp[i+1:])
        transforms = get_transform()()
        formatter = get_formatter()()

        trainset = dataset(
            mode = cfg.DATA.DATASET_MODE, 
            loader = loader, 
            estimator = None, 
            transforms = transforms, 
            formatter = formatter,
            # loader_cache = loader_cache,
            # loader_cache_transform = cache_trans,
        )
        sampler = DistributedSampler(
            dataset=trainset)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU, 
            sampler = sampler, 
            num_workers = cfg.DATA.NUM_WORKERS_PER_GPU, 
            drop_last = False, pin_memory = False,
            collate_fn = collate(), 
            # 20201006 now defaultly use the self crafted collate
        )
        return {
            'dataloader' : trainloader,
            'sampler'    : sampler}

class eval(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        dataset = get_dataset()()
        loader = get_loader()()
        # loader_cache = 'CACHE' in cfg.DATA.TRANS_PIPELINE
        # if not loader_cache:
        #     transforms = get_transform()()
        #     cache_trans = None
        # else:
        #     transp = cfg.DATA.TRANS_PIPELINE
        #     i = transp.index('CACHE')
        #     cache_trans = get_transform()(pipeline=transp[0:i])
        #     transforms = get_transform()(pipeline=transp[i+1:])
        transforms = get_transform()()
        formatter = get_formatter()()

        evalset = dataset(
            mode = cfg.DATA.DATASET_MODE, 
            loader = loader, 
            estimator = None, 
            transforms = transforms, 
            formatter = formatter,
            # loader_cache = loader_cache,
            # loader_cache_transform = cache_trans,
        )
        sampler = DistributedSampler(
            evalset, shuffle=False, extend=True)
        evalloader = torch.utils.data.DataLoader(
            evalset, batch_size = cfg.TEST.BATCH_SIZE_PER_GPU, 
            sampler = sampler, 
            num_workers = cfg.DATA.NUM_WORKERS_PER_GPU, 
            drop_last = False, pin_memory = False,
            collate_fn = collate(), 
            # 20201006 now defaultly use the self crafted collate
        )
        return {
            'dataloader' : evalloader,}
