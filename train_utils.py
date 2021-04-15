import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
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
import PIL
import copy
from easydict import EasyDict as edict

from lib import nputils
from lib import torchutils
from lib import loss as myloss

from configs.cfg_dataset import \
    cfg_textseg, cfg_cocots, cfg_mlt, cfg_icdar13, cfg_totaltext
from configs.cfg_model import cfg_texrnet as cfg_mdel

from lib.cfg_helper import \
    cfg_unique_holder as cfguh, \
    get_experiment_id, set_debug_cfg, \
    experiment_folder, hided_sig_to_str, \
    common_argparse, common_initiates

from lib.data_factory import \
    get_dataset, collate, \
    get_loader, get_transform, \
    get_formatter, DistributedSampler

from lib.model_zoo import \
    get_model, save_state_dict

from lib.optimizer import \
    get_optimizer, adjust_lr, lr_scheduler

from lib.log_service import print_log, torch_to_numpy, log_manager

cfguh().add_code(osp.basename(__file__))

class exec_container(object):
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
        self.RANK = RANK
        cfg = self.cfg
        cfguh().save_cfg(cfg) 
        dist.init_process_group(
            backend = cfg.DIST_BACKEND,
            init_method = cfg.DIST_URL,
            rank = RANK,
            world_size = cfg.GPU_COUNT,
        )

        # need to set random seed again
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
            if 'TRAIN' in cfg:
                self.save(**para)
        print_log(
            'Total {:.2f} seconds'.format(timeit.default_timer() - time_start))
        self.RANK = None
        dist.destroy_process_group()

    def prepare_dataloader(self):
        return {'dataloader' : None}

    def prepare_model(self):
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

        if istrain:
            net.train() 
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
        transforms = get_transform()()
        formatter = get_formatter()()

        trainset = dataset(
            mode = cfg.DATA.DATASET_MODE, 
            loader = loader, 
            estimator = None, 
            transforms = transforms, 
            formatter = formatter,
        )
        sampler = DistributedSampler(
            dataset=trainset)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU, 
            sampler = sampler, 
            num_workers = cfg.DATA.NUM_WORKERS_PER_GPU, 
            drop_last = False, pin_memory = False,
            collate_fn = collate(), 
        )
        return {
            'dataloader' : trainloader,
            'sampler'    : sampler}

##########
# config #
##########

def set_cfg(cfg, dsname):
    cfg.CUDA = True
    cfg.RND_SEED = 2

    cfg.DATA.DATASET_MODE = '>>>>later<<<<'
    cfg.DATA.LOADER_PIPELINE = [
        'NumpyImageLoader', 
        'NumpySeglabelLoader']
    cfg.DATA.ALIGN_CORNERS = True
    cfg.DATA.IGNORE_LABEL = cfg.DATA.SEGLABEL_IGNORE_LABEL

    cfg.DATA.RANDOM_SCALE_ONESIDE_DIM = 'shortside'
    cfg.DATA.RANDOM_SCALE_ONESIDE_RANGE = [513, 1025]
    cfg.DATA.RANDOM_SCALE_ONESIDE_ALIGN_CORNERS = \
        cfg.DATA.ALIGN_CORNERS
    cfg.DATA.RANDOM_CROP_SIZE = (513, 513)
    cfg.DATA.RANDOM_CROP_PADDING_MODE = 'random'
    cfg.DATA.RANDOM_CROP_FILL = {
        'image'    : [0, 0, 0], 
        'seglabel' : [cfg.DATA.IGNORE_LABEL]}
    cfg.DATA.TRANS_PIPELINE = [
        'UniformNumpyType',
        'NormalizeUint8ToZeroOne', 
        'Normalize',
        'RandomScaleOneSide',
        'RandomCrop', 
    ]

    cfg.DATA.FORMATTER = 'SemanticFormatter'
    cfg.DATA.EFFECTIVE_CLASS_NUM = cfg.DATA.CLASS_NUM

    if dsname == 'textseg':
        cfg.DATA.LOADER_DERIVED_CLS_MAP_TO = 'bg'
        cfg.DATA.LOADER_PIPELINE = [
            'NumpyImageLoader', 
            'TextSeg_SeglabelLoader']
    elif dsname == 'cocots':
        pass
    elif dsname == 'mlt':
        cfg.DATA.LOADER_PIPELINE = [
            'NumpyImageLoader', 
            'Mlt_SeglabelLoader']
    elif dsname == 'icdar13':
        pass
    elif dsname == 'totaltext':
        cfg.DATA.LOADER_PIPELINE = [
            'NumpyImageLoader', 
            'TotalText_SeglabelLoader']
    elif dsname == 'textssc':
        pass
    else:
        raise ValueError

    ##########
    # resnet #
    ##########
    cfg.MODEL.RESNET.MODEL_TAGS = ['base', 'dilated', 'resnet101', 'os16']
    cfg.MODEL.RESNET.PRETRAINED_PTH = osp.abspath(osp.join(
        osp.dirname(__file__), 'pretrained', 'init', 
        'resnet101_imagenet.pth.base'))
    cfg.MODEL.RESNET.CONV_TYPE = 'conv'
    cfg.MODEL.RESNET.BN_TYPE = ['bn', 'syncbn'][0] # 'inplace_abn'
    cfg.MODEL.RESNET.RELU_TYPE = 'relu' # 'lrelu|0.01'
    cfg.MODEL.RESNET.USE_MAXPOOL = True

    ###########
    # deeplab #
    ###########
    cfg.MODEL.DEEPLAB.MODEL_TAGS = ['resnet', 'v3+', 'os16', 'base']
    cfg.MODEL.DEEPLAB.PRETRAINED_PTH = None
    cfg.MODEL.DEEPLAB.OUTPUT_CHANNEL_NUM = 256
    cfg.MODEL.DEEPLAB.CONV_TYPE = cfg.MODEL.RESNET.CONV_TYPE
    cfg.MODEL.DEEPLAB.BN_TYPE = cfg.MODEL.RESNET.BN_TYPE
    cfg.MODEL.DEEPLAB.RELU_TYPE = cfg.MODEL.RESNET.RELU_TYPE
    cfg.MODEL.DEEPLAB.ASPP_WITH_GAP = True

    cfg.MODEL.DEEPLAB.FREEZE_BACKBONE_BN = False
    cfg.MODEL.DEEPLAB.INTERPOLATE_ALIGN_CORNERS = \
        cfg.DATA.ALIGN_CORNERS

    ###########
    # texrnet #
    ###########
    cfg.MODEL.TEXRNET.MODEL_TAGS = ['deeplab']
    cfg.MODEL.TEXRNET.PRETRAINED_PTH = None
    cfg.MODEL.TEXRNET.INPUT_CHANNEL_NUM = \
        cfg.MODEL.DEEPLAB.OUTPUT_CHANNEL_NUM
    cfg.MODEL.TEXRNET.SEMANTIC_CLASS_NUM = \
        cfg.DATA.EFFECTIVE_CLASS_NUM
    cfg.MODEL.TEXRNET.REFINEMENT_CHANNEL_NUM = [
        3+cfg.MODEL.DEEPLAB.OUTPUT_CHANNEL_NUM
        +cfg.DATA.EFFECTIVE_CLASS_NUM, 64, 64]

    cfg.MODEL.TEXRNET.CONV_TYPE = cfg.MODEL.RESNET.CONV_TYPE
    cfg.MODEL.TEXRNET.BN_TYPE = cfg.MODEL.RESNET.BN_TYPE
    cfg.MODEL.TEXRNET.RELU_TYPE = cfg.MODEL.RESNET.RELU_TYPE

    cfg.MODEL.TEXRNET.ALIGN_CORNERS = cfg.DATA.ALIGN_CORNERS    
    cfg.MODEL.TEXRNET.SEMANTIC_IGNORE_LABEL = \
        cfg.DATA.IGNORE_LABEL
    cfg.MODEL.TEXRNET.SEMANTIC_LOSS_TYPE = 'ce'
    cfg.MODEL.TEXRNET.REFINEMENT_LOSS_TYPE = {
        'lossrfn'    : 'ce',
        'lossrfntri' : 'trimapce',
    }
    cfg.MODEL.TEXRNET.INIT_BIAS_ATTENTION_WITH = None
    cfg.MODEL.TEXRNET.BIAS_ATTENTION_TYPE = 'cossim'
    # cfg.MODEL.TEXRNET.INTRAIN_GETPRED_FROM = 'sem'
    cfg.MODEL.TEXRNET.INTRAIN_GETPRED_FROM = None
    # cfg.MODEL.TEXRNET.INEVAL_OUTPUT_ARGMAX = False

    ###########
    # general #
    ###########
    cfg.DATA.NUM_WORKERS_PER_GPU = 4

    cfg.TRAIN.BATCH_SIZE_PER_GPU = 8
    cfg.TRAIN.MAX_STEP = 20500
    cfg.TRAIN.MAX_STEP_TYPE = 'iter'

    cfg.TRAIN.LR_ITER_BY = cfg.TRAIN.MAX_STEP_TYPE
    cfg.TRAIN.LR_BASE = 0.01
    cfg.TRAIN.LR_TYPE = [
        ('linear', 0, cfg.TRAIN.LR_BASE,  500),
        ('ploy', cfg.TRAIN.LR_BASE, 0, cfg.TRAIN.MAX_STEP-500, 0.9)
    ]
    cfg.TRAIN.ACTIVATE_REFINEMENT_AT_ITER = 0

    cfg.TRAIN.OPTIMIZER = 'sgd'
    cfg.TRAIN.SGD_MOMENTUM = 0.9
    cfg.TRAIN.SGD_WEIGHT_DECAY = 5e-4
    cfg.TRAIN.USE_OPTIM_MANAGER = True
    cfg.TRAIN.OPTIM_MANAGER_LRSCALE = {
        'resnet':1, 'deeplab':10, 'texrnet': 10}

    cfg.TRAIN.OVERFIT_A_BATCH = False
    cfg.TRAIN.LOSS_WEIGHT = {
        'losssem'   : 1, 
        'lossrfn'   : 0.5,
        'lossrfntri': 0.5,
    }
    cfg.TRAIN.LOSS_WEIGHT_NORMALIZED = False

    cfg.TRAIN.CKPT_EVERY = np.inf
    cfg.TRAIN.DISPLAY = 10
    cfg.TRAIN.VISUAL = False
    cfg.TRAIN.SAVE_INIT_MODEL = True
    cfg.TRAIN.COMMENT = '>>>>later<<<<'

    if cfg.DATA.DATASET_NAME not in [dsname]:
        raise ValueError
    if cfg.MODEL.MODEL_NAME not in ['texrnet']:
        raise ValueError
    return cfg

def set_cfg_hrnetw48(cfg):
    try:
        cfg.MODEL.pop('DEEPLAB')
    except:
        pass
    try:
        cfg.MODEL.pop('RESNET')
    except:
        pass

    cfg.MODEL.HRNET = edict()
    cfg.MODEL.HRNET.MODEL_TAGS = ['v0', 'base']
    cfg.MODEL.HRNET.PRETRAINED_PTH = osp.abspath(osp.join(
        osp.dirname(__file__), 'pretrained', 'init', 
        'hrnetv2_w48_imagenet_pretrained.pth.base'))

    cfg.MODEL.HRNET.STAGE1_PARA = {
        'NUM_MODULES'  : 1,
        'NUM_BRANCHES' : 1,
        'BLOCK'        : 'BOTTLENECK',
        'NUM_BLOCKS'   : [4],
        'NUM_CHANNELS' : [64],
        'FUSE_METHOD'  : 'SUM',}
    cfg.MODEL.HRNET.STAGE2_PARA = {
        'NUM_MODULES'  : 1,
        'NUM_BRANCHES' : 2,
        'BLOCK'        : 'BASIC',
        'NUM_BLOCKS'   : [4, 4],
        'NUM_CHANNELS' : [48, 96],
        'FUSE_METHOD'  : 'SUM',}
    cfg.MODEL.HRNET.STAGE3_PARA = {
        'NUM_MODULES'  : 4,
        'NUM_BRANCHES' : 3,
        'BLOCK'        : 'BASIC',
        'NUM_BLOCKS'   : [4, 4, 4],
        'NUM_CHANNELS' : [48, 96, 192],
        'FUSE_METHOD'  : 'SUM',}
    cfg.MODEL.HRNET.STAGE4_PARA = {
        'NUM_MODULES'  : 3,
        'NUM_BRANCHES' : 4,
        'BLOCK'        : 'BASIC',
        'NUM_BLOCKS'   : [4, 4, 4, 4],
        'NUM_CHANNELS' : [48, 96, 192, 384],
        'FUSE_METHOD'  : 'SUM',}
    cfg.MODEL.HRNET.FINAL_CONV_KERNEL = 1

    cfg.MODEL.HRNET.OUTPUT_CHANNEL_NUM = sum([48, 96, 192, 384])
    cfg.MODEL.HRNET.ALIGN_CORNERS = \
        cfg.DATA.ALIGN_CORNERS
    cfg.MODEL.HRNET.IGNORE_LABEL = \
        cfg.DATA.IGNORE_LABEL
    cfg.MODEL.HRNET.BN_MOMENTUM = 'hardcoded to 0.1'
    cfg.MODEL.HRNET.LOSS_TYPE = 'ce'
    cfg.MODEL.HRNET.INTRAIN_GETPRED = False

    ###########
    # texrnet #
    ###########
    cfg.MODEL.TEXRNET.MODEL_TAGS = ['hrnet']
    cfg.MODEL.TEXRNET.INPUT_CHANNEL_NUM = \
        cfg.MODEL.HRNET.OUTPUT_CHANNEL_NUM
    cfg.MODEL.TEXRNET.REFINEMENT_CHANNEL_NUM = [
        3+cfg.MODEL.HRNET.OUTPUT_CHANNEL_NUM
        +cfg.DATA.EFFECTIVE_CLASS_NUM, 64, 64]
    cfg.MODEL.TEXRNET.CONV_TYPE = 'conv'
    cfg.MODEL.TEXRNET.BN_TYPE = 'bn'
    cfg.MODEL.TEXRNET.RELU_TYPE = 'relu'

    ###########
    # general #
    ###########
    if not cfg.DEBUG:
        cfg.DATA.NUM_WORKERS_PER_GPU = 5
        cfg.TRAIN.BATCH_SIZE_PER_GPU = 5
    cfg.TRAIN.OPTIM_MANAGER_LRSCALE = {
        'hrnet':1, 'texrnet': 10}
    return cfg

###############
# train stage #
###############

class ts(object):
    def __init__(self):
        self.lossf = None

    def main(self,
             batch,
             net,
             lr,
             optimizer,
             opmgr,
             RANK,
             isinit,
             itern,
             **kwargs):
        cfg = cfguh().cfg
        im, gtsem, _ = batch

        try:
            if itern == cfg.TRAIN.ACTIVATE_REFINEMENT_AT_ITER:
                try:
                    net.module.activate_refinement()
                except:
                    net.activate_refinement()
        except:
            pass

        if cfg.CUDA:
            im = im.to(RANK)
            gtsem = gtsem.to(RANK)

        adjust_lr(optimizer, lr, opmgr=opmgr)
        optimizer.zero_grad()
        loss_item = net(im, gtsem)

        if self.lossf is None:
            self.lossf = myloss.finalize_loss(
                weight=cfg.TRAIN.LOSS_WEIGHT, 
                normalize_weight=cfg.TRAIN.LOSS_WEIGHT_NORMALIZED)
        loss, loss_item = self.lossf(loss_item)

        loss.backward()
        if isinit:
            optimizer.zero_grad()
        else:
            optimizer.step()

        return {'item': loss_item}

    def __call__(self,
                 **paras):
        cfg = cfguh().cfg
        logm = log_manager()
        epochn, itern = 0, 0

        dataloader = paras['dataloader']
        compute_lr = paras['compute_lr']
        RANK       = paras['RANK']

        while cfg.MAINLOOP_EXECUTE:
            for idx, batch in enumerate(dataloader):
                if not isinstance(batch[0], list):
                    batch_n = batch[0].shape[0]
                else:
                    batch_n = len(batch[0])

                if cfg.TRAIN.SKIP_PARTIAL \
                        and (batch_n != cfg.TRAIN.BATCH_SIZE_PER_GPU):
                    continue

                if cfg.TRAIN.LR_ITER_BY == 'epoch':
                    lr = compute_lr(epochn)
                elif cfg.TRAIN.LR_ITER_BY == 'iter':
                    lr = compute_lr(itern)
                else:
                    raise ValueError

                if itern==0:
                    self.main(
                        batch=batch,
                        lr=lr, 
                        isinit=True,
                        itern=itern,
                        **paras)

                paras_new = self.main(
                    batch=batch, 
                    lr=lr,
                    isinit=False,
                    itern=itern,
                    **paras)

                paras.update(paras_new)

                logm.accumulate(batch_n, paras['item'])
                itern += 1

                if itern % cfg.TRAIN.DISPLAY == 0:
                    print_log(logm.pop(
                        RANK, itern, epochn, (idx+1)*cfg.TRAIN.BATCH_SIZE, lr))

                if not isinstance(cfg.TRAIN.VISUAL, bool):
                    if itern % cfg.TRAIN.VISUAL == 0:
                        self.visual_f(paras['plot_item'])

                if cfg.TRAIN.MAX_STEP_TYPE == 'iter':
                    if itern >= cfg.TRAIN.MAX_STEP:
                        break
                    if itern % cfg.TRAIN.CKPT_EVERY == 0:
                        if RANK == 0:
                            print_log('Checkpoint... {}'.format(itern))
                            self.save(itern=itern, epochn=None, **paras)

                # loop end

            epochn += 1

            if cfg.TRAIN.MAX_STEP_TYPE == 'iter':
                if itern >= cfg.TRAIN.MAX_STEP:
                    break

            elif cfg.TRAIN.MAX_STEP_TYPE == 'epoch':
                if epochn >= cfg.TRAIN.MAX_STEP:
                    break
                if epochn % cfg.TRAIN.CKPT_EVERY == 0:
                    if RANK == 0:
                        print_log('Checkpoint... {}'.format(epochn))
                        self.save(itern=None, epochn=epochn, **paras)

    def visual_f(self, item):
        raise ValueError

    def save(self, itern, epochn, **paras):
        cfg = cfguh().cfg
        net = paras['net']
        if itern is not None:
            save_state_dict(
                net, 
                osp.join(
                    cfg.LOG_DIR,
                    '{}_iter_{}.pth'.format(cfg.EXPERIMENT_ID, itern)))
        elif epochn is not None:
            save_state_dict(
                net, 
                osp.join(
                    cfg.LOG_DIR,
                    '{}_epoch_{}.pth'.format(cfg.EXPERIMENT_ID, epochn)))
        else:
            save_state_dict(
                net, 
                osp.join(
                    cfg.LOG_DIR,
                    '{}.pth'.format(cfg.EXPERIMENT_ID)))

class ts_with_classifier_base(ts):
    def __init__(self):
        super().__init__()
        self.clsnet = None
        self.clsoptim = None
        # debug
        self.map = {}
        self.map.update({i:chr(48+i) for i in range(10)})
        self.map.update({10+i:chr(97+i) for i in range(26)})
        self.map.update({36:'#'})

    def get_classifier(self, RANK):
        from easydict import EasyDict as edict
        from lib.model_zoo.get_model import get_model
        from lib.optimizer.get_optimizer import get_optimizer
        cfg = cfguh().cfg
        cfgm = edict()
        cfgm.RESNET = edict()
        cfgm.RESNET.MODEL_TAGS = ['resnet50']
        cfgm.RESNET.PRETRAINED_PTH = cfg.TRAIN.CLASSIFIER_PATH
        cfgm.RESNET.INPUT_CHANNEL_NUM = 1
        cfgm.RESNET.CONV_TYPE = 'conv'
        cfgm.RESNET.BN_TYPE = 'bn'
        cfgm.RESNET.RELU_TYPE = 'relu'
        cfgm.RESNET.CLASS_NUM = 37
        cfgm.RESNET.IGNORE_LABEL = cfg.DATA.IGNORE_LABEL
        net = get_model()('resnet', cfgm)
        if cfg.CUDA:
            net.to(RANK)
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[RANK], 
                find_unused_parameters=True)        
        net.train()
        if not cfg.TRAIN.UPDATE_CLASSIFIER:
            from lib.model_zoo.utils import eval_bn
            # deactivate the running mean and var 
            net = eval_bn(net)

        optimizer = get_optimizer(net, opmgr=None)
        return net, optimizer

    def main(self,
             batch,
             net,
             lr,
             optimizer,
             opmgr,
             RANK,
             itern,
             isinit = False,
             **kwargs):
        cfg = cfguh().cfg
        roi_size = cfg.TRAIN.ROI_ALIGN_SIZE
        update_cls = cfg.TRAIN.UPDATE_CLASSIFIER
        act_after = cfg.TRAIN.ACTIVATE_CLASSIFIER_FOR_SEGMODEL_AFTER

        im, sem, bbx, chins, chcls, _ = batch
        # add batch index at front in bbx
        bbx = [
            torch.cat([torch.ones(ci.shape[0], 1).float()*idx, ci], dim=1) \
                for idx, ci in enumerate(bbx)]
        bbx = torch.cat(bbx, dim=0)

        if cfg.CUDA:
            im = im.to(RANK)
            sem = sem.to(RANK)
            bbx = bbx.to(RANK)
        zero = torch.zeros([], dtype=torch.float32, device=im.device)

        if self.clsnet is None:
            self.clsnet, self.clsoptim = self.get_classifier(RANK)

        if self.lossf is None:
            self.lossf = myloss.finalize_loss(
                weight=cfg.TRAIN.LOSS_WEIGHT, 
                normalize_weight=cfg.TRAIN.LOSS_WEIGHT_NORMALIZED)

        adjust_lr(optimizer, lr, opmgr=opmgr)
        optimizer.zero_grad()

        if update_cls:
            adjust_lr(self.clsoptim, lr, opmgr=None)
        self.clsoptim.zero_grad()

        loss_item = net(im, sem)
        pred = loss_item.pop('pred')

        h, w = pred.shape[-2:]
        osh, osw = im.shape[-2]/h, im.shape[-1]/w
        bbx[:, 1] /= osh
        bbx[:, 3] /= osh
        bbx[:, 2] /= osw
        bbx[:, 4] /= osw

        if cfg.TRAIN.ROI_BBOX_PADDING_TYPE == 'semcrop':
            # the bbox have already been squared. 
            # no further action is needed.
            bbx_reordered = torch.stack(
                [bbx[:, i] for i in [0, 2, 1, 4, 3]], dim=-1)
            # input bbx is <bs, w1, h1, w2, h2>
            # pred[:, 1:2] means we only get the fg part
            chpred = torchutils.roi_align(roi_size)(
                pred[:, 1:2], bbx_reordered)
        elif cfg.TRAIN.ROI_BBOX_PADDING_TYPE == 'inscrop':
            # the bbox haven't been squared yet. 
            # square the box before roi_align and pad out of box value to zero.
            dh, dw = [bbx[:, i]-bbx[:, i-2] for i in (3, 4)]
            bbx_sq = bbx.clone()
            bbx_sq[dw>dh , 1] -=  (dw-dh)[dw>dh]/2 # modify h1
            bbx_sq[dw>dh , 3] +=  (dw-dh)[dw>dh]/2 # modify h2
            bbx_sq[dw<=dh, 2] -= (dh-dw)[dw<=dh]/2 # modify w1
            bbx_sq[dw<=dh, 4] += (dh-dw)[dw<=dh]/2 # modify w2

            dhw = torch.max(dh, dw)
            bbx_offset = bbx[:, 1:5] - bbx_sq[:, 1:5]
            bbx_offset[:, 0] *= roi_size[0]/dhw
            bbx_offset[:, 2] *= roi_size[0]/dhw
            bbx_offset[:, 2] += roi_size[0]
            bbx_offset[:, 1] *= roi_size[1]/dhw
            bbx_offset[:, 3] *= roi_size[1]/dhw
            bbx_offset[:, 3] += roi_size[1]
            bbx_offset[:, 0:2] = torch.floor(bbx_offset[:, 0:2])
            bbx_offset[:, 2:4] = torch.ceil( bbx_offset[:, 2:4])
            bbx_offset = bbx_offset.long()

            bbx_reordered = torch.stack(
                [bbx_sq[:, i] for i in [0, 2, 1, 4, 3]], dim=-1)
            
            chpred = torchutils.roi_align(roi_size)(
                pred[:, 1:2], bbx_reordered)

            chpred_zeropad = torch.zeros(
                chpred.shape, device=chpred.device, 
                dtype=chpred.dtype)
            for idxi in range(chpred.shape[0]):
                h1, w1, h2, w2 = bbx_offset[idxi]
                chpred_zeropad[idxi, :, h1:h2, w1:w2] = chpred[idxi, :, h1:h2, w1:w2]
            chpred = chpred_zeropad
        else:
            raise ValueError

        chpredcls = bbx[:, 5].long()

        # compute the extra loss including the result from clsnet
        # do not update clsnet weight however. 

        loss_item['losscls'] = zero
        if update_cls:
            loss_item['lossupdatecls'] = zero

        if (chpred.shape[0] > 1) & (itern >= act_after):
            lossclsp_item = self.clsnet(chpred, chpredcls)
            loss_item['losscls'] = lossclsp_item['losscls']
        else:
            # we have to put a dummy forward and backward
            # with loss * zero otherwise it will stuck in 
            # multiprocess run.
            chpred_dummy = torch.zeros(
                [2, 1]+list(roi_size), dtype=torch.float32, 
                device=im.device)
            chpredcls_dummy = torch.zeros(
                [2], dtype=torch.int64,
                device=im.device)
            lossclsp_item_dummy = self.clsnet(chpred_dummy, chpredcls_dummy)
            loss_item['losscls'] = lossclsp_item_dummy['losscls'] * 0

        loss, loss_display = self.lossf(loss_item)
        loss.backward()
        if isinit:
            optimizer.zero_grad()
        else:
            optimizer.step()
        self.clsoptim.zero_grad()

        # update clsnet weight using the gt chins
        # do not update net
        if update_cls:
            chins = torch.cat(chins, dim=0)
            chcls = torch.cat(chcls, dim=0)
            loss_item = {ni:zero for ni in loss_item.keys()}

            if (chins.shape[0] > 1):
                if cfg.CUDA:
                    chins = chins.to(RANK)
                    chcls = chcls.to(RANK)
                cls_item = self.clsnet(chins.unsqueeze(1), chcls)
                loss_item['lossupdatecls'] = cls_item['losscls']
                # debug
                # print(cls_item['accnum'])

            loss, loss_display2 = self.lossf(loss_item)
            loss.backward()
            if isinit:
                self.clsoptim.zero_grad()
            else:
                self.clsoptim.step()
            optimizer.zero_grad()
            loss_display['lossupdatecls'] = loss_display2['lossupdatecls']

        return {'item': loss_display}

    def __call__(self, **para):
        rv = super().__call__(**para)
        cfg = cfguh().cfg
        if cfg.TRAIN.UPDATE_CLASSIFIER:
            output_model_file = osp.join(
                cfg.LOG_DIR,
                '{}_resnet50_clsnet.pth'.format(cfg.EXPERIMENT_ID))
            print_log('Saving model file {0}'.format(output_model_file))
            save_state_dict(self.clsnet, output_model_file)
        return rv

class ts_with_classifier(ts_with_classifier_base):
    def main(self, **para):
        cfg = cfguh().cfg
        try:
            if para['itern'] == cfg.TRAIN.ACTIVATE_REFINEMENT_AT_ITER:
                try:
                    para['net'].module.activate_refinement()
                except:
                    para['net'].activate_refinement()
        except:
            pass
        return super().main(**para)
