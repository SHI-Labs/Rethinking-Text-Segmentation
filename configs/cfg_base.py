import os
import os.path as osp
import numpy as np
import copy
import socket

from easydict import EasyDict as edict

cfg = edict()

# -----------------------------BASE-----------------------------

cfg.DEBUG = False
cfg.EXPERIMENT_ID = 0
cfg.GPU_DEVICE = 'all'
cfg.CUDA = False
cfg.MISC_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'log'))
cfg.LOG_FILE = None
cfg.RND_SEED = None
cfg.RND_RECORDING = False
# cfg.USE_FLOAT16 = False
cfg.MATPLOTLIB_MODE = 'Agg'
cfg.MAINLOOP_EXECUTE = True
cfg.MAIN_CODE_PATH = None
cfg.MAIN_CODE = []
cfg.SAVE_CODE = True
cfg.COMPUTER_NAME = socket.gethostname()
cfg.TORCH_VERSION = 'unknown'

cfg.DIST_URL = 'tcp://127.0.0.1:11233'
cfg.DIST_BACKEND = 'nccl'

cfg_train = copy.deepcopy(cfg)
cfg_test = copy.deepcopy(cfg)

# -----------------------------TRAIN-----------------------------

cfg_train.TRAIN = edict()
cfg_train.TRAIN.BATCH_SIZE = None
cfg_train.TRAIN.BATCH_SIZE_PER_GPU = None
cfg_train.TRAIN.MAX_STEP = 0
cfg_train.TRAIN.MAX_STEP_TYPE = None
cfg_train.TRAIN.SKIP_PARTIAL = True
# cfg_train.TRAIN.LR_ADJUST_MODE = None
cfg_train.TRAIN.LR_ITER_BY = None
cfg_train.TRAIN.OPTIMIZER = None
cfg_train.TRAIN.DISPLAY = 0
cfg_train.TRAIN.VISUAL = None
cfg_train.TRAIN.SAVE_INIT_MODEL = True
cfg_train.TRAIN.SAVE_CODE = True

# -----------------------------TEST-----------------------------

cfg_test.TEST = edict()
cfg_test.TEST.BATCH_SIZE = None
cfg_test.TEST.BATCH_SIZE_PER_GPU = None
cfg_test.TEST.VISUAL = None

# -----------------------------COMBINED-----------------------------

cfg.update(cfg_train)
cfg.update(cfg_test)
