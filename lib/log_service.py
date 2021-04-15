import timeit
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
from .cfg_helper import cfg_unique_holder as cfguh

def print_log(console_info):
    print(console_info)
    log_file = cfguh().cfg.LOG_FILE
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(console_info + '\n')

class log_manager(object):
    """
    The helper to print logs. 
    """
    def __init__(self,
                 **kwargs):
        self.data = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()

    def accumulate(self, 
                   n, 
                   data,
                   **kwargs):
        """
        Args:
            n: number of items (i.e. the batchsize)
            data: {itemname : float} data (i.e. the loss values)
                which are going to be accumulated. 
        """
        if n < 0:
            raise ValueError

        for itemn, di in data.items():
            try:
                self.data[itemn] += di * n
            except:
                self.data[itemn] = di * n
            
            try:
                self.cnt[itemn] += n
            except:
                self.cnt[itemn] = n

    def print(self, rank, itern, epochn, samplen, lr):
        console_info = [
            'Rank:{}'.format(rank),
            'Iter:{}'.format(itern),
            'Epoch:{}'.format(epochn),
            'Sample:{}'.format(samplen),
            'LR:{:.4f}'.format(lr)]

        cntgroups = {}
        for itemn, ci in self.cnt.items():
            try:
                cntgroups[ci].append(itemn)
            except:
                cntgroups[ci] = [itemn]

        for ci, itemng in cntgroups.items():
            console_info.append('cnt:{}'.format(ci)) 
            for itemn in sorted(itemng):
                console_info.append('{}:{:.4f}'.format(
                    itemn, self.data[itemn]/ci))

        console_info.append('Time:{:.2f}s'.format(
            timeit.default_timer() - self.time_check))
        return ' , '.join(console_info)

    def clear(self):
        self.data = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()

    def pop(self, rank, itern, epochn, samplen, lr):
        console_info = self.print(
            rank, itern, epochn, samplen, lr)
        self.clear()
        return console_info

# ----- also include some small utils -----

def torch_to_numpy(*argv):
    if len(argv) > 1:
        data = list(argv)
    else:
        data = argv[0]

    if isinstance(data, torch.Tensor):
        return data.to('cpu').detach().numpy()

    elif isinstance(data, (list, tuple)):
        out = []
        for di in data:
            out.append(torch_to_numpy(di))
        return out

    elif isinstance(data, dict):
        out = {}
        for ni, di in data.items():
            out[ni] = torch_to_numpy(di)
        return out
    
    else:
        return data
