import torch
import numpy as np
import numpy.random as npr
import torch.distributed as dist
import math

from ..log_service import print_log

class DistributedSampler(torch.utils.data.Sampler):
    def __init__(self, 
                 dataset, 
                 num_replicas=None, 
                 rank=None, 
                 shuffle=True,
                 extend=False,):
        if num_replicas is None:
            if not dist.is_available():
                raise ValueError
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise ValueError
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        num_samples = len(dataset) // num_replicas
        if extend:
            if len(dataset) != num_samples*num_replicas:
                num_samples+=1

        self.num_samples = num_samples
        self.total_size = num_samples * num_replicas
        self.shuffle = shuffle
        self.extend = extend

    def __iter__(self):
        indices = self.get_sync_order()
        if self.extend:
            # extend using the front indices
            indices = (indices+indices)[0:self.total_size]
        else:
            # truncate
            indices = indices[0:self.total_size]
        # subsample
        indices = indices[self.rank : len(indices) : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        # legacy
        pass

    def get_sync_order(self):
        # g = torch.Generator()
        # g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).to(self.rank)
            dist.broadcast(indices, src=0)
            indices = indices.to('cpu').tolist()
        else:
            indices = list(range(len(self.dataset)))
        print_log(str(indices[0:5]))
        return indices
