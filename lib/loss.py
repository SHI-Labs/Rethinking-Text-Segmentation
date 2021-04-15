import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from . import torchutils

class finalize_loss(object):
    def __init__(self, 
                 weight = None,
                 normalize_weight = True,
                 **kwargs):
        if weight is None:
            self.weight = None
        else:
            for _, wi in weight.items():
                if wi < 0:
                    raise ValueError

            if not normalize_weight:
                self.weight = weight
            else:
                sum_weight = 0
                for _, wi in weight.items():
                    sum_weight += wi
                if sum_weight == 0:
                    raise ValueError           
                self.weight = {
                    itemn:wi/sum_weight for itemn, wi in weight.items()}

        self.normalize_weight = normalize_weight

    def __call__(self, 
                 loss_input,):
        item = {n : v.item() for n, v in loss_input.items()}
        lossname = [n for n in loss_input.keys() if n[0:4]=='loss']

        if self.weight is not None:
            if sorted(lossname) \
                    != sorted(list(self.weight.keys())):
                raise ValueError

        loss_num = len(lossname)
        loss = None

        for n in lossname:
            v = loss_input[n]
            if loss is not None:
                if self.weight is not None:
                    loss += v * self.weight[n]
                else:
                    loss += v
            else:
                if self.weight is not None:
                    loss = v * self.weight[n]
                else:
                    loss = v

        if (self.weight is None) and (self.normalize_weight):
            loss /= loss_num

        item['Loss'] = loss.item()
        return loss, item
