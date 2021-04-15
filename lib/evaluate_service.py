import os
import os.path as osp
import numpy as np
import cv2
import copy
import json
from .log_service import print_log
import pprint

import torch
import torch.distributed as dist
from . import torchutils
from . import nputils


def single_run(func):
    def wrapper(self, *args, **kwargs):
        if self.bi is None:
            raise ValueError
            # for k in self.batchc.keys():
            #     rv = func(self.__getitem__(k), *args, **kwargs)
        else:
            rv = func(self, *args, **kwargs)        
        self.bi = None
        self.ri = None
        self.fi = None
        self.sn = None
        return rv
    return wrapper

class distributed_evaluator(object):
    def __init__(self, 
                 name,
                 sample_n,):

        if not dist.is_available():
            raise ValueError
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.session_name = name
        self.batchc = {n:{} for n in name}
        self.record = {n:{} for n in name}
        self.finalr = {n:{} for n in name}
        self.sample_n = {n:0 for n in name}

        if isinstance(sample_n, int):
            self.sample_n = {n:sample_n for n in name}
        elif isinstance(sample_n, dict):
            self.sample_n = sample_n
        else:
            raise ValueError

        self.bi = None
        self.ri = None
        self.fi = None
        self.sn = None

    def __getitem__(self, 
                    name):
        self.bi = self.batchc[name]
        self.ri = self.record[name]
        self.fi = self.finalr[name]
        self.sn = self.sample_n[name]
        return self

    def batchc_exist(self, 
                     itemname):
        if not isinstance(itemname, list):
            itemname = [itemname]
        for iname in itemname:
            if iname not in self.bi.keys():
                return False
        return True

    def record_exist(self,
                     itemname):
        if not isinstance(itemname, list):
            itemname = [itemname]
        for iname in itemname:
            if iname not in self.ri.keys():
                return False
        return True

    def merge(self):
        self.sync_batch()
        for sname in self.record.keys():
            idata = self.record[sname]
            if len(idata)==0:
                for iname, v in self.batchc[sname].items():
                    idata[iname] = [v]
            else:
                for iname, v in self.batchc[sname].items():
                    idata[iname].append(v)

        self.batchc = {n:{} for n in self.session_name}
        self.bi = None

    def sync_single(self,
                    data,
                    from_rank,):
        if from_rank == self.rank:
            # the send data to other gpu
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    # not allowed
                    raise ValueError
                n = torch.tensor(len(data)).long().to(self.rank)
                dist.broadcast(n, src=from_rank)
                for datai in data:
                    self.sync_single(datai, from_rank=from_rank)
                return data

            elif isinstance(data, str):
                n = torch.tensor(len(str)).long()
                d = torch.tensor([ord(i) for i in data]).long()
                sync = [n, d]
            elif isinstance(data, np.ndarray):
                n = torch.tensor(len(data.shape)).long()               
                s = torch.tensor(data.shape).long()
                d = torch.tensor(data)
                sync = [n, s, d]
            else:
                raise ValueError

            for synci in sync:
                synci = synci.to(self.rank)
                dist.broadcast(synci, src=from_rank)
            return data

        else:
            # receive data from other gpu
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    # not allowed
                    raise ValueError
                n = torch.tensor(0).long().to(self.rank)
                dist.broadcast(n, src=from_rank)
                dnew = [
                    self.sync_single(data[0], from_rank=from_rank)\
                        for _ in range(n)]
            elif isinstance(data, str):
                n = torch.tensor(0).long().to(self.rank)
                dist.broadcast(n, src=from_rank)
                dnew = torch.zeros(n).long().to(self.rank)
                dist.broadcast(dnew, src=from_rank)
                dnew = dnew.to('cpu').numpy()
                dnew = str([chr(i) for i in d])
            elif isinstance(data, np.ndarray):
                n = torch.tensor(0).long().to(self.rank)
                dist.broadcast(n, src=from_rank)
                s = torch.zeros(n).long().to(self.rank)
                dist.broadcast(s, src=from_rank)
                dref = torch.tensor(data)
                dnew = torch.zeros(tuple(s), dtype=dref.dtype).to(self.rank)
                dist.broadcast(dnew, src=from_rank)
                dnew = dnew.to('cpu').numpy().astype(data.dtype)
            else:
                raise ValueError
            return dnew

    def sync_batch(self):
        for sni in sorted(self.batchc.keys()):
            for ini in sorted(self.batchc[sni].keys()):
                bwdata = self.batchc[sni][ini]
                bwdata_sync = [
                    self.sync_single(bwdata, from_rank=i)
                    for i in range(self.world_size)
                ]

                # reorder the data into correct order
                # because the recieve order is something like
                # [0, 4, 8], [1, 5, 9], [2, 6, 10], ...
                bwdata = []
                for bwdatai in zip(*bwdata_sync):
                    bwdata += list(bwdatai)

                try:
                    # only work for size-match nparray
                    bwdata = np.stack(bwdata)
                except:
                    pass

                self.batchc[sni][ini] = bwdata

    def retrieve_record(self, itemname):
        """
        Retrieve data from record (self.ri)
        Args:
            itemname: str, or [] of str,
        """
        if not self.record_exist(itemname):
            raise ValueError

        data = []
        for iname in itemname:
            raw = self.ri[iname]

            if not isinstance(raw, list):
                raise ValueError

            if isinstance(raw[0], np.ndarray):
                datai = np.concatenate(raw, axis=0)
            elif isinstance(raw[0], list):
                datai = []
                for ri in raw:
                    datai += ri
            else:
                raise ValueError

            datai = datai[:self.sn]
            data.append(datai)

        return tuple(data)

    @single_run
    def bw_iandu(self, 
                 p, 
                 g, 
                 m=None, 
                 class_n=None):
        """
        Args:
            p: [bs x ...] int nparray
                predicted labels
            g: [bs x ...] int nparray -or- [bs x n x ...] bool nparray
                ground truth labels, can be multiclass.
            m: [bs x ...] bool nparray,
                array tells the data mask, if None, no mask. 
            class_n: int,
        Returns:
            intersection: [bs x n] int nparray,
                intersection over all classes
            union: [bs x n] int nparray, 
                union over all classes
        """
        if not self.batchc_exist(['intersection', 'union']):
            i, u = nputils.iandu_auto(class_n)(p, g, m)
            self.bi['intersection'] = i
            self.bi['union'] = u

        # added on 20201022 for fscore computation
        if not self.batchc_exist(['pred_num', 'gt_num']):
            # a trick to remove prediction from gt ignored region.
            p_modified = copy.deepcopy(p)
            p_modified[g>=class_n] = class_n 
            pn = nputils.label_count(class_n)(p_modified)
            gn = nputils.label_count(class_n)(g)
            self.bi['pred_num'] = pn
            self.bi['gt_num'] = gn

        return self.bi['intersection'], self.bi['union']

    @single_run
    def bw_acc(self, 
               p, 
               g, 
               m=None, 
               class_n=None):
        """
        Args:
            p: [bs x ...] or [bs x c x ...] int nparray,
                predicted label or predicted top k labels
            g: [bs x ...] int nparray,
                groud truth label
            m: [bs x ...] bool nparray or None, 
                the data mask, if None, no mask. 
            class_n: int,
        Returns:
            tot_n: [bs] int nparray,
                tells the total number of check
            acc_n: [bs] int nparray,
                tell the total correct number.
        """
        bs = p.shape[0]
        # if np.prod(bs.shape) == 0: 
        #     # a corner case where np.reshape fails to work
        #     if not self.batchc_exist('acc_n'):
        #         self.bi['acc_n'] = np.zeros([0]).astype(int)
        #     if not self.batchc_exist('tot_n'):
        #         self.bi['tot_n'] = np.zeros([0]).astype(int)
        #     return self.bi['acc_n'], self.bi['tot_n']

        if len(p.shape) != len(g.shape):
            p = [p[:, i].reshape((bs, -1)) for i in range(p.shape[1])]
        else:
            p = [p.reshape((bs, -1))]
        g = g.reshape((bs, -1))

        if m is not None:
            m = m.reshape((bs, -1)) 
        else:
            m = np.ones(g.shape, dtype=np.uint8)
        m &= (g<class_n)

        if not self.batchc_exist('acc_n'):
            gm = g*m
            extra_corr_on_zeros = nputils.batchwise_sum(m==0).astype(int)
            acc = np.zeros(g.shape, dtype=np.uint8)
            # hold the accuracy status on each predictions
            for pi in p:
                pm = pi*m
                acc |= (gm==pm)
            acc_n = nputils.batchwise_sum(acc).astype(int) - extra_corr_on_zeros
            self.bi['acc_n'] = acc_n.flatten()

        if not self.batchc_exist('tot_n'):
            self.bi['tot_n'] = nputils.batchwise_sum(m).astype(int).flatten()

        return self.bi['acc_n'], self.bi['tot_n']

    @single_run
    def miou(self, 
             classname=None,
             find_n_worst=None):
        """
        Args:
            classname: [] or str,
                name of the classes.
            find_n_worst: int -or- None -or- str,
                find the worst n cases.
                None: do not bother that.
                'all': find all worst cases
        """
        i, u = self.retrieve_record(['intersection', 'union']) 

        if find_n_worst is not None:
            umask = u==0
            subi = i.copy().astype(float)
            subu = u.copy().astype(float)
            subu[umask] = 1
            submiou = subi/subu
            submiou[umask] = np.nan
            submiou = np.nanmean(submiou, axis=1)
            worstindex = np.argsort(submiou)
            if find_n_worst!='all':
                worstindex = np.argsort(submiou)[0:find_n_worst]
            worstvalue = {ii:submiou[ii] for ii in worstindex}
        else:
            worstindex = None
            worstvalue = None

        ii = i.sum(axis=0)
        uu = u.sum(axis=0)
        iou = ii.astype(float)/(uu.astype(float))
        miou = np.nanmean(iou)

        if classname is None:
            self.fi['miou'] = {str(i).zfill(3):iou[i] for i in range(len(iou))}
        else:
            self.fi['miou'] = {
                str(i).zfill(3)+'-'+classname[i]:iou[i] for i in range(len(iou))}

        self.fi['miou']['----miou'] = miou

        return {
            'miou' : miou,
            'worstindex' : worstindex,
            'worstvalue' : worstvalue}

    @single_run
    def fscore_complex(self, 
               classname=None,
               find_n_worst=None):
        """
        Args:
            classname: [] or str,
                name of the classes.
            find_n_worst: int -or- None -or- str,
                find the worst n cases.
                None: do not bother that.
                'all': find all worst cases
                # so for not supported
        """
        tp, pn, gn = self.retrieve_record(['intersection', 'pred_num', 'gt_num']) 

        prec_imwise = tp.astype(float)/(pn.astype(float))
        recl_imwise = tp.astype(float)/(gn.astype(float))
        prec_imwise[pn==0] = 0
        recl_imwise[gn==0] = 0

        tp = tp.sum(axis=0)
        pn = pn.sum(axis=0)
        gn = gn.sum(axis=0)

        prec = tp.astype(float)/(pn.astype(float))
        recl = tp.astype(float)/(gn.astype(float))
        fscore = 2*prec*recl/(prec+recl)
        class_num = len(fscore)

        if find_n_worst is not None:
            raise NotImplementedError
        else:
            worstindex = None
            worstvalue = None

        # so far only support classwise count
        if classname is None:
            cname_display = [
                str(i).zfill(3) for i in range(class_num)]
        else:
            cname_display = [
                str(i).zfill(3)+'-'+classname[i] for i in range(class_num)]

        self.fi['precision'] = {
            cname_display[i]:prec[i] for i in range(class_num)}
        self.fi['recall'] = {
            cname_display[i]:recl[i] for i in range(class_num)}
        self.fi['fscore'] = {
            cname_display[i]:fscore[i] for i in range(class_num)}

        prec_imwise = prec_imwise.mean(axis=0)
        recl_imwise = recl_imwise.mean(axis=0)
        fscore_imwise = 2*prec_imwise*recl_imwise/(prec_imwise+recl_imwise)
        self.fi['fscore_imwise'] = {
            cname_display[i]:fscore_imwise[i] for i in range(class_num)}

        return {
            'fscore' : fscore,
            'fscore_imwise' : fscore_imwise,
            'worstindex' : worstindex,
            'worstvalue' : worstvalue}

    @single_run
    def fscore(self, 
               classname=None,
               find_n_worst=None):
        """
        Args:
            classname: [] or str,
                name of the classes.
            find_n_worst: int -or- None -or- str,
                find the worst n cases.
                None: do not bother that.
                'all': find all worst cases
                # so for not supported
        """
        tp, pn, gn = self.retrieve_record(['intersection', 'pred_num', 'gt_num'])

        pn_save = copy.deepcopy(pn); pn_save[pn==0]=1
        gn_save = copy.deepcopy(gn); gn_save[gn==0]=1
        prec_imwise = tp.astype(float)/(pn_save.astype(float))
        recl_imwise = tp.astype(float)/(gn_save.astype(float))
        prec_imwise[pn==0] = 0
        recl_imwise[gn==0] = 0

        prec_imwise = prec_imwise.mean(axis=0)
        recl_imwise = recl_imwise.mean(axis=0)
        fscore_imwise = 2*prec_imwise*recl_imwise/(prec_imwise+recl_imwise)

        class_num = len(fscore_imwise)
        if classname is None:
            cname_display = [
                str(i).zfill(3) for i in range(class_num)]
        else:
            cname_display = [
                str(i).zfill(3)+'-'+classname[i] for i in range(class_num)]

        self.fi['fscore'] = {
            cname_display[i]:fscore_imwise[i] for i in range(class_num)}
        return {
            'fscore' : fscore_imwise}

    @single_run
    def acc(self,
            find_n_worst=None):
        acc_n, tot_n = self.retrieve_record(['acc_n', 'tot_n'])

        if find_n_worst is not None:
            subacc = acc_n/tot_n
            worstindex = np.argsort(subacc)[0:find_n_worst]
            worstvalue = {ii:subacc[ii] for ii in worstindex}
        else:
            worstindex = None
            worstvalue = None

        acc = acc_n.sum()/tot_n.sum()
        self.fi['acc'] = acc

        return {
            'acc' : acc,
            'worstindex' : worstindex,
            'worstvalue' : worstvalue,}

    def summary(self):
        info = pprint.pformat(self.finalr)
        print_log(info)

    def save(self, path, json_info):
        json_info['result'] = self.finalr
        if not osp.exists(osp.dirname(path)):
            os.makedirs(osp.dirname(path))
        with open(path, 'w') as f:
            json.dump(json_info, f, indent=4)
