import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class optim_manager(object):
    def __init__(self, 
                 group, 
                 order=None,
                 group_lrscale=None,):
        self.group = {}
        self.order = []
        for gn in order:
            self.pushback(gn, group[gn])
        self.group_lrscale = group_lrscale

    def pushback(self, 
                 group_name, 
                 module_name):
        if group_name in self.group.keys():
            raise ValueError
        if isinstance(module_name, (list, tuple)):
            self.group[group_name] = list(module_name)
        else:
            self.group[group_name] = [module_name]
        self.order += [group_name]
        self.group_lrscale = None

    def popback(self):
        group = self.group.pop(self.order[-1])
        self.order = self.order[:-1]
        self.group_lrscale = None
        return group

    def replace(self,
                group_name, 
                module_name,):
        if group_name not in self.group.keys():
            raise ValueError
        if isinstance(module_name, (list, tuple)):
            module_name = list(module_name)
        else:
            module_name = [module_name]
        self.group[group_name] = module_name
        self.group_lrscale = None

    def inheritant(self, 
                   supername):
        for gn in self.group.keys():
            module_name = self.group[gn]
            module_name_new = []
            for mn in module_name:
                if mn == 'self':
                    module_name_new.append(supername)
                else:
                    module_name_new.append(supername+'.'+mn)
            self.group[gn] = module_name_new

    def set_lrscale(self,
                    group_lrscale):
        if sorted(self.order) != \
                sorted(list(group_lrscale.keys())):
            raise ValueError
        self.group_lrscale = group_lrscale

    def pg_generator(self,
                     netref, 
                     module_name):
        if not isinstance(module_name, list):
            raise ValueError
        # the "self" special case
        if (len(module_name)==1) \
                and (module_name[0] == 'self'):
            return netref.parameters()
        pg = []
        for mn in module_name:
            if mn == 'self':
                raise ValueError
            mn = mn.split('.')
            module = netref
            for mni in mn:
                module = getattr(module, mni)
            
            pg.append(module.parameters())

        pg = itertools.chain(*pg)
        return pg
                
    def get_pg(self, net):
        try:
            netref = net.module
        except:
            netref = net

        return [
            {'params': self.pg_generator(netref, self.group[gn])} \
                for gn in self.order]

    def get_pglr(self, idx, base_lr):
        return base_lr * self.group_lrscale[self.order[idx]]
