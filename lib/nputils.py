import os
import os.path as osp
import numpy as np
import numpy.random as npr
import cv2
import copy
import collections

def batch_op(dim): 
    def unsqueeze_dim0(x):
        if x is None:
            return None
        return np.expand_dims(x, axis=0)

    def squeeze_dim0(x):
        if x is None:
            return None
        return x[0]

    def reshape_dimX(x, d, s):
        if x is None:
            return None
        s = copy.deepcopy(s)
        s.extend(list(x.shape[d:]))
        return np.reshape(x, s)

    def wrapper(func):
        def inner(self, x, *argv, **kwargs):
            x = copy.deepcopy(x)
            argv = copy.deepcopy(argv)
            kwargs = copy.deepcopy(kwargs)

            cond  = [
                isinstance(x, np.ndarray)]
            cond += [
                (arg is None) or isinstance(arg, np.ndarray) \
                for arg in argv]
            cond += [
                (arg is None) or isinstance(arg, np.ndarray) \
                for _, arg in kwargs.items()]

            if sum(cond)!=len(cond):
                raise ValueError

            if len(x.shape) < dim:
                raise ValueError

            if len(x.shape) == dim:
                x = unsqueeze_dim0(x)
                argv = [
                    unsqueeze_dim0(arg) \
                    for arg in argv]
                kwargs = {
                    k:unsqueeze_dim0(arg) \
                    for k, arg in kwargs.items()}
                r = func(self, x, *argv, **kwargs)
                if isinstance(r, tuple):
                    rn = [
                        squeeze_dim0(ri) \
                        for ri in r]
                    return tuple(rn)
                else:
                    return squeeze_dim0(r)

            if len(x.shape) == dim+1:
                n = x.shape[0]
                cond  = [
                    (arg is None) or (arg.shape[0]==n) \
                    for arg in argv]
                cond += [
                    (arg is None) or (arg.shape[0]==n) \
                    for _, arg in kwargs.items()]
                if sum(cond)!=len(cond):
                    raise ValueError
                return func(self, x, *argv, **kwargs)

            if len(x.shape) > dim+1:
                d = len(x.shape)-dim
                n = list(x.shape[0:d])
                n_new = [np.prod(n)]

                cond  = [
                    (arg is None) or list(arg.shape[0:d])==n \
                    for arg in argv]
                cond += [
                    (arg is None) or list(arg.shape[0:d])==n \
                    for _, arg in kwargs.items()]
                if sum(cond)!=len(cond):
                    raise ValueError

                x = reshape_dimX(x, d, n_new)
                argv = [
                    reshape_dimX(arg, d, n_new) \
                    for arg in argv]
                kwargs = {
                    k:reshape_dimX(arg, d, n_new) 
                    for k, arg in kwargs.items()}
                
                r = func(self, x, *argv, **kwargs)
                if isinstance(r, tuple):
                    rn = [
                        reshape_dimX(ri, 1, n) \
                        for ri in r]
                    return tuple(rn)
                else:
                    return reshape_dimX(r, 1, n)

        return inner
    return wrapper

def batchwise_mean(x):
    return np.mean(np.reshape(x, (x.shape[0], -1)), axis=1, keepdims=1)

def batchwise_sum(x):
    return np.sum(np.reshape(x, (x.shape[0], -1)), axis=1, keepdims=1)

def deltas_to_offsets(deltas):
    # [0, 0] -> [oh, ow] after padding
    deltas = np.array(deltas)

    temp_a = (-min(deltas[:, 0]), -min(deltas[:, 1]))
    # in case that deltas are all positive
    temp_a = (max(0, temp_a[0]), max(0, temp_a[1]))

    temp_b = (max(deltas[:, 0]), max(deltas[:, 1]))
    temp_b = (max(0, temp_b[0]), max(0, temp_b[1]))

    # ph1, pw1, ph2, pw2
    paddings = list(temp_a) + list(temp_b)
    shifted_deltas = deltas + np.array(temp_a)

    return paddings, shifted_deltas

class one_hot_2d(object):
    def __init__(self, 
                 max_dim=None, 
                 ignore_label=None, 
                 **kwargs):
        self.max_dim = max_dim
        if isinstance(ignore_label, int):
            self.ignore_label = [ignore_label]
        elif ignore_label is None:
            self.ignore_label = []
        else:
            self.ignore_label = ignore_label
        
    @batch_op(2)
    def __call__(self, 
                 x, 
                 mask=None):
        if mask is not None:
            x *= mask==1

        check = []
        for i, n in enumerate(np.bincount(x.flatten())):
            if (i not in self.ignore_label) and (n>0):
                check.append(i)
        
        if self.max_dim is None: 
            max_dim = check[-1]+1
        else:
            max_dim = self.max_dim
        batch_n, h, w = x.shape

        oh = np.zeros((batch_n, max_dim, h, w)).astype(np.uint8)
        for c in check:
            if c >= max_dim:
                continue
            oh[:, c, :, :] = x==c

        if mask is not None:
            # remove the unwanted one-hot zeros
            oh[:, 0, :, :] *= mask==1
        return oh

class nearest_interpolate_2d(object):
    def __init__(self,
                 size,
                 align_corners=True):
        self.size = size
        self.align_corners = align_corners
        self.eps = np.finfo(np.float64).eps

    @batch_op(2)
    def __call__(self, 
                 x):
        _, h, w = x.shape
        if self.align_corners:
            hn = np.linspace(0, h-1, num=self.size[0])
            wn = np.linspace(0, w-1, num=self.size[1])
        else:
            hn = np.linspace(0, h, num=self.size[0]+1)[:-1]
            wn = np.linspace(0, w, num=self.size[1]+1)[:-1]
            hn = hn + h/(self.size[0]*2) - 0.5
            wn = wn + w/(self.size[1]*2) - 0.5

        hnsp = [hn[i]+self.eps*hn[i] for i in range(0, len(hn)//2)] \
             + [hn[i]-self.eps*hn[i] for i in range(len(hn)//2, len(hn))]
        wnsp = [wn[i]+self.eps*wn[i] for i in range(0, len(wn)//2)] \
             + [wn[i]-self.eps*wn[i] for i in range(len(wn)//2, len(wn))]
        hn = np.rint(hnsp).astype(int)
        wn = np.rint(wnsp).astype(int)
        y = x[:, hn, :]
        y = y[:, :, wn]
        return y.copy()

class interpolate_2d(object):
    def __init__(self, 
                 size,
                 mode='nearest',
                 align_corners=True,
                 **kwargs):
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    @batch_op(2)
    def __call__(self, 
                 x):
        if self.mode == 'nearest':
            y = nearest_interpolate_2d(self.size, align_corners=self.align_corners)(x)
        else:
            import torch
            x = torch.as_tensor(x).unsqueeze(1).float()
            y = torch.nn.functional.interpolate(
                x, self.size, mode=self.mode, 
                align_corners=self.align_corners)
            y = y.squeeze(1).numpy()
        return y

class auto_interpolate_2d(object):
    def __init__(self, 
                 size,
                 align_corners=True,
                 **kwargs):
        self.size = size
        self.align_corners = align_corners
        self.int_types = [np.uint8, np.int64]
        self.float_types = [np.float32]

    @batch_op(2)
    def __call__(self, 
                 x):
        dtype = x.dtype
        if dtype in self.int_types:
            y = nearest_interpolate_2d(self.size, align_corners=self.align_corners)(x)
        elif dtype in self.float_types:
            import torch
            x = torch.as_tensor(x).unsqueeze(1).float()
            y = torch.nn.functional.interpolate(
                x, self.size, mode='bilinear', 
                align_corners=self.align_corners)
            y = y.squeeze(1).numpy()
        return y

class iandu_auto(object):
    def __init__(self,
                 max_index):
        self.max_index = max_index

    def __call__(self,
                 x,
                 gt,
                 m=None,
                 **kwargs):
        x_shape = list(x.shape)
        gt_shape = list(gt.shape)

        if x_shape == gt_shape:
            return iandu_normal(self.max_index)(x, gt, m)
        elif x_shape == gt_shape[0:1]+gt_shape[2:]:
            return iandu_iv(self.max_index)(x, gt, m)
        else:
            raise ValueError

class iandu_normal(object):
    def __init__(self, 
                 max_index=None):
        self.max_index = max_index

    def __call__(self,
                 x,
                 gt,
                 m=None,
                 **kwargs):
        if x.dtype not in [np.bool, np.uint8, np.int32, np.int64, int]:
            raise ValueError
        if gt.dtype not in [np.bool, np.uint8, np.int32, np.int64, int]:
            raise ValueError

        x, gt = x.copy().astype(int), gt.copy().astype(int)

        if self.max_index is None:
            max_index = max(np.max(x), np.max(gt))+1
        else:
            max_index = self.max_index
        
        bs = x.shape[0]
        if m is None:
            m = np.ones(x.shape, dtype=np.uint8)

        mgt = m*(gt>=0)&(gt<max_index)
        gt[mgt==0] = max_index 
        # here set it with a new "ignore" index

        mx = mgt*((x>=0)&(x<max_index))
        # all gt ignores will be ignored, but if x predict
        # ignore on non-ignore pixel, that will count as an error.
        x[mx==0] = max_index 

        bdd_index = max_index+1 
        # include the ignore
            
        i, u = [], [] 

        for bi in range(bs):
            cmp = np.bincount((x[bi]+gt[bi]*bdd_index).flatten())
            cm = np.zeros((bdd_index*bdd_index)).astype(int)
            cm[0:len(cmp)] = cmp
            cm = np.reshape(cm, (bdd_index, bdd_index))
            pdn = np.sum(cm, axis=0)
            gtn = np.sum(cm, axis=1)
            tp = np.diag(cm)
            i.append(tp)
            u.append(pdn+gtn-tp)            
        i = np.stack(i)[:, :max_index] # remove ignore
        u = np.stack(u)[:, :max_index]
        return i, u

class iandu_iv(object):
    def __init__(self, 
                 max_index = None):
        self.max_index = max_index

    def __call__(self,
                 x,
                 gt,
                 m=None,
                 **kwargs):
        x, gt = x.copy(), gt.copy()
        
        if self.max_index is not None:
            max_index = self.max_index
        else:
            max_index = max(np.max(x)+1, gt.shape[1])

        bs = x.shape[0]
        if m is None:
            m = np.ones(x.shape, dtype=np.uint8)
            
        if gt.shape[1] > max_index:
            mgt = m*(gt[:, max_index:].sum(axis=1)==0)
        else:
            mgt = m
        
        gt = gt[:, 0:max_index]
        gt *= mgt[:, np.newaxis]
        gt = np.concatenate([gt, (mgt==0)[:, np.newaxis]], axis=1)
        # the last channel is ignore and it is mutually exclusive
        # with other multi class.

        mx = mgt*((x>=0)&(x<max_index))
        x[mx==0] = max_index # here set it with a new index
        # any predictions with its corresponding 
        # gts == ignore, will be forces labeled ignore so they will be
        # excluded from the confusion matrix.

        bdd_index = max_index+1 # include the ignore

        i, u = [], []
        for bi in range(bs):
            ii, uu = self.iandu_iv_single(x[bi], gt[bi], bdd_index)
            i.append(ii)
            u.append(uu)
        i = np.stack(i)[:, :max_index] # remove ignore
        u = np.stack(u)[:, :max_index]
        return i, u

    def iandu_iv_single(self, x, gt, dim_n):
        gt_matrix = gt[np.newaxis] & gt[:, np.newaxis]
        x_new_shape = [-1] + list(x.shape)
        x = one_hot_1d(dim_n)(x.reshape(-1)).reshape(x_new_shape)
        i = x[np.newaxis] & gt_matrix
        i = i.sum(axis=1)
        u = gt + (x & (i==0) & (gt==0)) * gt.sum(0) 
        # this is what different from the future version
        return i.reshape(dim_n, -1).sum(-1), u.reshape(dim_n, -1).sum(-1)

class normalize_2d(object):
    def __init__(self, 
                 mean,
                 std,
                 **kwargs):
        self.mean = np.array(mean)
        self.std = np.array(std)

    @batch_op(3)
    def __call__(self,
                 x):
        y = x-self.mean[:, np.newaxis, np.newaxis]
        y /= self.std[:, np.newaxis, np.newaxis]
        return y

class crop_2d(object):
    def __init__(self, 
                 offset, 
                 fill=None,
                 **kwargs):
        self.offset = offset
        self.fill = np.array(fill)

    def shift(self, x):
        x = np.array(x)
        xmin = x.min()
        return x-xmin

    @batch_op(3)
    def __call__(self, 
                 x):
        oh1, ow1, oh2, ow2 = self.offset
        bn, cn, ih, iw = x.shape
        ih1, iw1, ih2, iw2 = 0, 0, ih, iw
        ih1, ih2, oh1, oh2 = self.shift([ih1, ih2, oh1, oh2])
        iw1, iw2, ow1, ow2 = self.shift([iw1, iw2, ow1, ow2])
        ph = max(ih2, oh2)
        pw = max(iw2, ow2)

        if self.fill is not None:
            y = np.zeros((bn, cn, ph, pw), dtype=x.dtype)
            y[:, :, :, :] = self.fill[:, np.newaxis, np.newaxis]
            y[:, :, ih1:ih2, iw1:iw2] = x
        else:
            y = x

        y = y[:, :, oh1:oh2, ow1:ow2]
        return y

class label_count(object):
    def __init__(self, 
                 max_index=None):
        self.max_index = max_index

    def __call__(self,
                 x,
                 m=None,
                 **kwargs):
        if x.dtype not in [np.bool, np.uint8, 
                           np.int32, np.uint32, 
                           np.int64, np.uint64,
                           int]:
            raise ValueError

        x = x.copy().astype(int)
        if self.max_index is None:
            max_index = np.max(x)+1
        else:
            max_index = self.max_index
        
        if m is None:
            m = np.ones(x.shape, dtype=int)
        else:
            m = (m>0).astype(int)

        mx = m*(x >=0)&(x <max_index)
        # here set it with a new "ignore" index
        x[mx==0] = max_index 

        count = []
        for bi in range(x.shape[0]):
            counti = np.bincount(x[bi].flatten())
            counti = counti[0:max_index]
            counti = list(counti)
            if len(counti) < max_index:
                counti += [0] * (max_index - len(counti))
            count.append(counti)
        count = np.array(count, dtype=int)
        return count

class bbox_binary(object):
    def __init__(self,
                 is_coord=True,
                 **kwargs):
        self.is_coord = is_coord

    @batch_op(2)
    def __call__(self, 
                 x):
        hloc = np.any(x, axis=1)
        wloc = np.any(x, axis=2)

        # +1 to avoid 0 collition
        widx = (np.indices((wloc.shape))[-1].astype(float)+1)*wloc
        h2 = np.max(widx, axis=-1)-1

        hidx = (np.indices((hloc.shape))[-1].astype(float)+1)*hloc
        w2 = np.max(hidx, axis=-1)-1

        widx[widx==0] = np.inf
        h1 = np.min(widx, axis=-1)-1

        hidx[hidx==0] = np.inf
        w1 = np.min(hidx, axis=-1)-1

        h1[h1==np.inf] = 0
        w1[w1==np.inf] = 0

        if not self.is_coord:
            h2 = h2+1
            w2 = w2+1

        bbox = np.stack([h1, w1, h2, w2])
        bbox = np.transpose(bbox, (1, 0))            
        return bbox
