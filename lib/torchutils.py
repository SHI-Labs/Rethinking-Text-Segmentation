import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
import copy

def batch_op(dim): 
    def unsqueeze_dim0(x):
        if x is None:
            return None
        return x.unsqueeze(0)

    def squeeze_dim0(x):
        if x is None:
            return None
        return x.squeeze(0)

    def reshape_dimX(x, d, s):
        if x is None:
            return None
        s = copy.deepcopy(s)
        s.extend(list(x.shape)[d:])
        return x.view(s)

    def wrapper(func):
        def inner(self, x, *argv, **kwargs):
            cond  = [
                isinstance(x, torch.Tensor)]
            cond += [
                (arg is None) or isinstance(arg, torch.Tensor) \
                for arg in argv]
            cond += [
                (arg is None) or isinstance(arg, torch.Tensor) \
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
                    (arg is None) or (arg.size(0)==n) \
                    for arg in argv]
                cond += [
                    (arg is None) or (arg.size(0)==n) \
                    for _, arg in kwargs.items()]
                if sum(cond)!=len(cond):
                    raise ValueError
                return func(self, x, *argv, **kwargs)

            if len(x.shape) > dim+1:
                d = len(x.shape)-dim
                n = list(x.shape)[0:d]
                n_new = [np.prod(n)]

                cond  = [
                    (arg is None) or list(arg.shape)[0:d]==n \
                    for arg in argv]
                cond += [
                    (arg is None) or list(arg.shape)[0:d]==n \
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

def batchwise_sum(x, m=None):
    x = x.view(x.size(0), -1)
    if m is not None:
        m = m.view(x.size(0), -1).float()
        return torch.sum(x*m, dim=1, keepdim=True)
    else:
        return torch.sum(x, dim=1, keepdim=True)

def batchwise_mean(x, m=None):
    x = x.view(x.size(0), -1)
    if m is not None:
        m = m.view(x.size(0), -1).float()
        s = torch.sum(x*m, dim=1, keepdim=True)
        c = torch.sum(m, dim=1, keepdim=True)
        c[c==0] = 1
        return s/c
    else:
        return torch.mean(x, dim=1, keepdim=True)

def batchwise_norm(x):
    s = x.shape
    bs = x.size(0)
    xn = batchwise_sum(x)
    xn[xn==0] = 1 # safe
    x = x.view(bs, -1)/xn.view(bs, 1)
    return x.view(s)

class nearest_interpolate_2d(object):
    """
    A modified version of nearest interpolation
        to avoid a kind of unwanted behavior:
            round(0.5)=0, round(1.5)=2, round(2.5)=2, round(3.5)=4, ...
        Adjusted behavior: suppose width is n
            round(k.5)=k+1 if k<n/2; round(k.5)=k if k>=n/2
    """
    def __init__(self,
                 size,
                 align_corners=True):
        """
        Args:
            size: (int, int), the output size
        """
        self.size = size
        self.align_corners = align_corners
        self.eps = np.finfo(np.float64).eps

    @batch_op(2)
    def __call__(self, 
                 x):
        """
        Args:
            x: [bs x 2D] int array, input array 
        """
        _, h, w = x.shape

        if self.align_corners:
            hn = np.linspace(0, h-1, num=self.size[0])
            wn = np.linspace(0, w-1, num=self.size[1])
        else:
            hn = np.linspace(0, h, num=self.size[0]+1)[:-1]
            wn = np.linspace(0, w, num=self.size[1]+1)[:-1]
            hn = hn + h/(self.size[0]*2) - 0.5
            wn = wn + w/(self.size[1]*2) - 0.5

        # + eps because round(0.5)=0 but round(1.5)=2, I want round(0.5)=1
        hnsp = [hn[i]+self.eps*hn[i] for i in range(0, len(hn)//2)] \
             + [hn[i]-self.eps*hn[i] for i in range(len(hn)//2, len(hn))]
        wnsp = [wn[i]+self.eps*wn[i] for i in range(0, len(wn)//2)] \
             + [wn[i]-self.eps*wn[i] for i in range(len(wn)//2, len(wn))]
        hn = np.rint(hnsp).astype(int)
        wn = np.rint(wnsp).astype(int)
        y = x[:, hn, :]
        y = y[:, :, wn]
        return y.clone()

class interpolate_2d(object):
    """
    This is new interpolate function. 
    Backend of nearest is nearest_interpolate_2d (self-crafted)
    Backend of other interpolation is torch.nn.functional.interpolate
    """

    def __init__(self, 
                 size,
                 mode='nearest',
                 align_corners=True,
                 **kwargs):
        """
        Args:
            size: (int, int), (h, w) of the target size.
            mode: str, mode of the interpolation, choose from:
                'nearest', 'bilinear', etc...
            align_corners: bool, whether aligns the corner.
        """
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
    
    @batch_op(2)
    def __call__(self, 
                 x):
        """
        Args:
            x: [bs x 2D] tensor, 
        Output:
            y: [bs x 2D] tensor, the interpolated output

        """
        if self.mode == 'nearest':
            y = nearest_interpolate_2d(self.size, align_corners=self.align_corners)(x)
        else:
            x = x.unsqueeze(1).float()
            y = F.interpolate(
                x, self.size, mode=self.mode, 
                align_corners=self.align_corners)
            y = y.squeeze(1)
        return y

class auto_interpolate_2d(object):
    """
    This an auto interpolate function that
        'nearest' under torch.int64 and torch.uint8
        'bilinear' under torch.float32
    """
    def __init__(self, 
                 size,
                 align_corners=True,
                 **kwargs):
        """
        Args:
            size: (int, int), (h, w) of the target size.
            align_corners: bool, whether aligns the corner.
        """
        self.size = size
        self.align_corners = align_corners
        self.int_types = [torch.uint8, torch.int64]
        self.float_types = [torch.float32]

    @batch_op(2)
    def __call__(self, 
                 x):
        """
        Args:
            x: [bs x 2D] array, 
        Output:
            y: [bs x 2D] array, the interpolated output
        """
        dtype = x.dtype
        if dtype in self.int_types:
            y = nearest_interpolate_2d(self.size, align_corners=self.align_corners)(x)
        elif dtype in self.float_types:
            x = x.unsqueeze(1).float()
            y = F.interpolate(
                x, self.size, mode='bilinear', 
                align_corners=self.align_corners)
            y = y.squeeze(1)
        return y

class interpolate_2d_lossf(object):
    """
    This function add a intepolation on input x or y before the 
        loss computation. 
    This function uses the auto_interpolate_2d that resize 
        tensor automatically based on its type
    """
    def __init__(self, 
                 lossf,
                 resize_x=False,
                 align_corners=True,
                 **kwargs):
        """
        Args:
            lossf: a torch.nn specified loss function. Reduction has to 
                be "none". Or an element wise-operation on two same size 
                torch.Tensor.
            resize_x: a bool. When True, x is resized to y (ground truth)
                size, vise versa. 
            align_corners: bool, whether aligns the corner.
        """
        self.lossf = lossf
        self.resize_x = resize_x
        self.align_corners = align_corners

    def __call__(self, 
                 x,
                 y, 
                 **kwargs):
        """
        Args:
            x: [bs x c x 2D] float32 tensor, 
                the prediction. 
            y: [bs x c x 2D] float32 tensor or [bs x 2D] int64 tensor, 
                ground truth (label) tensor. 
        Returns:
            l: [bs x c x 2D] or [bs x 2D] float32 tensor, 
                the non reduced loss. 
        """
        if self.resize_x:
            x = auto_interpolate_2d(
                size=y.shape[-2:], 
                align_corners=self.align_corners)(x)
        else:
            y = auto_interpolate_2d(
                size=x.shape[-2:], 
                align_corners=self.align_corners)(y)
        l = self.lossf(x, y)
        return l

class affinity_2d(object):
    """
    Given a [bs x 2D] int array (label), output binary array with 
        size [bs x n x 2D] that tells the inter-pixel affinity: 
            0 - not same ID,
            1 - same ID. 
            ig - one out of two is ignore.
    """
    def __init__(self, 
                 deltas=[[-1, 0], [0, 1], [1, 0], [0, -1]],
                 ignore_label=-1):
        """
        Args:
            deltas: [n x 2] array, the delta shift to compute
                affiliation. 
                ex. [[-1, 0], [0, 1], [1, 0], [0, -1]] is the 'urdl'.
            ignore_label: int, the ignore label. Result will be 
                reviewed in the output mask matrix.
        """
        from .nputils import deltas_to_offsets
        self.paddings, self.s_deltas = deltas_to_offsets(deltas)
        self.ignore_label = ignore_label

    @batch_op(2)
    def __call__(self, 
                 x, 
                 **kwargs):
        """
        Args: 
            x: [bs x 2D] int nparray
        Returns:
            y: [bs x n_deltas x 2D] int nparray
                affinity between each pixel and its delta neighbor. 
                0 - not same ID,
                1 - same ID. 
                ig - one out of two is ignore.
        """
        d = x.device
        bn, h, w = x.shape
        ph1, pw1, ph2, pw2 = self.paddings
        xp = torch.empty(
            (bn, h+ph1+ph2, w+pw1+pw2), device=d, dtype=torch.int64)
        vp = torch.zeros(
            (bn, h+ph1+ph2, w+pw1+pw2), device=d, dtype=torch.uint8)

        xp[:, ph1:ph1+h, pw1:pw1+w] = x

        v = x!=self.ignore_label
        vp[:, ph1:ph1+h, pw1:pw1+w] = v

        xs, vs = [], []
        for dh, dw in self.s_deltas:
            xs.append(xp[:, dh:dh+h, dw:dw+w])
            vs.append(vp[:, dh:dh+h, dw:dw+w])

        y = torch.stack([x==i for i in xs], dim=1).long()
        m = torch.stack([v*i  for i in vs], dim=1)
        y[m==0] = self.ignore_label
        return y

class boundary_2d(object):
    """
    Given [bs x 2d] int array (label), find all boundary
        where two pixels under deltas has different label.
    """
    def __init__(self, 
                 deltas=[[-1, 0], [1, 0], [0, -1], [0, 1]],
                 ignore_label=-1,
                 reverse=False):
        """
        Args:
            reverse: bool,
                original 0-not boundary, 1-boundary.
                whether flip 0 and 1
        """
        self.aff_f = affinity_2d(
            deltas=deltas, ignore_label=ignore_label)
        self.ignore_label = ignore_label
        self.reverse = reverse

    @batch_op(2)
    def __call__(self,
                 x):
        """
        Args:
            x: [bs x 2D] int array, the labels
        Returns:
            y: [bs x 2D] int array, 
                0 - noboundary 
                1 - bourdary
                ig - ignore.                
        """
        x = self.aff_f(x)
        # x: [bs x n x 2D] 0=no-same ID, 1=same ID, ig=ignore 
        y = (torch.sum(x==0, dim=1)>0).long()
        # any direction is no-same ID to neighbor is a boundary
        # y: [bs x 2D] 0=nobdr, 1=bdr, 
        if self.reverse:
            y = 1-y

        ig = torch.sum(x==self.ignore_label, dim=1)>0
        # any direction is ignore means the pixel should be ignored.
        # ig: [bs x 2D] 0=noig, 1=ig, 

        y[ig] = self.ignore_label
        return y

class boundaryce_lossf(interpolate_2d_lossf):
    """
    Loss function that ignore all nonboundary part.
    """
    def __init__(self, 
                 ignore_label=None,
                 resize_x=False,
                 align_corners=True,
                 **kwargs):
        if ignore_label is None:
            ignore_label = -100
        super().__init__(
            lossf = nn.CrossEntropyLoss(ignore_index=ignore_label),
            resize_x = resize_x,
            align_corners = align_corners)
        self.boundary_f = boundary_2d(ignore_label=ignore_label)
        self.ignore_label = ignore_label

    def __call__(self, 
                 x,
                 y, 
                 **kwargs):
        """
        Args:
            x: [bs x c x 2D] float32 tensor, 
                the prediction. 
            y: [bs x c x 2D] float32 tensor or [bs x 2D] int64 tensor, 
                ground truth (label) tensor. 
        Returns:
            l: [bs x c x 2D] or [bs x 2D] float32 tensor, 
                the non reduced loss. 
        """
        if self.resize_x:
            x = auto_interpolate_2d(
                size=y.shape[-2:], 
                align_corners=self.align_corners)(x)
        else:
            y = auto_interpolate_2d(
                size=x.shape[-2:], 
                align_corners=self.align_corners)(y)

        # if it is ignore label -> we don't count it as boundary
        m = self.boundary_f(y)!=1
        y[m] = self.ignore_label
        l = self.lossf(x, y)
        return l

class roi_align(object):
    """
    This is the wrapper for torchvision.ops.roi_align
    """
    def __init__(self, ori_size, **kwargs):
        from torchvision.ops import roi_align
        self.f = roi_align
        self.ori_size = ori_size

    def __call__(self, x, bbx):
        return self.f(x, bbx, self.ori_size, aligned=True)
