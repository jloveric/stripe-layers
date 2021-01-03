import numpy as np
import math
import torch
from high_order_layers_torch.Basis import BasisExpand

import torch.nn as nn
import torch
from torch.autograd import Variable
from high_order_layers_torch.LagrangePolynomial import *
from high_order_layers_torch.utils import *


"""
class LagrangeBasis:
    def __init__(self, n: int, length: float = 2.0):
        self.n = n
        self.X = (length/2.0)*chebyshevLobatto(n)

    def __call__(self, x, j: int):

        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        b = torch.stack(b)
        ans = torch.prod(b, dim=0)
        return ans
"""


class BasisShared:
    # TODO: Is this the same as above? No! It is not!
    def __init__(self, n, basis):
        self.n = n
        self.basis = basis

    def interpolate(self, x, w):
        """
        Args:
            - x: size[batch, input]
            - w: size[batch, input, output, basis]
        Returns:
            - result: size[batch, output]
        """

        mat = []
        for j in range(self.n):
            basis_j = self.basis(x, j)
            mat.append(basis_j)
        mat = torch.stack(mat)

        #print('mat.shape', mat.shape, 'w.shape', w.shape)
        #print('mat.device', mat.device, 'w.device', w.device)
        assemble = torch.einsum("ijkl,jlki->jkl", mat, w)
        #print('shape.assemble', assemble.shape)

        return assemble


class LagrangePolyShared(BasisShared):
    def __init__(self, n: int, length: float = 2.0, **kwargs):
        super().__init__(n, LagrangeBasis(n, length=length), **kwargs)


'''
class PiecewiseExpandShared:
    """"
    Expand function with shared weights for each feature
    """"

    def __init__(self, basis, n: int, segments: int, length: float = 2.0, sparse=False):
        super().__init__()
        self._basis = basis
        self._n = n
        self._segments = segments
        self._expand = BasisExpand(basis, n)
        self._variables = (self._n-1)*self._segments+1
        self._length = length
        self._half = 0.5*length

    def __call__(self, x):
        """
        Apply basis function to each input.
        Args :
            x : Tensor of shape [batch, channels, x]
        Out :
            Tensor of shape [variables, batch, channels, x]
        """
        # get the segment index
        id_min = (((x+self._half)/self._length)*self._segments).long()
        device = x.device
        id_min = torch.where(id_min <= self._segments-1, id_min,
                             torch.tensor(self._segments-1, device=device))
        id_min = torch.where(id_min >= 0, id_min,
                             torch.tensor(0, device=device))
        id_max = id_min+1

        wid_min = id_min*(self._n-1)

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = self._length*((x-x_min)/(x_max-x_min))-self._half

        # These are the outputs, but they need to be in a sparse tensor
        # so they work with everything, do dense for now.
        out = self._expand(x_in)

        mat = torch.zeros(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            self._variables,
            device=device
        )

        wid_min_flat = wid_min.view(-1)

        wrange = wid_min_flat.unsqueeze(-1) + \
            torch.arange(self._n, device=device).view(-1)

        # This needs to be
        windex = (torch.arange(
            wrange.numel())//self._n)

        out = out.permute(1, 2, 3, 4, 0)

        mat_trans = mat.reshape(-1, self._variables)
        mat_trans[windex, wrange.view(-1)] = out.flatten()
        mat = mat_trans.reshape(
            mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3], mat.shape[4])

        mat = mat.permute(4, 0, 1, 2, 3)

        return mat

    def _eta(self, index):
        """
        Arg:
            - index is the segment index
        """
        eta = index/float(self._segments)
        return eta*2-1
'''


class PiecewiseShared(nn.Module):
    def __init__(self, n: int, in_channels: int, segments: int, length: int = 2.0, weight_magnitude=1.0, poly=None, periodicity=None, device='cuda', **kwargs):
        super().__init__()
        self._poly = poly(n)
        self._n = n
        self._segments = segments
        self.in_channels = in_channels
        self.periodicity = periodicity
        """
        self.w = torch.nn.Parameter(data=torch.Tensor(
            out_features, in_features, ((n-1)*segments+1)), requires_grad=True)
        """
        self.w = torch.nn.Parameter(data=torch.Tensor(
            in_channels, ((n-1)*segments+1)), requires_grad=True)
        self.w.data.uniform_(-weight_magnitude,
                             weight_magnitude)
        self.wrange = None
        self._length = length
        self._half = 0.5*length

    def forward(self, x):
        """
        Args :
            x : [batch, channels, nodes]
        """
        periodicity = self.periodicity
        if periodicity is not None:
            #x = make_periodic(x, periodicity)
            pass

        # get the segment index
        id_min = (((x+self._half)/self._length)*self._segments).long()
        device = id_min.device
        id_min = torch.where(id_min <= self._segments-1, id_min,
                             torch.tensor(self._segments-1, device=device))
        id_min = torch.where(id_min >= 0, id_min,
                             torch.tensor(0, device=device))
        id_max = id_min+1

        # determine which weights are active
        wid_min = (id_min*(self._n-1)).long()
        wid_max = (id_max*(self._n-1)).long()+1

        # Fill in the ranges
        wid_min_flat = wid_min.view(-1)
        wid_max_flat = wid_max.view(-1)
        wrange = wid_min_flat.unsqueeze(-1) + \
            torch.arange(self._n, device=device).view(-1)
        #print("wrange.shape",wrange.shape)
        # We only choose n interpolation points (weights) so
        # we divide by n instead of (segments*n...) therefore
        # the column index increases
        windex = (torch.arange(
            wrange.shape[0]*wrange.shape[1])//self._n) % (self.in_channels)
        wrange = wrange.flatten()
        
        # [channel index, weight index]
        w = self.w[windex, wrange]
        
        # Now
        #w = w.view(self.out_features, -1, self.in_features, self._n)
        #w = w.permute(1, 2, 0, 3)

        # TODO: Not totally convinced this is right.  Needs a test
        w = w.view(-1, wid_min.shape[-1], self.in_channels, self._n)
        #w = w.view(wid_min.shape[-1],-1, self.in_channels, self._n)

        #w = w.permute(1, 0, 2, 3)
        #print('w_final.shape', w.shape)

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = self._length*((x-x_min)/(x_max-x_min))-self._half

        #print('x_in.shape', x_in.shape)
        #print('w.shape', w.shape)

        result = self._poly.interpolate(x_in, w)
        return result

    def _eta(self, index):
        """
        Arg:
            - index is the segment index
        """
        eta = index/float(self._segments)
        return eta*2-1


class PiecewisePolynomialShared(PiecewiseShared):
    def __init__(self, n, in_channels, segments, length=2.0, weight_magnitude=1.0, periodicity: float = None, device='cuda', ** kwargs):
        super().__init__(n, in_channels, segments,
                         length, weight_magnitude, poly=LagrangePolyShared, periodicity=periodicity, device=device)
