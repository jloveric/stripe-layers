import torch
from expansion import *
import math
from position_encode import *
from typing import Callable


def fully_connected_stripe(n, in_channels, out_features, segments, length=2.0, weight_magnitude=1.0, periodicity: float = None, device='cuda', ** kwargs):
    """
    Number of outputs is out_features and all inputs
    are connected to outputs.  Shared weights are applied
    in given ranges to given features.
    """
    def create():
        return PiecewisePolynomialSharedFullyConnected(
            n=n,
            in_channels=in_channels,
            out_features=out_features,
            segments=segments,
            length=length,
            weight_magnitude=weight_magnitude,
            periodicity=periodicity,
            device=device,
            **kwargs
        )

    return create


def stripe_expansion(n, in_channels, segments, length=2.0, weight_magnitude=1.0, periodicity: float = None, device='cuda', ** kwargs):
    """
    Number of outputs is the same as the number of inputs
    """
    def create():
        return PiecewisePolynomialShared(
            n,
            in_channels,
            segments,
            length,
            weight_magnitude,
            periodicity=periodicity,
            device=device,
            **kwargs
        )

    return create


class StripeLayer2d(torch.nn.Module):

    def __init__(self, layer_creator: Callable[[], torch.nn.Module], width: int, height: int, rotations: int = 1, device='cuda'):

        super().__init__()

        self.width = width
        self.height = height
        self.max_dim = max(width, height)
        self.ratio = self.max_dim/(self.max_dim+1)
        self.rotations = rotations

        # Create and center the coordinates
        # TODO: These don't have the ranges I thought so they will need to be fixed.
        xv, yv = torch.meshgrid(
            [torch.arange(width), torch.arange(height)])
        xv = xv.to(device=device)
        yv = yv.to(device=device)
        #print('yv.device', yv.device)

        # Coordinate values range from
        line_list = []
        for i in range(rotations):
            theta = (math.pi/2.0)*(i/rotations)
            rot_x = math.cos(theta)
            rot_y = math.sin(theta)
            rot_sum = math.fabs(rot_x)+math.fabs(rot_y)

            # Add the line and the line orthogonal
            r1 = (rot_x*xv+rot_y*yv)
            r1_max = torch.max(r1)
            r1_min = torch.min(r1)
            dr1 = r1_max-r1_min

            r2 = (rot_x*xv-rot_y*yv)
            r2_max = torch.max(r2)
            r2_min = torch.min(r2)
            dr2 = r2_max-r2_min

            # Rescale these so they have length segments
            # and are centered at (0,0)
            r1 = ((r1-r1_min)*self.ratio/dr1-0.5)*self.max_dim
            r2 = ((r2-r2_min)*self.ratio/dr2-0.5)*self.max_dim
            print('r1_max', dr1, 'r2_max', dr2)
            line_list.append(r1)
            line_list.append(r2)

        self.positions = line_list

        self.layer_list = []
        for i in range(2*rotations):
            self.layer_list.append(layer_creator().to(device))

    def forward(self, x):

        accum = None
        for i in range(len(self.positions)):
            pos = self.positions[i]

            # TODO this should be 0.5*self.max_dim
            dl = position_encode(x, pos) / (0.5*self.max_dim)
            #print('max pos', torch.max(pos))
            #print('max dl',torch.max(dl))
            #print('dl.shape', dl.shape)
            dl = dl.flatten(start_dim=2)
            dl = self.layer_list[i](dl)
            if i == 0:
                accum = dl
            else:
                accum = accum + dl
        accum = accum/(len(self.positions))
        
        return accum


class StripePolynomial2d(torch.nn.Module):
    """
    Piecewise continuous polynomial.
    """

    def __init__(self, n: int, in_channels: int, width: int, height: int,
                 segments: int, length: float = 2.0, rotations: int = 1,
                 periodicity=None, device='cuda', weight_magnitude: int = 1.0, layer=PiecewisePolynomialShared):
        super().__init__()

        self.width = width
        self.height = height
        self.max_dim = max(width, height)
        self.ratio = self.max_dim/(self.max_dim+1)
        self.rotations = rotations
        self.segments = segments

        # Create and center the coordinates
        # TODO: These don't have the ranges I thought so they will need to be fixed.
        xv, yv = torch.meshgrid(
            [torch.arange(width), torch.arange(height)])
        xv = xv.to(device=device)
        yv = yv.to(device=device)
        #print('yv.device', yv.device)

        # Coordinate values range from
        line_list = []
        for i in range(rotations):
            theta = (math.pi/2.0)*(i/rotations)
            rot_x = math.cos(theta)
            rot_y = math.sin(theta)
            rot_sum = math.fabs(rot_x)+math.fabs(rot_y)

            # Add the line and the line orthogonal
            r1 = (rot_x*xv+rot_y*yv)
            r1_max = torch.max(r1)
            r1_min = torch.min(r1)
            dr1 = r1_max-r1_min

            r2 = (rot_x*xv-rot_y*yv)
            r2_max = torch.max(r2)
            r2_min = torch.min(r2)
            dr2 = r2_max-r2_min

            # Rescale these so they have length segments
            # and are centered at (0,0)
            r1 = ((r1-r1_min)*self.ratio/dr1-0.5)*self.max_dim
            r2 = ((r2-r2_min)*self.ratio/dr2-0.5)*self.max_dim
            print('r1_max', dr1, 'r2_max', dr2)
            line_list.append(r1)
            line_list.append(r2)

        self.positions = line_list

        self.layer_list = []
        for i in range(2*rotations):
            self.layer_list.append(layer(
                n, in_channels=in_channels, segments=segments, length=length, weight_magnitude=weight_magnitude, periodicity=periodicity).to(device))

    def forward(self, x):

        accum = None
        for i in range(len(self.positions)):
            pos = self.positions[i]

            # TODO this should be 0.5*self.max_dim
            dl = position_encode(x, pos) / (0.5*self.max_dim)
            #print('max pos', torch.max(pos))
            #print('max dl',torch.max(dl))
            #print('dl.shape', dl.shape)
            dl = dl.flatten(start_dim=2)
            dl = self.layer_list[i](dl)
            if i == 0:
                accum = dl
            else:
                accum = accum + dl
        accum = accum/(len(self.positions))
        return accum.reshape(*x.shape)
