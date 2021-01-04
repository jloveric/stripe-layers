import torch
from expansion import *
import math
from position_encode import *


class StripePolynomial2d(torch.nn.Module):
    """
    Piecewise continuous polynomial.
    """

    def __init__(self, n: int, in_channels: int, width: int, height: int, segments: int, length: float = 2.0, rotations: int = 1, periodicity=None, device='cuda', weight_magnitude: int = 1.0):
        super().__init__()

        self.width = width
        self.height = height
        self.max_dim = max(width, height)
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
            r1 = ((r1-r1_min)/dr1-0.5)*self.max_dim
            r2 = ((r2-r2_min)/dr2-0.5)*self.max_dim
            print('r1_max', dr1,'r2_max', dr2)
            line_list.append(r1)
            line_list.append(r2)

        self.positions = line_list

        self.layer_list = []
        for i in range(2*rotations):
            self.layer_list.append(PiecewisePolynomialShared(
                n, in_channels=3, segments=segments, length=length, weight_magnitude=1.0, periodicity=periodicity).to(device))

    def forward(self, x):

        accum = None
        for i in range(self.rotations):
            pos = self.positions[i]

            # TODO this should be 0.5*self.max_dim
            dl = position_encode(x, pos)/(self.max_dim)
            #print('max dl',torch.max(dl))
            dl = dl.flatten(start_dim=2)
            dl = self.layer_list[i](dl)
            if i == 0:
                accum = dl
            else:
                accum = accum + dl

        return accum.reshape(*x.shape)
