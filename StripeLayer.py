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

        xv, yv = torch.meshgrid(
            [torch.arange(width), torch.arange(height)])
        xv = xv.to(device=device)
        yv = yv.to(device=device)
        #print('yv.device', yv.device)
        if rotations == 2:
            self.positions = [
                [xv, yv, (xv-yv)/2.0, (xv+yv)/2.0]]
        elif rotations == 1:
            self.positions = [xv, yv]
        else:
            line_list = []
            for i in range(rotations):
                theta = (math.pi/2.0)*(i/rotations)
                rot_x = math.cos(theta)
                rot_y = math.sin(theta)
                rot_sum = math.fabs(rot_x)+math.fabs(rot_y)

                # Add the line and the line orthogonal
                line_list.append((rot_x*xv+rot_y*yv)/rot_sum)
                line_list.append((rot_x*xv-rot_y*yv)/rot_sum)

            self.positions = line_list

        self.layer_list = []
        for i in range(2*rotations):
            self.layer_list.append(PiecewisePolynomialShared(
                n, in_channels=3, segments=segments, length=length, weight_magnitude=1.0, periodicity=None).to(device))

    def forward(self, x):

        accum = None
        for i in range(self.rotations):
            pos = self.positions[i]
            dl = position_encode(x, pos)/(0.5*self.max_dim) - 1.0
            dl = dl.flatten(start_dim=2)
            dl = self.layer_list[i](dl)
            if i == 0:
                accum = dl
            else:
                accum = accum + dl

        return accum
