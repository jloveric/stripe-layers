import torch
from expansion import *

class StripePolynomial2d(torch.nn.Module):
    """
    Piecewise continuous polynomial.
    """
    def __init__(self, n, in_channels, segments, length=2.0, periodicity=None, device='cuda'):
        super().__init__()
        
         if rotations == 2:
            torch_position = torch.cat(
                [xv, yv, (xv-yv)/2.0, (xv+yv)/2.0], dim=2)
            torch_position = torch_position.reshape(-1, 4)
        elif rotations == 1:
            torch_position = torch.cat([xv, yv], dim=2)
            torch_position = torch_position.reshape(-1, 2)
        else:
            line_list = []
            for i in range(rotations):
                theta = (math.pi/2.0)*(i/rotations)
                print('theta', theta)
                rot_x = math.cos(theta)
                rot_y = math.sin(theta)
                rot_sum = math.fabs(rot_x)+math.fabs(rot_y)

                # Add the line and the line orthogonal
                line_list.append((rot_x*xv+rot_y*yv)/rot_sum)
                line_list.append((rot_x*xv-rot_y*yv)/rot_sum)

        torch_position = torch.cat(line_list, dim=2)
        torch_position = torch_position.reshape(-1, 2*rotations)

        self.layer1x = PiecewisePolynomialShared(
            n, in_channels=3, segments=segments, length=2.0, weight_magnitude=1.0, periodicity=None)
        self.layer1y = PiecewisePolynomialShared(
            n, in_channels=3, segments=segments, length=2.0, weight_magnitude=1.0, periodicity=None)
        self.layer1xy = PiecewisePolynomialShared(
            n, in_channels=3, segments=segments, length=2.0, weight_magnitude=1.0, periodicity=None)
        self.layer1yx = PiecewisePolynomialShared(
            n, in_channels=3, segments=segments, length=2.0, weight_magnitude=1.0, periodicity=None)

        xv, yv = torch.meshgrid(
            [torch.arange(32), torch.arange(32)])
        self.xv = xv.to(device=device)
        self.yv = yv.to(device=device)

    def forward(self, x):

        # print('torch.max',torch.max(xv))

        # We want the result between -1 and 1
        dx = position_encode(x, self.xv)/16.0 - 1.0
        dy = position_encode(x, self.yv)/16.0 - 1.0
        
        dxy = position_encode(x, self.xv-self.yv)/32.0 - 1.0
        dyx = position_encode(x, self.xv+self.yv)/32.0 - 1.0
        
        #position_encode(x, px-py)
        # position_encode(x,px+py)

        # Keep the batch and channels
        dx = dx.flatten(start_dim=2)
        ans_dx = self.layer1x(dx)

        dy = dy.flatten(start_dim=2)
        ans_dy = self.layer1y(dy)

        dxy = dxy.flatten(start_dim=2)
        ans_dxy = self.layer1xy(dxy)

        dyx = dyx.flatten(start_dim=2)
        ans_dyx = self.layer1yx(dyx)

        ans = ans_dx+ans_dy+ans_dxy+ans_dyx
        ans = ans.reshape(-1, x.shape[1], x.shape[2], x.shape[3])

        # do some average pooling
        ans = self.pool(ans)

        ans = self.convolution1(ans)
        #print('ans.shape', ans.shape)
        ans = ans.flatten(start_dim=1)
        ans = self.fc1(ans)

        return ans