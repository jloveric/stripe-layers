from expansion import PiecewisePolynomialSharedFullyConnected
import torch


def test_piecewise_polynomial_shared_fully_connected():
    n = 2
    batches = 4
    in_channels = 4
    in_elements = 3*3
    out_features = 2
    in_vals = torch.rand((batches, in_channels, in_elements))

    segments = 5

    layer = PiecewisePolynomialSharedFullyConnected(
        n, in_channels, in_elements,
        out_features, segments, length=2.0,
        weight_magnitude=1.0, periodicity=None, device='cpu'
    )

    out_vals = layer(in_vals)
    assert out_vals.shape == torch.Size([4, out_features])