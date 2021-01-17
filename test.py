from expansion import PiecewisePolynomialSharedFullyConnected
import torch


def test_piecewise_polynomial_shared_fully_connected():
    n = 2
    batches = 4
    in_channels = 3
    in_elements = 3*3
    out_features = 2
    in_vals = torch.zeros((batches, in_channels, in_elements))

    # Only batch 0 has non-zero values
    in_vals[0, 0, :] = torch.arange(0, 9)

    segments = 5

    layer = PiecewisePolynomialSharedFullyConnected(
        n, in_channels, in_elements,
        out_features, segments, length=2.0,
        weight_magnitude=1.0, periodicity=None, device='cpu'
    )

    # Only channel 0 has non-zero weights
    # so the resulting outputs in batch 1:
    # should all be the same
    layer.w[:, 1:, :] = 0.0

    out_vals = layer(in_vals)

    assert out_vals.shape == torch.Size([4, out_features])
    
    # The remaining outputs should be identical because the
    # batches are identical.
    for i in range(2,4) :
        assert torch.equal(out_vals[i], out_vals[1])
