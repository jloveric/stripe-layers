from stripe_layers.expansion import (
    PiecewisePolynomialSharedFullyConnected,
    PiecewisePolynomialShared,
)
import torch
import pytest


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("batches", [4, 5, 6])
@pytest.mark.parametrize("in_channels", [2, 3, 5])
@pytest.mark.parametrize("in_elements", [2, 6, 11])
@pytest.mark.parametrize("out_features", [2, 3, 4])
@pytest.mark.parametrize("segments", [2, 5])
def test_piecewise_polynomial_shared_fully_connected(
    n, batches, in_channels, in_elements, out_features, segments
):
    in_vals = torch.zeros((batches, in_channels, in_elements))

    # Only batch 0 has non-zero values
    in_vals[0, 0, :] = torch.arange(0, in_elements)

    layer = PiecewisePolynomialSharedFullyConnected(
        n,
        in_channels,
        in_elements,
        out_features,
        segments,
        length=2.0,
        weight_magnitude=1.0,
        periodicity=None,
        device="cpu",
    )

    # Only channel 0 has non-zero weights
    # so the resulting outputs in batch 1:
    # should all be the same
    with torch.no_grad():
        layer.w[:, 1:, :] = 0.0

    out_vals = layer(in_vals)

    assert out_vals.shape == torch.Size([batches, out_features])

    # The remaining outputs should be identical because the
    # batches are identical.
    for i in range(2, batches):
        assert torch.equal(out_vals[i], out_vals[1])


@pytest.mark.parametrize("n", [2, 7])
@pytest.mark.parametrize("batches", [4, 11])
@pytest.mark.parametrize("in_channels", [2, 3])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("in_elements", [4, 6])
@pytest.mark.parametrize("segments", [2, 5])
def test_piecewise_polynomial_shared(
    n, batches, in_channels, out_channels, in_elements, segments
):
    in_vals = torch.zeros((batches, in_channels, in_elements))

    # Only batch 0 has non-zero values
    rand_in = torch.rand(in_elements)
    in_vals[0, 0, :] = rand_in

    layer = PiecewisePolynomialShared(
        n,
        in_channels=in_channels,
        out_channels=out_channels,
        segments=segments,
        length=2.0,
        weight_magnitude=1.0,
        periodicity=2.0,
    )

    # Only channel 0 has non-zero weights
    # so the resulting outputs in batch 1:
    # should all be the same
    with torch.no_grad():
        layer.w[1:, :] = 0.0

    out_vals = layer(in_vals)

    print("out_vals.shape", out_vals.shape)
    assert out_vals.shape == torch.Size([batches, out_channels, in_elements])

    # The remaining outputs should be identical because the
    # batches are identical.
    for i in range(2, batches):
        assert torch.equal(out_vals[i], out_vals[1])
