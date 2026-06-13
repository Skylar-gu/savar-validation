"""CNN forecaster shape/param invariants (requires torch; runs on CPU)."""
import pytest

pytest.importorskip("torch")
import torch
import cnn_forecaster as cf


def test_forward_output_shape():
    m = cf.SpatioTemporalCNN(ny=50, nx=50, k=cf.K, base_ch=cf.BASE_CH).eval()
    x = torch.randn(2, 1, cf.K, 50, 50)        # (B, 1, k, ny, nx)
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (2, 1, 50, 50)    # predicts a single next frame


def test_param_count_in_spec_range():
    m = cf.SpatioTemporalCNN(ny=50, nx=50, k=cf.K, base_ch=cf.BASE_CH)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert 2_000_000 < n < 10_000_000          # pipeline spec target: 2M–10M


def test_window_size():
    assert cf.K == 3                            # k=3 per requirements
