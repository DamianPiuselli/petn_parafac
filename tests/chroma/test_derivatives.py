import pytest
import torch
import numpy as np
from scipy.signal import savgol_coeffs
from src.chroma import HPLC_PETN

def test_sg_coefficients():
    """Verify that model's registered SG kernel matches SciPy exactly."""
    window_size = 11
    polyorder = 2
    deriv = 2
    
    model = HPLC_PETN(
        num_samples=5,
        num_time=100,
        num_spec=50,
        num_components=3,
        derivative_order=deriv,
        sg_window_size=window_size,
        sg_polyorder=polyorder
    )
    
    expected_coeffs = savgol_coeffs(window_size, polyorder, deriv=deriv)[::-1]
    registered_coeffs = model.sg_kernel.cpu().numpy().flatten()
    
    np.testing.assert_allclose(registered_coeffs, expected_coeffs, rtol=1e-6)

def test_derivative_order_zero_identity():
    """Verify that derivative_order=0 produces identical results to raw forward."""
    num_samples = 3
    num_time = 50
    num_spec = 20
    num_components = 2
    
    model = HPLC_PETN(
        num_samples=num_samples,
        num_time=num_time,
        num_spec=num_spec,
        num_components=num_components,
        derivative_order=0
    )
    
    sample_idx = torch.randint(0, num_samples, (100,))
    time_idx = torch.randint(0, num_time, (100,))
    spec_idx = torch.randint(0, num_spec, (100,))
    
    # Check that it matches raw trilinear coordinate-based formula
    y_pred = model(sample_idx, time_idx, spec_idx)
    a, b, c = model._forward_raw_coo(sample_idx, time_idx, spec_idx)
    y_raw = torch.sum(a * b * c, dim=1)
    
    torch.testing.assert_close(y_pred, y_raw)

def test_gradient_flow():
    """Verify that gradients flow back from derivative loss to all model parameters."""
    num_samples = 3
    num_time = 50
    num_spec = 20
    num_components = 2
    
    model = HPLC_PETN(
        num_samples=num_samples,
        num_time=num_time,
        num_spec=num_spec,
        num_components=num_components,
        derivative_order=2,
        sg_window_size=7,
        sg_polyorder=2,
        warp_type='linear'
    )
    
    sample_idx = torch.randint(0, num_samples, (100,))
    time_idx = torch.randint(0, num_time, (100,))
    spec_idx = torch.randint(0, num_spec, (100,))
    
    y_pred = model(sample_idx, time_idx, spec_idx)
    loss = torch.mean(y_pred ** 2)
    loss.backward()
    
    # Assert gradients exist and are non-zero
    assert model.A.grad is not None
    assert torch.sum(torch.abs(model.A.grad)) > 0.0
    
    assert model.B.grad is not None
    assert torch.sum(torch.abs(model.B.grad)) > 0.0
    
    assert model.C.grad is not None
    assert torch.sum(torch.abs(model.C.grad)) > 0.0
    
    assert model.alpha.grad is not None
    assert torch.sum(torch.abs(model.alpha.grad)) > 0.0
    
    assert model.beta.grad is not None
    assert torch.sum(torch.abs(model.beta.grad)) > 0.0

def test_boundary_clamping():
    """Verify that the model handles boundary time steps (0 and num_time-1) without crashing."""
    num_samples = 2
    num_time = 10
    num_spec = 5
    
    model = HPLC_PETN(
        num_samples=num_samples,
        num_time=num_time,
        num_spec=num_spec,
        num_components=2,
        derivative_order=1,
        sg_window_size=5,
        sg_polyorder=2
    )
    
    # Edge coordinate inputs: time index 0 and num_time-1
    sample_idx = torch.tensor([0, 1])
    time_idx = torch.tensor([0, num_time - 1])
    spec_idx = torch.tensor([0, 2])
    
    # Should run successfully with boundary clamping
    try:
        y_pred = model(sample_idx, time_idx, spec_idx)
        assert y_pred.shape == (2,)
    except IndexError as e:
        pytest.fail(f"Boundary clamping failed with IndexError: {e}")
