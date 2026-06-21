"""
Unit tests for Chroma-PETN Model.
"""
import torch
import numpy as np
from src.chroma import BaseChromaPETN, HPLC_PETN, GCMS_PETN

def test_chroma_model_forward():
    num_samples = 5
    num_time = 30
    num_spec = 25
    model = HPLC_PETN(num_samples, num_time, num_spec, num_components=3, derivative_order=0)
    
    # Create batch of coordinates
    batch_size = 10
    sample_idx = torch.randint(0, num_samples, (batch_size,))
    time_idx = torch.randint(0, num_time, (batch_size,))
    spec_idx = torch.randint(0, num_spec, (batch_size,))
    
    y_pred = model(sample_idx, time_idx, spec_idx)
    assert y_pred.shape == (batch_size,)

def test_chroma_model_constraints():
    num_samples = 4
    num_time = 20
    num_spec = 15
    for cs_warp in [False, True]:
        model = HPLC_PETN(num_samples, num_time, num_spec, num_components=2, derivative_order=0, component_specific_warp=cs_warp)
        
        # Introduce negative parameters manually
        with torch.no_grad():
            model.A[0, 0] = -1.0
            model.B[0, 0] = -0.5
            model.C[0, 0] = -2.0
            
            model.alpha.data.fill_(0.5) # mean will be 0.5
            model.beta.data.fill_(-0.2)  # mean will be -0.2
            
        model.project_constraints()
        
        # Assert non-negativity
        assert torch.all(model.A >= 0.0)
        assert torch.all(model.B >= 0.0)
        assert torch.all(model.C >= 0.0)
        
        # Assert mean centering
        assert torch.abs(model.alpha.mean()) < 1e-6
        assert torch.abs(model.beta.mean()) < 1e-6

def test_chroma_model_quadratic():
    num_samples = 3
    num_time = 40
    num_spec = 30
    for cs_warp in [False, True]:
        model = HPLC_PETN(num_samples, num_time, num_spec, num_components=3, warp_type='quadratic', derivative_order=0, component_specific_warp=cs_warp)
        
        # Check forward pass
        batch_size = 15
        sample_idx = torch.randint(0, num_samples, (batch_size,))
        time_idx = torch.randint(0, num_time, (batch_size,))
        spec_idx = torch.randint(0, num_spec, (batch_size,))
        y_pred = model(sample_idx, time_idx, spec_idx)
        assert y_pred.shape == (batch_size,)
        
        # Introduce negative parameters and non-zero mean warps manually
        with torch.no_grad():
            model.alpha.data.fill_(0.4)
            model.beta.data.fill_(-0.3)
            model.gamma.data.fill_(0.2)
            
        model.project_constraints()
        
        # Assert mean centering
        assert torch.abs(model.alpha.mean()) < 1e-6
        assert torch.abs(model.beta.mean()) < 1e-6
        assert torch.abs(model.gamma.mean()) < 1e-6
        
        # Verify monotonicity: dt'/dt > 0.
        t = torch.linspace(0.0, 1.0, num_time)
        dim2 = model.num_components if cs_warp else 1
        for i in range(num_samples):
            for r in range(dim2):
                alpha = model.alpha[i, r].item()
                beta = model.beta[i, r].item()
                gamma = model.gamma[i, r].item()
                t_warped = t - (alpha * (t ** 2) + beta * t + gamma)
                
                # Differences should be positive (monotonic increase)
                diffs = torch.diff(t_warped)
                assert torch.all(diffs > 0.0)

def test_chroma_model_spline():
    num_samples = 3
    num_time = 40
    num_spec = 30
    num_segments = 5
    for cs_warp in [False, True]:
        model = HPLC_PETN(num_samples, num_time, num_spec, num_components=3, warp_type='spline', num_segments=num_segments, derivative_order=0, component_specific_warp=cs_warp)
        
        # Check forward pass
        batch_size = 15
        sample_idx = torch.randint(0, num_samples, (batch_size,))
        time_idx = torch.randint(0, num_time, (batch_size,))
        spec_idx = torch.randint(0, num_spec, (batch_size,))
        y_pred = model(sample_idx, time_idx, spec_idx)
        assert y_pred.shape == (batch_size,)
        
        # Introduce non-zero mean parameters manually
        with torch.no_grad():
            model.beta.data.fill_(0.3)
            model.log_increments.data.fill_(0.5)
            
        model.project_constraints()
        
        # Assert mean centering
        assert torch.abs(model.beta.mean()) < 1e-6
        assert torch.abs(model.log_increments.mean(dim=0)).max() < 1e-6
        
        # Verify monotonicity: t_warped increases monotonically
        dim2 = model.num_components if cs_warp else 1
        for i in range(num_samples):
            for r in range(dim2):
                shift = model.beta[i, r].item()
                log_inc = model.log_increments[i, :, r]
                inc = (1.0 / num_segments) * torch.exp(log_inc)
                w = shift + torch.cumsum(torch.cat([torch.tensor([0.0], device=inc.device), inc]), dim=0)
                
                # w elements must be strictly increasing
                diffs = torch.diff(w)
                assert torch.all(diffs > 0.0)

def test_chroma_subclasses():
    num_samples = 4
    num_time = 30
    num_spec = 20
    
    # 1. Test GC-MS subclass
    model_gcms = GCMS_PETN(num_samples, num_time, num_spec, num_components=2)
    assert isinstance(model_gcms, BaseChromaPETN)
    assert hasattr(model_gcms, 'delta_B')
    assert not hasattr(model_gcms, 'sg_kernel')
    
    # Create batch of coordinates
    batch_size = 10
    sample_idx = torch.randint(0, num_samples, (batch_size,))
    time_idx = torch.randint(0, num_time, (batch_size,))
    spec_idx = torch.randint(0, num_spec, (batch_size,))
    
    y_gcms = model_gcms(sample_idx, time_idx, spec_idx)
    assert y_gcms.shape == (batch_size,)
    
    # 2. Test HPLC/DAD subclass
    model_dad = HPLC_PETN(
        num_samples, num_time, num_spec, num_components=2,
        derivative_order=2, sg_window_size=11, sg_polyorder=2
    )
    assert isinstance(model_dad, BaseChromaPETN)
    assert hasattr(model_dad, 'sg_kernel')
    assert model_dad.sg_kernel.shape == (1, 1, 11)
    
    y_dad = model_dad(sample_idx, time_idx, spec_idx)
    assert y_dad.shape == (batch_size,)


def test_chroma_svd_initialization():
    num_samples = 5
    num_time = 30
    num_spec = 20
    num_components = 3
    
    # Generate random positive data
    X = torch.rand(num_samples, num_time, num_spec) + 0.1
    
    # Test on HPLC model
    model_dad = HPLC_PETN(num_samples, num_time, num_spec, num_components=num_components)
    model_dad.init_from_svd(X)
    
    # Verify shapes and non-negativity
    assert model_dad.A.shape == (num_samples, num_components)
    assert model_dad.B.shape == (num_time, num_components)
    assert model_dad.C.shape == (num_spec, num_components)
    
    assert torch.all(model_dad.A >= 0.0)
    assert torch.all(model_dad.B >= 0.0)
    assert torch.all(model_dad.C >= 0.0)
    
    # Now check GC-MS model
    model_gcms = GCMS_PETN(num_samples, num_time, num_spec, num_components=num_components)
    # Set warping/delta_B to non-zero values to make sure they are reset to 0 during svd init
    with torch.no_grad():
        model_gcms.alpha.fill_(0.1)
        model_gcms.delta_B.fill_(1.5)
        
    model_gcms.init_from_svd(X)
    
    assert torch.all(model_gcms.alpha == 0.0)
    assert torch.all(model_gcms.delta_B == 0.0)
