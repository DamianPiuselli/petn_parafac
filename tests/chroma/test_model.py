"""
Unit tests for Chroma-PETN Model.
"""
import torch
import numpy as np
from src.chroma.model import ChromaPETN

def test_chroma_model_forward():
    num_samples = 5
    num_time = 30
    num_spec = 25
    model = ChromaPETN(num_samples, num_time, num_spec, num_components=3)
    
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
    model = ChromaPETN(num_samples, num_time, num_spec, num_components=2)
    
    # Introduce negative parameters and non-zero mean warps manually
    with torch.no_grad():
        model.sample_embeddings.weight[0, 0] = -1.0
        model.time_embeddings.weight[0, 0] = -0.5
        model.spec_embeddings.weight[0, 0] = -2.0
        
        model.warp_stretch.data.fill_(0.5) # mean will be 0.5
        model.warp_shift.data.fill_(-0.2)  # mean will be -0.2
        
    model.project_constraints()
    
    # Assert non-negativity
    assert torch.all(model.sample_embeddings.weight >= 0.0)
    assert torch.all(model.time_embeddings.weight >= 0.0)
    assert torch.all(model.spec_embeddings.weight >= 0.0)
    
    # Assert mean centering
    assert torch.abs(model.warp_stretch.mean()) < 1e-6
    assert torch.abs(model.warp_shift.mean()) < 1e-6

def test_chroma_model_quadratic():
    num_samples = 3
    num_time = 40
    num_spec = 30
    model = ChromaPETN(num_samples, num_time, num_spec, num_components=3, warp_type='quadratic')
    
    # Check forward pass
    batch_size = 15
    sample_idx = torch.randint(0, num_samples, (batch_size,))
    time_idx = torch.randint(0, num_time, (batch_size,))
    spec_idx = torch.randint(0, num_spec, (batch_size,))
    y_pred = model(sample_idx, time_idx, spec_idx)
    assert y_pred.shape == (batch_size,)
    
    # Introduce negative parameters and non-zero mean warps manually
    with torch.no_grad():
        model.warp_alpha.data.fill_(0.4)
        model.warp_beta.data.fill_(-0.3)
        model.warp_gamma.data.fill_(0.2)
        
    model.project_constraints()
    
    # Assert mean centering
    assert torch.abs(model.warp_alpha.mean()) < 1e-6
    assert torch.abs(model.warp_beta.mean()) < 1e-6
    assert torch.abs(model.warp_gamma.mean()) < 1e-6
    
    # Verify monotonicity: dt'/dt > 0.
    t = torch.linspace(0.0, 1.0, num_time)
    for i in range(num_samples):
        alpha = model.warp_alpha[i].item()
        beta = model.warp_beta[i].item()
        gamma = model.warp_gamma[i].item()
        t_warped = t - (alpha * (t ** 2) + beta * t + gamma)
        
        # Differences should be positive (monotonic increase)
        diffs = torch.diff(t_warped)
        assert torch.all(diffs > 0.0)

def test_chroma_model_spline():
    num_samples = 3
    num_time = 40
    num_spec = 30
    num_segments = 5
    model = ChromaPETN(num_samples, num_time, num_spec, num_components=3, warp_type='spline', num_segments=num_segments)
    
    # Check forward pass
    batch_size = 15
    sample_idx = torch.randint(0, num_samples, (batch_size,))
    time_idx = torch.randint(0, num_time, (batch_size,))
    spec_idx = torch.randint(0, num_spec, (batch_size,))
    y_pred = model(sample_idx, time_idx, spec_idx)
    assert y_pred.shape == (batch_size,)
    
    # Introduce non-zero mean parameters manually
    with torch.no_grad():
        model.warp_shift.data.fill_(0.3)
        model.warp_log_increments.data.fill_(0.5)
        
    model.project_constraints()
    
    # Assert mean centering
    assert torch.abs(model.warp_shift.mean()) < 1e-6
    assert torch.abs(model.warp_log_increments.mean(dim=0)).max() < 1e-6
    
    # Verify monotonicity: t_warped increases monotonically
    for i in range(num_samples):
        shift = model.warp_shift[i].item()
        log_inc = model.warp_log_increments[i]
        inc = (1.0 / num_segments) * torch.exp(log_inc)
        w = shift + torch.cumsum(torch.cat([torch.tensor([0.0]), inc]), dim=0)
        
        # w elements must be strictly increasing
        diffs = torch.diff(w)
        assert torch.all(diffs > 0.0)

def test_chroma_subclasses():
    from src.chroma.model import ChromaPETNBase, ChromaPETNGCMS, ChromaPETNDAD
    
    num_samples = 4
    num_time = 30
    num_spec = 20
    
    # 1. Test GC-MS subclass (no derivative buffer registered)
    model_gcms = ChromaPETNGCMS(num_samples, num_time, num_spec, num_components=2)
    assert isinstance(model_gcms, ChromaPETNBase)
    assert not hasattr(model_gcms, 'sg_kernel')
    
    # Create batch of coordinates
    batch_size = 10
    sample_idx = torch.randint(0, num_samples, (batch_size,))
    time_idx = torch.randint(0, num_time, (batch_size,))
    spec_idx = torch.randint(0, num_spec, (batch_size,))
    
    y_gcms = model_gcms(sample_idx, time_idx, spec_idx)
    assert y_gcms.shape == (batch_size,)
    
    # 2. Test DAD subclass (registers SG kernel buffer when derivative_order > 0)
    model_dad = ChromaPETNDAD(
        num_samples, num_time, num_spec, num_components=2,
        derivative_order=2, sg_window_size=11, sg_polyorder=2
    )
    assert isinstance(model_dad, ChromaPETNBase)
    assert hasattr(model_dad, 'sg_kernel')
    assert model_dad.sg_kernel.shape == (1, 1, 11)
    
    y_dad = model_dad(sample_idx, time_idx, spec_idx)
    assert y_dad.shape == (batch_size,)
