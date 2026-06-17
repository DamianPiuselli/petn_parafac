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
