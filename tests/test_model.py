"""
Tests for PINN Custom Model.
"""
import torch
import numpy as np
from src.model import PINNParafac

def get_dummy_wavelengths(num_ex=20, num_em=30):
    ex_wavelens = np.linspace(250.0, 400.0, num_ex)
    em_wavelens = np.linspace(300.0, 550.0, num_em)
    return ex_wavelens, em_wavelens

def test_model_initialization():
    """Verify PINNParafac initializes layers and weights with positive values."""
    ex_w, em_w = get_dummy_wavelengths(20, 30)
    ex_bg = torch.ones(20) * 0.05
    em_bg = torch.ones(30) * 0.05
    model = PINNParafac(
        num_samples=10, num_ex=20, num_em=30,
        ex_wavelens=ex_w, em_wavelens=em_w,
        ex_bg=ex_bg, em_bg=em_bg, num_components=3
    )
    
    assert model.sample_embeddings.weight.shape == (10, 3)
    assert model.ex_embeddings.weight.shape == (20, 3)
    assert model.em_embeddings.weight.shape == (30, 3)
    assert model.alpha.shape == (3,)
    assert model.ex_bg.shape == (20,)
    assert model.em_bg.shape == (30,)
    
    # Check that initialization is positive
    assert torch.all(model.sample_embeddings.weight >= 0.1)
    assert torch.all(model.ex_embeddings.weight >= 0.1)
    assert torch.all(model.em_embeddings.weight >= 0.1)
    assert torch.all(model.alpha >= 0.01)
    assert torch.all(model.ex_bg == 0.05)
    assert torch.all(model.em_bg == 0.05)
 
def test_model_forward():
    """Verify that model forward pass runs and returns the expected shape."""
    ex_w, em_w = get_dummy_wavelengths(20, 30)
    ex_bg = torch.ones(20) * 0.05
    em_bg = torch.ones(30) * 0.05
    model = PINNParafac(
        num_samples=10, num_ex=20, num_em=30,
        ex_wavelens=ex_w, em_wavelens=em_w,
        ex_bg=ex_bg, em_bg=em_bg, num_components=3
    )
    
    sample_idx = torch.tensor([0, 1, 2, 0])
    ex_idx = torch.tensor([5, 10, 15, 5])
    em_idx = torch.tensor([12, 24, 8, 12])
    
    pred = model(sample_idx, ex_idx, em_idx)
    
    assert pred.shape == (4,)
    # Output should be positive since embeddings and parameters are positive
    assert torch.all(pred >= 0.0)
 
def test_model_constraint_projection():
    """Verify that project_constraints correctly clips negative weights to 0.0."""
    ex_w, em_w = get_dummy_wavelengths(5, 5)
    ex_bg = torch.ones(5) * 0.05
    em_bg = torch.ones(5) * 0.05
    model = PINNParafac(
        num_samples=5, num_ex=5, num_em=5,
        ex_wavelens=ex_w, em_wavelens=em_w,
        ex_bg=ex_bg, em_bg=em_bg, num_components=2
    )
    
    # Manually inject negative weights
    with torch.no_grad():
        model.sample_embeddings.weight[0, 0] = -0.5
        model.ex_embeddings.weight[1, 1] = -1.2
        model.em_embeddings.weight[2, 0] = -0.01
        model.alpha[0] = -0.3
        
    assert torch.any(model.sample_embeddings.weight < 0.0)
    assert torch.any(model.ex_embeddings.weight < 0.0)
    assert torch.any(model.em_embeddings.weight < 0.0)
    assert torch.any(model.alpha < 0.0)
    
    # Project constraints
    model.project_constraints()
    
    # Verify no negative weights remain
    assert torch.all(model.sample_embeddings.weight >= 0.0)
    assert torch.all(model.ex_embeddings.weight >= 0.0)
    assert torch.all(model.em_embeddings.weight >= 0.0)
    assert torch.all(model.alpha >= 0.0)
    
    # Check that clipped weights are exactly 0.0
    assert model.sample_embeddings.weight[0, 0] == 0.0
    assert model.ex_embeddings.weight[1, 1] == 0.0
    assert model.em_embeddings.weight[2, 0] == 0.0
    assert model.alpha[0] == 0.0
 
def test_model_ife_head():
    """Verify that get_learned_absorptivities returns correct shapes and bounds."""
    ex_w, em_w = get_dummy_wavelengths(25, 35)
    ex_bg = torch.ones(25) * 0.05
    em_bg = torch.ones(35) * 0.05
    model = PINNParafac(
        num_samples=5, num_ex=25, num_em=35,
        ex_wavelens=ex_w, em_wavelens=em_w,
        ex_bg=ex_bg, em_bg=em_bg, num_components=2
    )
    
    E, M = model.get_learned_absorptivities()
    
    assert E.shape == (25, 2)
    assert M.shape == (35, 2)
    
    # Molar absorptivities must be non-negative
    assert np.all(E >= 0.0)
    # Emission molar absorptivities must be exactly zero (physics constraint)
    assert np.all(M == 0.0)
