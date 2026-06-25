"""
Tests for PETN Custom Model.
"""
import torch
import numpy as np
from src.eem.model import PETNParafac

def get_dummy_wavelengths(num_ex=20, num_em=30):
    ex_wavelens = np.linspace(250.0, 400.0, num_ex)
    em_wavelens = np.linspace(300.0, 550.0, num_em)
    return ex_wavelens, em_wavelens

def test_model_initialization():
    """Verify PETNParafac initializes layers and weights with positive values under both modes."""
    ex_w, em_w = get_dummy_wavelengths(20, 30)
    ex_bg = torch.ones(20) * 0.05
    em_bg = torch.ones(30) * 0.05
    
    # 1. Test use_softplus=False
    model_clamp = PETNParafac(
        num_samples=10, num_ex=20, num_em=30,
        ex_wavelens=ex_w, em_wavelens=em_w,
        ex_bg=ex_bg, em_bg=em_bg, num_components=3,
        use_softplus=False
    )
    
    assert model_clamp.sample_embeddings.weight.shape == (10, 3)
    assert model_clamp.ex_embeddings.weight.shape == (20, 3)
    assert model_clamp.em_embeddings.weight.shape == (30, 3)
    assert model_clamp.alpha.shape == (3,)
    assert model_clamp.ex_bg.shape == (20,)
    assert model_clamp.em_bg.shape == (30,)
    
    assert torch.all(model_clamp.sample_embeddings.weight >= 0.1)
    assert torch.all(model_clamp.ex_embeddings.weight >= 0.1)
    assert torch.all(model_clamp.em_embeddings.weight >= 0.1)
    assert torch.all(model_clamp.alpha >= 0.01)
    assert torch.all(model_clamp.ex_bg == 0.05)
    assert torch.all(model_clamp.em_bg == 0.05)

    # 2. Test use_softplus=True
    model_sp = PETNParafac(
        num_samples=10, num_ex=20, num_em=30,
        ex_wavelens=ex_w, em_wavelens=em_w,
        ex_bg=ex_bg, em_bg=em_bg, num_components=3,
        use_softplus=True
    )
    
    assert model_sp.sample_embeddings.weight.shape == (10, 3)
    A_np, B_np, C_np, alpha_np, comp_weights_np = model_sp.get_resolved_factors()
    
    assert np.all(A_np >= 0.09)  # allow small floating point tolerances in softplus initialization mapping
    assert np.all(B_np >= 0.09)
    assert np.all(C_np >= 0.09)
    assert np.all(alpha_np >= 0.009)
    assert comp_weights_np is None
 
def test_model_forward():
    """Verify that model forward pass runs and returns the expected shape under both modes."""
    ex_w, em_w = get_dummy_wavelengths(20, 30)
    ex_bg = torch.ones(20) * 0.05
    em_bg = torch.ones(30) * 0.05
    
    for use_softplus in [False, True]:
        model = PETNParafac(
            num_samples=10, num_ex=20, num_em=30,
            ex_wavelens=ex_w, em_wavelens=em_w,
            ex_bg=ex_bg, em_bg=em_bg, num_components=3,
            use_softplus=use_softplus
        )
        
        sample_idx = torch.tensor([0, 1, 2, 0])
        ex_idx = torch.tensor([5, 10, 15, 5])
        em_idx = torch.tensor([12, 24, 8, 12])
        
        pred = model(sample_idx, ex_idx, em_idx)
        
        assert pred.shape == (4,)
        assert torch.all(pred >= 0.0)
 
def test_model_constraint_projection():
    """Verify that project_constraints correctly clips negative weights to 0.0 in clamp mode."""
    ex_w, em_w = get_dummy_wavelengths(5, 5)
    ex_bg = torch.ones(5) * 0.05
    em_bg = torch.ones(5) * 0.05
    model = PETNParafac(
        num_samples=5, num_ex=5, num_em=5,
        ex_wavelens=ex_w, em_wavelens=em_w,
        ex_bg=ex_bg, em_bg=em_bg, num_components=2,
        use_softplus=False
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
    
    for use_softplus in [False, True]:
        model = PETNParafac(
            num_samples=5, num_ex=25, num_em=35,
            ex_wavelens=ex_w, em_wavelens=em_w,
            ex_bg=ex_bg, em_bg=em_bg, num_components=2,
            use_softplus=use_softplus
        )
        
        E, M = model.get_learned_absorptivities()
        
        assert E.shape == (25, 2)
        assert M.shape == (35, 2)
        
        # Molar absorptivities must be non-negative
        assert np.all(E >= 0.0)
        # Emission molar absorptivities must be exactly zero (physics constraint)
        assert np.all(M == 0.0)

def test_learnable_and_dynamic_bg():
    """Verify learnable and dynamic background features work correctly."""
    ex_w, em_w = get_dummy_wavelengths(10, 15)
    ex_bg = torch.ones(10) * 0.1
    em_bg = torch.ones(15) * 0.1
    
    model = PETNParafac(
        num_samples=4, num_ex=10, num_em=15,
        ex_wavelens=ex_w, em_wavelens=em_w,
        ex_bg=ex_bg, em_bg=em_bg, num_components=2,
        use_softplus=True,
        learnable_bg=True,
        dynamic_bg=True
    )
    
    # Check learnable parameters
    assert isinstance(model.ex_bg, torch.nn.Parameter)
    assert isinstance(model.em_bg, torch.nn.Parameter)
    
    # Check drift embeddings
    assert model.ex_bg_drift.weight.shape == (4, 10)
    assert model.em_bg_drift.weight.shape == (4, 15)
    
    # Forward pass should run and use dynamic bg
    sample_idx = torch.tensor([0, 1, 2, 3])
    ex_idx = torch.tensor([1, 2, 3, 4])
    em_idx = torch.tensor([5, 6, 7, 8])
    
    pred = model(sample_idx, ex_idx, em_idx)
    assert pred.shape == (4,)
    
    # Compute smoothness loss
    smooth_loss = model.get_background_smoothness_loss()
    assert isinstance(smooth_loss, torch.Tensor)
    assert smooth_loss >= 0.0


