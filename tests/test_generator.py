"""
Tests for EEM Synthetic Data Generator.
"""
import numpy as np
from src.generator import EEMGenerator

def test_generator_initialization():
    """Verify that EEMGenerator initializes with correct shapes and wavelengths."""
    gen = EEMGenerator(num_samples=15, num_ex=40, num_em=80, num_components=3)
    assert gen.num_samples == 15
    assert len(gen.ex_wavelens) == 40
    assert len(gen.em_wavelens) == 80
    assert gen.num_components == 3

def test_generator_profiles():
    """Verify that generated loading profiles are Gaussian and positive."""
    gen = EEMGenerator(num_samples=10, num_ex=50, num_em=90, num_components=3)
    B, C = gen.generate_profiles()
    
    assert B.shape == (50, 3)
    assert C.shape == (90, 3)
    
    # Gaussian profiles should peak at 1.0 (or close to it depending on grid coverage)
    # and be strictly positive
    assert np.all(B >= 0)
    assert np.all(C >= 0)
    assert np.any(B > 0.9)
    assert np.any(C > 0.9)

def test_generator_scores():
    """Verify that scores (concentrations) are strictly positive."""
    gen = EEMGenerator(num_samples=12, num_components=3)
    A = gen.generate_scores()
    
    assert A.shape == (12, 3)
    assert np.all(A >= 0.1)
    assert np.all(A <= 1.5)

def test_generator_dataset():
    """Verify that generated full EEM dataset matches expected shape and values."""
    gen = EEMGenerator(num_samples=10, num_ex=30, num_em=40, num_components=2)
    data = gen.generate_dataset(noise_std=0.01)
    
    assert 'X' in data
    assert 'X_true' in data
    assert 'A' in data
    assert 'B' in data
    assert 'C' in data
    
    assert data['X'].shape == (10, 30, 40)
    assert data['X_true'].shape == (10, 30, 40)
    assert data['A'].shape == (10, 2)
    assert data['B'].shape == (30, 2)
    assert data['C'].shape == (40, 2)
