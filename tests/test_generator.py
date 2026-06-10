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

def test_generator_scattering_and_mask():
    """Verify that scattering matrices and masks are generated with correct values and shapes."""
    gen = EEMGenerator(num_samples=5, num_ex=20, num_em=30, num_components=2)
    scatter, mask = gen.generate_scatter_and_mask()
    
    assert scatter.shape == (20, 30)
    assert mask.shape == (20, 30)
    
    # Mask values should only be 0.0 or 1.0
    assert np.all((mask == 0.0) | (mask == 1.0))
    
    # There should be some masked (0) and some unmasked (1) regions
    assert np.any(mask == 0.0)
    assert np.any(mask == 1.0)
    
    # 1st order Rayleigh diagonal (em == ex) should be masked (0)
    # Find indices where wavelength is closest
    for j, ex in enumerate(gen.ex_wavelens):
        closest_k = np.argmin(np.abs(gen.em_wavelens - ex))
        if np.abs(gen.em_wavelens[closest_k] - ex) < 2.0:
            assert mask[j, closest_k] == 0.0

def test_generator_dataset_corrupted():
    """Verify that dataset generation under scattering corruption behaves properly."""
    gen = EEMGenerator(num_samples=5, num_ex=20, num_em=30, num_components=2)
    data = gen.generate_dataset(noise_std=0.01, corrupt_scatter=True)
    
    assert 'mask' in data
    assert data['mask'] is not None
    assert data['mask'].shape == (20, 30)
    
    # Maximum value in X (with scattering) should be much higher than in clean X_true
    # because of high intensity scatter bands
    assert np.max(data['X']) > np.max(data['X_true'])

