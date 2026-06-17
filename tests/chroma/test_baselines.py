"""
Unit and Integration Tests for Chromatography Baselines.
Tests COW, COW-PARAFAC, and MCR-ALS.
"""
import numpy as np
from src.chroma.generator import ChromatographicDataGenerator
from src.chroma.baselines import (
    cow_1d,
    correlation_optimized_warping,
    COWPARAFAC,
    MCRALS,
    pearson_correlation
)

def test_pearson_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    assert np.allclose(pearson_correlation(x, y), 1.0)
    
    # Constant vector case
    x_const = np.array([1.0, 1.0, 1.0])
    assert pearson_correlation(x_const, y) == 0.0

def test_cow_1d():
    # Generate reference and a shifted profile
    t = np.linspace(0, 10, 50)
    T = np.exp(-0.5 * ((t - 5.0) / 1.0)**2)
    P = np.exp(-0.5 * ((t - 5.5) / 1.0)**2) # Shifted peak
    
    v_opt = cow_1d(T, P, N_seg=5, slack=3)
    assert len(v_opt) == 6
    assert v_opt[0] == 0
    assert v_opt[-1] == len(T) - 1

def test_cow_alignment_3d():
    # Generate small dataset
    gen = ChromatographicDataGenerator(num_samples=3, num_time=30, num_spec=20, num_components=2, seed=42)
    dataset = gen.generate_dataset(noise_std=0.01, max_shift=0.05, max_stretch=0.05)
    X = dataset['X']
    
    X_aligned, paths = correlation_optimized_warping(X, N_seg=5, slack=2)
    assert X_aligned.shape == X.shape
    assert len(paths) == X.shape[0]
    for path in paths:
        assert len(path) == 6

def test_cow_parafac_baseline():
    gen = ChromatographicDataGenerator(num_samples=4, num_time=30, num_spec=20, num_components=3, seed=42)
    dataset = gen.generate_dataset(noise_std=0.01, max_shift=0.03, max_stretch=0.03)
    X = dataset['X']
    
    model = COWPARAFAC(num_components=3, N_seg=5, slack=2)
    model.fit(X)
    
    assert model.A_.shape == (4, 3)
    assert model.B_.shape == (30, 3)
    assert model.C_.shape == (20, 3)
    assert not np.any(np.isnan(model.A_))
    assert not np.any(np.isnan(model.B_))
    assert not np.any(np.isnan(model.C_))

def test_mcr_als_baseline():
    gen = ChromatographicDataGenerator(num_samples=4, num_time=30, num_spec=20, num_components=3, seed=42)
    dataset = gen.generate_dataset(noise_std=0.01, max_shift=0.03, max_stretch=0.03)
    X = dataset['X']
    
    model = MCRALS(num_components=3, max_iter=20, tol=1e-5)
    model.fit(X)
    
    assert model.A_.shape == (4, 3)
    assert model.B_.shape == (30, 3)
    assert model.C_.shape == (20, 3)
    assert model.B_samples_.shape == (4, 30, 3)
    assert not np.any(np.isnan(model.A_))
    assert not np.any(np.isnan(model.B_))
    assert not np.any(np.isnan(model.C_))
