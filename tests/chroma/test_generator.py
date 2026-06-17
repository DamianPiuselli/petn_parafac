"""
Unit tests for Chromatography Data Generator.
"""
import numpy as np
from src.chroma.generator import ChromatographicDataGenerator

def test_chroma_generator_init():
    gen = ChromatographicDataGenerator(num_samples=10, num_time=50, num_spec=40, num_components=2)
    assert gen.num_samples == 10
    assert gen.num_time == 50
    assert gen.num_spec == 40
    assert gen.num_components == 2

def test_chroma_generator_dataset():
    gen = ChromatographicDataGenerator(num_samples=8, num_time=60, num_spec=50, num_components=3, seed=42)
    dataset = gen.generate_dataset(noise_std=0.01)
    
    assert dataset['X'].shape == (8, 60, 50)
    assert dataset['A'].shape == (8, 3)
    assert dataset['B'].shape == (60, 3)
    assert dataset['C'].shape == (50, 3)
    assert len(dataset['shifts']) == 8
    assert len(dataset['stretches']) == 8
    assert np.all(dataset['A'] >= 0)
    assert np.all(dataset['B'] >= 0)
    assert np.all(dataset['C'] >= 0)
