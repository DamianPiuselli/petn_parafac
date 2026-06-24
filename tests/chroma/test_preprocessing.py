import pytest
import numpy as np
from src.chroma.preprocessing import resample_chromatographic_runs

def test_resample_chromatographic_runs_basic():
    # Construct mock raw chromatographic runs with non-uniform time arrays
    # 2 runs (samples), 3 wavelengths/spectral channels
    t1 = np.array([1.0, 2.0, 3.1, 4.0, 5.2])
    data1 = np.stack([t1, t1 * 2, t1**2], axis=-1)  # shape: (5, 3)
    
    t2 = np.array([0.8, 1.9, 2.8, 4.2, 5.0, 5.8])
    data2 = np.stack([t2, t2 * 2, t2**2], axis=-1)  # shape: (6, 3)
    
    runs_data = [data1, data2]
    runs_times = [t1, t2]
    
    # 1. Test basic linear resampling
    X_dense, time_grid = resample_chromatographic_runs(
        runs_data, runs_times, num_points=10, method='linear'
    )
    
    # Assert correct dimensions
    assert X_dense.shape == (2, 10, 3)
    assert len(time_grid) == 10
    
    # Time grid should be overlapping range: [max(t1.min, t2.min), min(t1.max, t2.max)] -> [1.0, 5.2]
    assert np.isclose(time_grid.min(), 1.0)
    assert np.isclose(time_grid.max(), 5.2)
    
    # Check linear interpolation at t=1.0 for run 0 (which has exact matching point at index 0)
    assert np.allclose(X_dense[0, 0], [1.0, 2.0, 1.0])
    
    # 2. Test cubic spline resampling
    X_dense_cubic, time_grid_cubic = resample_chromatographic_runs(
        runs_data, runs_times, num_points=10, method='cubic_spline'
    )
    assert X_dense_cubic.shape == (2, 10, 3)
    assert np.allclose(time_grid_cubic, time_grid)

def test_resample_chromatographic_runs_custom_grid():
    t1 = np.array([1.0, 2.0, 3.0])
    data1 = np.ones((3, 4))
    
    runs_data = [data1]
    runs_times = [t1]
    
    custom_grid = np.array([1.5, 2.5])
    X_dense, time_grid = resample_chromatographic_runs(
        runs_data, runs_times, target_time_grid=custom_grid
    )
    
    assert X_dense.shape == (1, 2, 4)
    assert np.allclose(time_grid, custom_grid)

def test_resample_chromatographic_runs_validation_errors():
    # Empty inputs
    with pytest.raises(ValueError, match="Input runs list cannot be empty"):
        resample_chromatographic_runs([], [])
        
    # Length mismatch
    with pytest.raises(ValueError, match="Length of runs_data .* must match length of runs_times"):
        resample_chromatographic_runs([np.ones((3, 2))], [])
        
    # Dimension mismatch in data
    with pytest.raises(ValueError, match="data must be a 2D array"):
        resample_chromatographic_runs([np.ones(3)], [np.array([1, 2, 3])])
        
    # Dimension mismatch in times
    with pytest.raises(ValueError, match="times must be a 1D array"):
        resample_chromatographic_runs([np.ones((3, 2))], [np.ones((3, 2))])
        
    # Length mismatch between data time dimension and times array
    with pytest.raises(ValueError, match="mismatch between time points in data"):
        resample_chromatographic_runs([np.ones((3, 2))], [np.array([1, 2])])
        
    # Wavelength mismatch between different runs
    t1 = np.array([1, 2])
    t2 = np.array([1, 2])
    with pytest.raises(ValueError, match="different number of spectral channels"):
        resample_chromatographic_runs(
            [np.ones((2, 3)), np.ones((2, 4))], 
            [t1, t2]
        )
        
    # No overlap
    with pytest.raises(ValueError, match="No overlapping time range found"):
        resample_chromatographic_runs(
            [np.ones((2, 2)), np.ones((2, 2))],
            [np.array([1, 2]), np.array([3, 4])]
        )
