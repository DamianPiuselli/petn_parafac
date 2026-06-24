"""
Preprocessing and resampling utilities for chromatography datasets.
Provides functions to align raw runs with non-uniform time axes onto a shared grid.
"""

import numpy as np
from scipy.interpolate import interp1d

def resample_chromatographic_runs(runs_data, runs_times, target_time_grid=None, 
                                  num_points=None, method='linear'):
    """
    Interpolates and aligns raw chromatographic runs with unique, sample-specific 
    non-uniform time arrays onto a shared, uniform time grid.
    
    Parameters
    ----------
    runs_data : list of numpy.ndarray
        List of 2D arrays, each of shape (N_time_i, Wavelengths) representing raw HPLC/GC runs.
    runs_times : list of numpy.ndarray
        List of 1D arrays, each of shape (N_time_i,) containing the non-uniform time coordinates for each run.
    target_time_grid : numpy.ndarray, optional
        A 1D array containing the target uniform time grid. If None, it will be automatically 
        generated using the overlap (intersection) of all time ranges.
    num_points : int, optional
        Number of points for the generated target_time_grid. If None, it defaults to the maximum 
        number of time points among the input runs.
    method : str, optional
        Interpolation method: 'linear' or 'cubic_spline'. Default is 'linear'.
        
    Returns
    -------
    X_dense : numpy.ndarray
        A single 3D dense tensor of shape (Samples, Time, Wavelengths) suitable for 
        SVD warm-start initialization and training in Chroma-PETN.
    time_grid : numpy.ndarray
        The 1D shared uniform time grid used for the resampling.
    """
    if len(runs_data) != len(runs_times):
        raise ValueError(
            f"Length of runs_data ({len(runs_data)}) must match length of runs_times ({len(runs_times)})."
        )
    
    if len(runs_data) == 0:
        raise ValueError("Input runs list cannot be empty.")
        
    # Check shapes of individual runs
    for idx, (data, times) in enumerate(zip(runs_data, runs_times)):
        if data.ndim != 2:
            raise ValueError(f"Run {idx} data must be a 2D array, but got shape {data.shape}.")
        if times.ndim != 1:
            raise ValueError(f"Run {idx} times must be a 1D array, but got shape {times.shape}.")
        if data.shape[0] != len(times):
            raise ValueError(
                f"Run {idx} has mismatch between time points in data ({data.shape[0]}) and times ({len(times)})."
            )

    num_wavelengths = runs_data[0].shape[1]
    for idx, data in enumerate(runs_data):
        if data.shape[1] != num_wavelengths:
            raise ValueError(
                f"Run {idx} has different number of spectral channels ({data.shape[1]}) than run 0 ({num_wavelengths})."
            )

    # Determine target time grid if not provided
    if target_time_grid is None:
        t_start = max(t.min() for t in runs_times)
        t_end = min(t.max() for t in runs_times)
        if t_start >= t_end:
            raise ValueError("No overlapping time range found across the raw runs.")
            
        if num_points is None:
            num_points = max(len(t) for t in runs_times)
            
        time_grid = np.linspace(t_start, t_end, num_points)
    else:
        time_grid = np.asarray(target_time_grid)
        if time_grid.ndim != 1:
            raise ValueError("target_time_grid must be a 1D array.")

    # Select interpolation kind
    if method == 'linear':
        kind = 'linear'
    elif method == 'cubic_spline':
        kind = 'cubic'
    else:
        raise ValueError(f"Unknown interpolation method: {method}. Supported: 'linear', 'cubic_spline'.")

    # Interpolate each run onto the shared grid
    num_samples = len(runs_data)
    num_time = len(time_grid)
    X_dense = np.zeros((num_samples, num_time, num_wavelengths))

    for idx, (data, times) in enumerate(zip(runs_data, runs_times)):
        # Interpolate along time axis (axis 0)
        # Using bounds_error=False, fill_value="extrapolate" or 0.0
        # Since chromatography signals should decay to baseline, filling out-of-bounds with 0.0 is physically sound
        f_interp = interp1d(times, data, axis=0, kind=kind, bounds_error=False, fill_value=0.0)
        X_dense[idx] = f_interp(time_grid)

    return X_dense, time_grid
