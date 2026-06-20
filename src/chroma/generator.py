"""
Chromatography Synthetic Data Generators.
Provides specialized generators for GC-MS (sparse spectra, tailing peaks)
and HPLC-DAD (continuous spectra, baseline solvent drift).
"""
import numpy as np
from scipy.special import erf

def emg_profile(t, mu, sigma, tau):
    """
    Computes the Exponentially Modified Gaussian (EMG) profile at time points t.
    """
    if tau < 1e-5:
        # Fall back to symmetric Gaussian if tau is near-zero
        return np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    
    # Clip exponent argument to prevent overflow
    exp_arg = (sigma ** 2) / (2 * (tau ** 2)) - (t - mu) / tau
    exp_arg = np.clip(exp_arg, -50, 50)
    
    term1 = (1.0 / (2.0 * tau)) * np.exp(exp_arg)
    
    erf_arg = (t - mu) / (np.sqrt(2.0) * sigma) - sigma / (np.sqrt(2.0) * tau)
    term2 = 1.0 + erf(erf_arg)
    
    val = term1 * term2
    # Normalize peak max height to 1.0
    max_val = np.max(val)
    if max_val > 0:
        val = val / max_val
    return val

class BaseChromatographicGenerator:
    """
    Abstract base data generator for multi-component chromatography runs.
    """
    def __init__(self, num_samples=15, num_time=100, num_spec=80, num_components=3, seed=42):
        self.num_samples = num_samples
        self.num_time = num_time
        self.num_spec = num_spec
        self.num_components = num_components
        
        self.rng = np.random.default_rng(seed)
        
        self.time_grid = np.linspace(0.0, 1.0, num_time)
        self.spec_grid = np.linspace(200.0, 400.0, num_spec)
        
        # Distribute component peak centers and widths
        if num_components == 3:
            self.true_peak_centers = [0.3, 0.5, 0.7]
            self.true_peak_widths = [0.04, 0.05, 0.06]
        else:
            self.true_peak_centers = list(np.linspace(0.2, 0.8, num_components))
            self.true_peak_widths = [0.04 + 0.01 * r for r in range(num_components)]

    def generate_scores(self):
        """Generates concentrations/scores (A)."""
        return self.rng.uniform(0.5, 2.0, size=(self.num_samples, self.num_components))

    def generate_profiles(self):
        """Generates aligned profiles B and C. Must be implemented by subclasses."""
        raise NotImplementedError

    def generate_dataset(self, noise_std=0.02, max_shift=0.06, max_stretch=0.08, warp_type='linear'):
        """
        Generates the warped noisy dataset.
        """
        A = self.generate_scores()
        B, C = self.generate_profiles()
        
        # Warp parameters
        true_shifts = self.rng.uniform(-max_shift, max_shift, size=self.num_samples)
        true_stretches = self.rng.uniform(-max_stretch, max_stretch, size=self.num_samples)
        
        true_alphas = self.rng.uniform(-max_stretch * 0.5, max_stretch * 0.5, size=self.num_samples)
        true_betas = self.rng.uniform(-max_stretch * 0.8, max_stretch * 0.8, size=self.num_samples)
        true_gammas = self.rng.uniform(-max_shift, max_shift, size=self.num_samples)
        
        X_true = np.zeros((self.num_samples, self.num_time, self.num_spec))
        X_true_unshifted = np.zeros((self.num_samples, self.num_time, self.num_spec))
        
        for i in range(self.num_samples):
            if warp_type == 'linear':
                shift = true_shifts[i]
                stretch = true_stretches[i]
                t_warped = (self.time_grid - shift) / (1.0 + stretch)
            elif warp_type == 'quadratic':
                alpha = true_alphas[i]
                beta = true_betas[i]
                gamma = true_gammas[i]
                t_warped = self.time_grid - (alpha * (self.time_grid ** 2) + beta * self.time_grid + gamma)
            elif warp_type == 'spline':
                shift = true_shifts[i]
                stretch = true_stretches[i] * 0.5
                t_warped = self.time_grid - (shift + stretch * np.sin(np.pi * self.time_grid))
            else:
                raise ValueError(f"Unknown warp_type: {warp_type}")
                
            t_warped = np.clip(t_warped, 0.0, 1.0)
            
            # Interpolate B at warped time points
            B_warped = np.zeros((self.num_time, self.num_components))
            for r in range(self.num_components):
                # Interpolate canonical B[:, r] over t_warped
                B_warped[:, r] = np.interp(t_warped, self.time_grid, B[:, r])
                
            X_true[i] = np.einsum('r,jr,kr->jk', A[i], B_warped, C)
            X_true_unshifted[i] = np.einsum('r,jr,kr->jk', A[i], B, C)
            
        # Add subclass baseline if applicable
        baseline = getattr(self, 'generate_baseline', lambda: 0.0)()
        X_obs = X_true + baseline
        
        # Add noise
        noise = self.rng.normal(0.0, noise_std, size=X_obs.shape)
        X_noisy = X_obs + noise
        
        # Clip negative readings to 0 for GCMS (since mass spectra intensities are physical counts)
        if isinstance(self, GCMSDataGenerator):
            X_noisy = np.clip(X_noisy, 0.0, None)
            
        return {
            'X': X_noisy,
            'X_true_unshifted': X_true_unshifted,
            'A': A,
            'B': B,
            'C': C,
            'shifts': true_shifts,
            'stretches': true_stretches,
            'alphas': true_alphas,
            'betas': true_betas,
            'gammas': true_gammas,
            'warp_type': warp_type,
            'baseline': baseline
        }


class GCMSDataGenerator(BaseChromatographicGenerator):
    """
    Generator simulating sparse GC-MS data with Exponentially Modified Gaussian (EMG) peaks.
    """
    def __init__(self, num_samples=15, num_time=100, num_spec=80, num_components=3, seed=42):
        super().__init__(num_samples, num_time, num_spec, num_components=num_components, seed=seed)
        # Randomly assign tailing constants (tau) to each component
        self.true_taus = [0.03, 0.05, 0.07] if num_components == 3 else list(self.rng.uniform(0.02, 0.08, num_components))

    def generate_profiles(self):
        # 1. Chromatography profiles (B) with EMG tailing
        B = np.zeros((self.num_time, self.num_components))
        for r in range(self.num_components):
            mu = self.true_peak_centers[r]
            sigma = self.true_peak_widths[r]
            tau = self.true_taus[r]
            B[:, r] = emg_profile(self.time_grid, mu, sigma, tau)
            
        # 2. Sparse mass spectra (C)
        C = np.zeros((self.num_spec, self.num_components))
        for r in range(self.num_components):
            # Select random active mass fragments (e.g. 10% of channels)
            num_active = max(4, int(self.num_spec * 0.1))
            active_channels = self.rng.choice(self.num_spec, size=num_active, replace=False)
            intensities = self.rng.uniform(0.1, 1.0, size=num_active)
            # Re-scale to unit norm
            intensities /= np.linalg.norm(intensities)
            C[active_channels, r] = intensities
            
        return B, C


class HPLCDADDataGenerator(BaseChromatographicGenerator):
    """
    Generator simulating continuous HPLC-DAD data with symmetric peaks and baseline drift.
    """
    def __init__(self, num_samples=15, num_time=100, num_spec=80, num_components=3, seed=42):
        super().__init__(num_samples, num_time, num_spec, num_components=num_components, seed=seed)
        # Distribute broad wavelength spectrum peak parameters
        if num_components == 3:
            self.true_spec_centers = [240.0, 290.0, 340.0]
            self.true_spec_widths = [15.0, 20.0, 25.0]
        else:
            self.true_spec_centers = list(np.linspace(220.0, 380.0, num_components))
            self.true_spec_widths = [15.0 + 5.0 * r for r in range(num_components)]

    def generate_profiles(self):
        # 1. Symmetric Gaussian elution profiles (B)
        B = np.zeros((self.num_time, self.num_components))
        for r in range(self.num_components):
            mu = self.true_peak_centers[r]
            sigma = self.true_peak_widths[r]
            B[:, r] = np.exp(-0.5 * ((self.time_grid - mu) / sigma) ** 2)
            
        # 2. Continuous broad-band absorption spectra (C)
        C = np.zeros((self.num_spec, self.num_components))
        for r in range(self.num_components):
            mu = self.true_spec_centers[r]
            sigma = self.true_spec_widths[r]
            # UV-Vis peak spectrum
            C[:, r] = np.exp(-0.5 * ((self.spec_grid - mu) / sigma) ** 2)
            # Normalize spectrum
            norm_val = np.linalg.norm(C[:, r])
            if norm_val > 0:
                C[:, r] /= norm_val
                
        return B, C

    def generate_baseline(self):
        """Simulates continuous UV-Vis solvent absorption baseline drift."""
        # Solvent absorption spectrum centered at 210 nm
        solvent_spectrum = np.exp(-0.5 * ((self.spec_grid - 210.0) / 35.0) ** 2)
        
        baseline = np.zeros((self.num_samples, self.num_time, self.num_spec))
        for i in range(self.num_samples):
            # Sample-specific polynomial drift over time grid
            c0 = self.rng.uniform(0.04, 0.08)
            c1 = self.rng.uniform(-0.05, 0.05)
            c2 = self.rng.uniform(-0.02, 0.02)
            
            time_drift = c0 + c1 * self.time_grid + c2 * (self.time_grid ** 2)
            time_drift = np.clip(time_drift, 0.01, None)
            
            baseline[i] = np.outer(time_drift, solvent_spectrum)
            
        return baseline


# Compatibility layer: Maps ChromatographicDataGenerator to HPLCDADDataGenerator
class ChromatographicDataGenerator(HPLCDADDataGenerator):
    pass
