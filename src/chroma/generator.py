"""
Chromatography Synthetic Data Generator.
Simulates multi-component GC-MS or HPLC-DAD datasets with retention time shifts, stretching, and noise.
"""
import numpy as np

class ChromatographicDataGenerator:
    """
    Generates synthetic chromatographic data (Samples x Time x Spectra)
    with randomized retention time shifting, stretching, and noise.
    """
    def __init__(self, num_samples=15, num_time=100, num_spec=80, num_components=3, seed=42):
        self.num_samples = num_samples
        self.num_time = num_time
        self.num_spec = num_spec
        self.num_components = num_components
        
        self.rng = np.random.default_rng(seed)
        
        self.time_grid = np.linspace(0.0, 1.0, num_time)
        self.spec_grid = np.linspace(200.0, 400.0, num_spec)
        
        # Ground truth peak parameters (Gaussian chromatography profiles)
        self.true_peak_centers = [0.3, 0.5, 0.7]
        self.true_peak_widths = [0.05, 0.06, 0.08]
        
        # Ground truth spectral parameters (Gaussian-like spectra)
        self.true_spec_centers = [250.0, 300.0, 350.0]
        self.true_spec_widths = [20.0, 25.0, 30.0]

    def generate_profiles(self):
        """
        Generates the true aligned chromatography profiles (B) and spectra (C).
        Returns:
            B: shape (num_time, num_components)
            C: shape (num_spec, num_components)
        """
        # Chromatography profiles (B)
        B = np.zeros((self.num_time, self.num_components))
        for r in range(self.num_components):
            mu = self.true_peak_centers[r]
            sigma = self.true_peak_widths[r]
            B[:, r] = np.exp(-0.5 * ((self.time_grid - mu) / sigma) ** 2)
            
        # Spectral profiles (C)
        C = np.zeros((self.num_spec, self.num_components))
        for r in range(self.num_components):
            mu = self.true_spec_centers[r]
            sigma = self.true_spec_widths[r]
            C[:, r] = np.exp(-0.5 * ((self.spec_grid - mu) / sigma) ** 2)
            
        return B, C

    def generate_scores(self):
        """
        Generates concentrations/scores (A).
        Returns:
            A: shape (num_samples, num_components)
        """
        return self.rng.uniform(0.5, 2.0, size=(self.num_samples, self.num_components))

    def generate_dataset(self, noise_std=0.02, max_shift=0.06, max_stretch=0.08):
        """
        Generates the synthetic shifted dataset.
        
        Args:
            noise_std: standard deviation of homoscedastic Gaussian noise.
            max_shift: maximum shift offset (translation).
            max_stretch: maximum stretch factor (scaling).
            
        Returns:
            dataset: dict containing data matrix X, true matrices A, B, C, and shift parameters.
        """
        A = self.generate_scores()
        B, C = self.generate_profiles()
        
        # Generate sample-specific shifting parameters
        # Shift: constant offset delta_t
        # Stretch: scaling factor s
        # Warp formula: t_observed = s * t_true + delta_t
        # Therefore, t_true = (t_observed - delta_t) / s
        true_shifts = self.rng.uniform(-max_shift, max_shift, size=self.num_samples)
        true_stretches = self.rng.uniform(-max_stretch, max_stretch, size=self.num_samples)
        
        X = np.zeros((self.num_samples, self.num_time, self.num_spec))
        X_true_unshifted = np.zeros((self.num_samples, self.num_time, self.num_spec))
        
        for i in range(self.num_samples):
            shift = true_shifts[i]
            stretch = true_stretches[i]
            
            # Compute warped time grid for sample i
            # t_true = (t_observed - shift) / (1.0 + stretch)
            t_warped = (self.time_grid - shift) / (1.0 + stretch)
            
            # Interpolate B at warped time points
            B_warped = np.zeros((self.num_time, self.num_components))
            for r in range(self.num_components):
                mu = self.true_peak_centers[r]
                sigma = self.true_peak_widths[r]
                B_warped[:, r] = np.exp(-0.5 * ((t_warped - mu) / sigma) ** 2)
                
            # Compute shifted and unshifted signals
            X[i] = np.einsum('r,jr,kr->jk', A[i], B_warped, C)
            X_true_unshifted[i] = np.einsum('r,jr,kr->jk', A[i], B, C)
            
        # Add noise
        noise = self.rng.normal(0.0, noise_std, size=X.shape)
        X_noisy = X + noise
        
        return {
            'X': X_noisy,
            'X_true_unshifted': X_true_unshifted,
            'A': A,
            'B': B,
            'C': C,
            'shifts': true_shifts,
            'stretches': true_stretches
        }
