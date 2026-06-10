"""
EEM Synthetic Data Generator.
Responsible for generating trilinear synthetic tensors, simulating Rayleigh/Raman scattering,
and applying Inner Filter Effect (IFE) non-linearities.
"""
import numpy as np

class EEMGenerator:
    """
    Generates synthetic Excitation-Emission Matrix (EEM) data for benchmarking
    chemometric and neural network models.
    """
    def __init__(self, num_samples=20, num_ex=60, num_em=100, num_components=3,
                 ex_range=(250.0, 400.0), em_range=(300.0, 550.0), seed=42):
        self.num_samples = num_samples
        self.num_ex = num_ex
        self.num_em = num_em
        self.num_components = num_components
        self.ex_range = ex_range
        self.em_range = em_range
        self.rng = np.random.default_rng(seed)
        
        # Wavelength grids
        self.ex_wavelens = np.linspace(ex_range[0], ex_range[1], num_ex)
        self.em_wavelens = np.linspace(em_range[0], em_range[1], num_em)
        
        # Define component properties (means and standard deviations for Gaussians)
        # Component 1: Phenanthrene-like (short ex/em)
        # Component 2: Anthracene-like (medium ex/em)
        # Component 3: Humic-like (broad, longer ex/em)
        self.comp_ex_means = [270.0, 310.0, 350.0]
        self.comp_ex_stds = [15.0, 20.0, 25.0]
        self.comp_em_means = [340.0, 390.0, 450.0]
        self.comp_em_stds = [20.0, 25.0, 30.0]

    def generate_profiles(self):
        """
        Generates the true excitation (B) and emission (C) profiles.
        Returns:
            B: shape (num_ex, num_components)
            C: shape (num_em, num_components)
        """
        B = np.zeros((self.num_ex, self.num_components))
        C = np.zeros((self.num_em, self.num_components))
        
        for r in range(self.num_components):
            # Excitation Gaussian peak
            mu_ex = self.comp_ex_means[r]
            std_ex = self.comp_ex_stds[r]
            B[:, r] = np.exp(-0.5 * ((self.ex_wavelens - mu_ex) / std_ex) ** 2)
            
            # Emission Gaussian peak
            mu_em = self.comp_em_means[r]
            std_em = self.comp_em_stds[r]
            C[:, r] = np.exp(-0.5 * ((self.em_wavelens - mu_em) / std_em) ** 2)
            
        return B, C

    def generate_scores(self):
        """
        Generates positive sample concentrations/scores (A).
        Returns:
            A: shape (num_samples, num_components)
        """
        return self.rng.uniform(0.1, 1.5, size=(self.num_samples, self.num_components))

    def generate_dataset(self, noise_std=0.01):
        """
        Generates synthetic clean EEM data with additive Gaussian noise.
        Returns:
            dataset: dict containing:
                'X': 3D numpy array of shape (num_samples, num_ex, num_em)
                'X_true': 3D numpy array of shape (num_samples, num_ex, num_em) without noise
                'A': ground truth scores, shape (num_samples, num_components)
                'B': ground truth excitation, shape (num_ex, num_components)
                'C': ground truth emission, shape (num_em, num_components)
                'ex': excitation wavelengths
                'em': emission wavelengths
        """
        A = self.generate_scores()
        B, C = self.generate_profiles()
        
        # Calculate tensor outer product (PARAFAC model)
        # X(i,j,k) = sum_r A(i,r) * B(j,r) * C(k,r)
        X_true = np.einsum('ir,jr,kr->ijk', A, B, C)
        
        # Add homoscedastic noise
        noise = self.rng.normal(0.0, noise_std, size=X_true.shape)
        X = X_true + noise
        
        return {
            'X': X,
            'X_true': X_true,
            'A': A,
            'B': B,
            'C': C,
            'ex': self.ex_wavelens,
            'em': self.em_wavelens
        }
