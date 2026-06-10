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

    def generate_scatter_and_mask(self):
        """
        Generates 1st and 2nd order Rayleigh scattering and solvent Raman scattering
        along with a binary mask that equals 0 on the scattering diagonals and 1 elsewhere.
        
        Returns:
            scatter: 2D numpy array of shape (num_ex, num_em) containing the combined scattering intensity
            mask: 2D numpy array of shape (num_ex, num_em) containing 0 in scattering regions and 1 elsewhere
        """
        scatter = np.zeros((self.num_ex, self.num_em))
        mask = np.ones((self.num_ex, self.num_em))
        
        # Scattering parameters
        # 1st-order Rayleigh: em = ex
        h_ray1 = 15.0
        sigma_ray1 = 3.5
        
        # 2nd-order Rayleigh: em = 2 * ex
        h_ray2 = 5.0
        sigma_ray2 = 4.0
        
        # Solvent Raman: 3400 cm-1 shift
        h_raman = 4.0
        sigma_raman = 4.0
        
        for j in range(self.num_ex):
            ex = self.ex_wavelens[j]
            # Raman peak position for this excitation
            em_raman = ex / (1.0 - 3.4e-4 * ex)
            
            for k in range(self.num_em):
                em = self.em_wavelens[k]
                
                # 1. 1st-order Rayleigh
                dist_ray1 = em - ex
                val_ray1 = h_ray1 * np.exp(-0.5 * (dist_ray1 / sigma_ray1) ** 2)
                
                # 2. 2nd-order Rayleigh
                dist_ray2 = em - 2 * ex
                val_ray2 = h_ray2 * np.exp(-0.5 * (dist_ray2 / sigma_ray2) ** 2)
                
                # 3. Water Raman
                dist_raman = em - em_raman
                val_raman = h_raman * np.exp(-0.5 * (dist_raman / sigma_raman) ** 2)
                
                # Combine scattering intensities
                scatter[j, k] = val_ray1 + val_ray2 + val_raman
                
                # Mask out regions where scattering is non-negligible
                # Mask out within 2.0 standard deviations (95% of peak width)
                if (abs(dist_ray1) <= 2.0 * sigma_ray1 or 
                    abs(dist_ray2) <= 2.0 * sigma_ray2 or 
                    abs(dist_raman) <= 2.0 * sigma_raman):
                    mask[j, k] = 0.0
                    
        return scatter, mask

    def generate_ife_attenuation(self):
        """
        Generates a 2D attenuation matrix gamma of shape (num_ex, num_em)
        based on the Lakowicz geometric correction formula and background matrix absorbance.
        
        Returns:
            gamma: 2D numpy array of shape (num_ex, num_em) containing values in (0, 1]
        """
        # Background CDOM absorption parameters
        c_bg = 0.25
        eta = 0.015
        lambda_0 = 240.0
        
        # Absorbance profile at excitation and emission wavelengths
        A_ex = c_bg * np.exp(-eta * (self.ex_wavelens - lambda_0))
        A_em = c_bg * np.exp(-eta * (self.em_wavelens - lambda_0))
        
        # Calculate attenuation matrix: 10^(-(A_ex + A_em))
        gamma = 10.0 ** (-(A_ex[:, np.newaxis] + A_em[np.newaxis, :]))
        return gamma

    def generate_dataset(self, noise_std=0.01, corrupt_scatter=False, corrupt_ife=False):
        """
        Generates synthetic EEM data with additive Gaussian noise and optional scattering/IFE.
        
        Args:
            noise_std: standard deviation of homoscedastic noise.
            corrupt_scatter: if True, inject Rayleigh and Raman scatter lines and output a mask.
            corrupt_ife: if True, apply Inner Filter Effect (matrix absorption attenuation) to X_true.
            
        Returns:
            dataset: dict containing:
                'X': 3D numpy array of shape (num_samples, num_ex, num_em)
                'X_true': 3D numpy array of shape (num_samples, num_ex, num_em) without noise/scatter/IFE
                'A': ground truth scores, shape (num_samples, num_components)
                'B': ground truth excitation, shape (num_ex, num_components)
                'C': ground truth emission, shape (num_em, num_components)
                'ex': excitation wavelengths
                'em': emission wavelengths
                'mask': 2D numpy array of shape (num_ex, num_em) or None
                'gamma': 2D numpy array of shape (num_ex, num_em) or None
        """
        A = self.generate_scores()
        B, C = self.generate_profiles()
        
        # Calculate tensor outer product (PARAFAC model)
        X_true = np.einsum('ir,jr,kr->ijk', A, B, C)
        
        # Apply IFE if requested
        gamma = None
        if corrupt_ife:
            gamma = self.generate_ife_attenuation()
            X_signal = X_true * gamma[np.newaxis, :, :]
        else:
            X_signal = X_true.copy()
            
        mask = None
        if corrupt_scatter:
            scatter_2d, mask_2d = self.generate_scatter_and_mask()
            # Add scattering to every sample
            X_corrupted = X_signal + scatter_2d[np.newaxis, :, :]
            mask = mask_2d
        else:
            X_corrupted = X_signal
            
        # Add noise
        noise = self.rng.normal(0.0, noise_std, size=X_corrupted.shape)
        X = X_corrupted + noise
            
        return {
            'X': X,
            'X_true': X_true,
            'A': A,
            'B': B,
            'C': C,
            'ex': self.ex_wavelens,
            'em': self.em_wavelens,
            'mask': mask,
            'gamma': gamma
        }
