"""
PETN Custom Model Architecture.
Implements the trilinear core with non-negative constraints and the feedforward Dense matrix attenuation head.
"""
import torch
import torch.nn as nn

class PETNParafac(nn.Module):
    """
    Physics-Embedded Tensor Network (PETN) implementing the trilinear PARAFAC core.
    It embeds sample indices, excitation wavelengths, and emission wavelengths
    into positive component representations and models their trilinear interaction.
    """
    def __init__(self, num_samples, num_ex, num_em, ex_wavelens, em_wavelens,
                 ex_bg=None, em_bg=None, num_components=3):
        super().__init__()
        self.num_samples = num_samples
        self.num_ex = num_ex
        self.num_em = num_em
        self.num_components = num_components
        
        # 1. Trilinear core embedding layers
        self.sample_embeddings = nn.Embedding(num_samples, num_components)
        self.ex_embeddings = nn.Embedding(num_ex, num_components)
        self.em_embeddings = nn.Embedding(num_em, num_components)
        
        # 2. IFE Molar absorptivity scaling parameters (non-negative)
        self.alpha = nn.Parameter(torch.ones(num_components) * 0.1)
        
        # 3. Register physical background CDOM profiles as fixed buffers
        if ex_bg is None:
            ex_bg = torch.zeros(num_ex)
        else:
            ex_bg = torch.as_tensor(ex_bg, dtype=torch.float32)
            
        if em_bg is None:
            em_bg = torch.zeros(num_em)
        else:
            em_bg = torch.as_tensor(em_bg, dtype=torch.float32)
            
        self.register_buffer('ex_bg', ex_bg)
        self.register_buffer('em_bg', em_bg)
        
        # Register physical wavelengths
        self.register_buffer('ex_wavelens', torch.tensor(ex_wavelens, dtype=torch.float32))
        self.register_buffer('em_wavelens', torch.tensor(em_wavelens, dtype=torch.float32))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes all embeddings with positive values."""
        # Core embeddings initialization
        nn.init.uniform_(self.sample_embeddings.weight, a=0.1, b=1.0)
        nn.init.uniform_(self.ex_embeddings.weight, a=0.1, b=1.0)
        nn.init.uniform_(self.em_embeddings.weight, a=0.1, b=1.0)
        
        # Absorptivity scaling initialization
        nn.init.uniform_(self.alpha.data, a=0.01, b=0.20)

    @torch.no_grad()
    def project_constraints(self):
        """
        Applies the physical non-negativity constraint via weight clipping/projection.
        Forces concentrations, spectral profiles, and absorptivity scaling to be non-negative.
        """
        self.sample_embeddings.weight.clamp_(min=0.0)
        self.ex_embeddings.weight.clamp_(min=0.0)
        self.em_embeddings.weight.clamp_(min=0.0)
        self.alpha.clamp_(min=0.0)

    @torch.no_grad()
    def get_learned_absorptivities(self):
        """
        Extracts the resolved excitation and emission molar absorptivities.
        
        Returns:
            E: 2D numpy array of shape (num_ex, num_components)
            M: 2D numpy array of shape (num_em, num_components)
        """
        import numpy as np
        alpha = self.alpha.cpu().numpy()
        B = self.ex_embeddings.weight.cpu().numpy()
        E = alpha * B
        M = np.zeros((self.num_em, self.num_components))
        return E, M

    def forward(self, sample_idx, ex_idx, em_idx):
        """
        Computes the forward pass with sample-dependent IFE attenuation.
        Args:
            sample_idx: Tensor of shape (BatchSize,) containing sample indices
            ex_idx: Tensor of shape (BatchSize,) containing excitation wavelength indices
            em_idx: Tensor of shape (BatchSize,) containing emission wavelength indices
        Returns:
            y_pred: Tensor of shape (BatchSize,) containing the predicted observed fluorescence intensity
        """
        # 1. Trilinear core lookup: shape (BatchSize, num_components)
        a = self.sample_embeddings(sample_idx)
        b = self.ex_embeddings(ex_idx)
        c = self.em_embeddings(em_idx)
        
        # Calculate unattenuated true intensity: I_true = sum_r A(i,r) * B(j,r) * C(k,r)
        I_true = torch.sum(a * b * c, dim=1)
        
        # 2. Calculate total absorbances: Abs = sum_r a_ir * (alpha_r * b_jr) + bg
        Abs_ex = torch.sum(a * (self.alpha * b), dim=1) + self.ex_bg[ex_idx]
        Abs_em = self.em_bg[em_idx]
        
        # 3. Calculate attenuation: gamma = 10^(-(Abs_ex + Abs_em))
        gamma = torch.pow(10.0, -(Abs_ex + Abs_em))
        
        # Combine: I_obs = I_true * gamma
        y_pred = I_true * gamma
        return y_pred
