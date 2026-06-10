"""
PINN Custom Model Architecture.
Implements the trilinear core with non-negative constraints and the feedforward Dense matrix attenuation head.
"""
import torch
import torch.nn as nn

class PINNParafac(nn.Module):
    """
    Physics-Informed Neural Network (PINN) implementing the trilinear PARAFAC core.
    It embeds sample indices, excitation wavelengths, and emission wavelengths
    into positive component representations and models their trilinear interaction.
    """
    def __init__(self, num_samples, num_ex, num_em, num_components=3):
        super().__init__()
        self.num_samples = num_samples
        self.num_ex = num_ex
        self.num_em = num_em
        self.num_components = num_components
        
        # Initialize three independent embedding layers
        self.sample_embeddings = nn.Embedding(num_samples, num_components)
        self.ex_embeddings = nn.Embedding(num_ex, num_components)
        self.em_embeddings = nn.Embedding(num_em, num_components)
        
        # Initialize weights to be positive
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes all embeddings with positive values."""
        nn.init.uniform_(self.sample_embeddings.weight, a=0.1, b=1.0)
        nn.init.uniform_(self.ex_embeddings.weight, a=0.1, b=1.0)
        nn.init.uniform_(self.em_embeddings.weight, a=0.1, b=1.0)

    @torch.no_grad()
    def project_constraints(self):
        """
        Applies the physical non-negativity constraint via weight clipping/projection.
        Forces all scores and spectral profiles to be non-negative.
        """
        self.sample_embeddings.weight.clamp_(min=0.0)
        self.ex_embeddings.weight.clamp_(min=0.0)
        self.em_embeddings.weight.clamp_(min=0.0)

    def forward(self, sample_idx, ex_idx, em_idx):
        """
        Computes the forward pass.
        Args:
            sample_idx: Tensor of shape (BatchSize,) containing sample indices
            ex_idx: Tensor of shape (BatchSize,) containing excitation wavelength indices
            em_idx: Tensor of shape (BatchSize,) containing emission wavelength indices
        Returns:
            y_pred: Tensor of shape (BatchSize,) containing the predicted fluorescence intensity
        """
        # Look up embeddings: shape (BatchSize, num_components)
        a = self.sample_embeddings(sample_idx)
        b = self.ex_embeddings(ex_idx)
        c = self.em_embeddings(em_idx)
        
        # Element-wise product along components and sum over the components dimension
        # y_pred = sum_r A(i,r) * B(j,r) * C(k,r)
        y_pred = torch.sum(a * b * c, dim=1)
        return y_pred
