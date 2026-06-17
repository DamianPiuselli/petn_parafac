"""
Chroma-PETN Model Architecture.
Implements the Physics-Embedded Tensor Network for Chromatographic Alignment.
Contains the trilinear core with non-negative constraints and a differentiable 1D warping layer.
"""
import torch
import torch.nn as nn

class ChromaPETN(nn.Module):
    """
    Physics-Embedded Tensor Network for Chromatographic Alignment (Chroma-PETN).
    Models the trilinear PARAFAC core with a differentiable coordinate warping layer.
    """
    def __init__(self, num_samples, num_time, num_spec, num_components=3):
        super().__init__()
        self.num_samples = num_samples
        self.num_time = num_time
        self.num_spec = num_spec
        self.num_components = num_components
        
        # 1. Core Embeddings (constrained to non-negative)
        self.sample_embeddings = nn.Embedding(num_samples, num_components)
        self.time_embeddings = nn.Embedding(num_time, num_components)
        self.spec_embeddings = nn.Embedding(num_spec, num_components)
        
        # 2. Warping Parameters: stretch (alpha) and shift (beta) per sample
        self.warp_stretch = nn.Parameter(torch.zeros(num_samples))
        self.warp_shift = nn.Parameter(torch.zeros(num_samples))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize embeddings with positive values
        nn.init.uniform_(self.sample_embeddings.weight, a=0.2, b=1.0)
        nn.init.uniform_(self.time_embeddings.weight, a=0.2, b=1.0)
        nn.init.uniform_(self.spec_embeddings.weight, a=0.2, b=1.0)
        
        # Initialize warp offsets to zero (initially aligned)
        nn.init.constant_(self.warp_stretch, 0.0)
        nn.init.constant_(self.warp_shift, 0.0)

    @torch.no_grad()
    def project_constraints(self):
        # A. Apply physical non-negativity constraint via parameter clipping
        self.sample_embeddings.weight.clamp_(min=0.0)
        self.time_embeddings.weight.clamp_(min=0.0)
        self.spec_embeddings.weight.clamp_(min=0.0)
        
        # B. Center shifts and stretches to resolve translation/scaling degeneracies
        self.warp_stretch.data -= self.warp_stretch.data.mean()
        self.warp_shift.data -= self.warp_shift.data.mean()
        
        # C. Clip warping factors to physically plausible ranges to ensure monotonicity
        self.warp_stretch.clamp_(min=-0.2, max=0.2)
        self.warp_shift.clamp_(min=-0.15, max=0.15)

    def forward(self, sample_idx, time_idx, spec_idx):
        """
        Args:
            sample_idx: Tensor of shape (BatchSize,) containing sample indices
            time_idx: Tensor of shape (BatchSize,) containing time channel indices
            spec_idx: Tensor of shape (BatchSize,) containing spectral channel indices
        Returns:
            y_pred: Predicted observed intensities of shape (BatchSize,)
        """
        # 1. Lookup scores and spectra
        a = self.sample_embeddings(sample_idx) # Shape: (BatchSize, num_components)
        c = self.spec_embeddings(spec_idx)     # Shape: (BatchSize, num_components)
        
        # 2. Compute warped time coordinates for each coordinate
        t = time_idx.float() / (self.num_time - 1)  # Normalized time in [0, 1]
        
        stretch_i = self.warp_stretch[sample_idx]
        shift_i = self.warp_shift[sample_idx]
        
        # Warp equation: t' = t - (stretch_i * t + shift_i)
        t_warped = t - (stretch_i * t + shift_i)
        
        # Scale back to continuous index coordinate space [0, J-1]
        x_warped = t_warped * (self.num_time - 1)
        
        # 3. Differentiable 1D Linear Interpolation over canonical B
        B_weights = self.time_embeddings.weight # Shape: (num_time, num_components)
        
        # Clamp coordinates to safe range to prevent out-of-bounds indexing
        x_clamped = torch.clamp(x_warped, 0.0, self.num_time - 1.0 - 1e-5)
        x_0 = torch.floor(x_clamped).long()
        x_1 = x_0 + 1
        
        # Calculate interpolation weights
        w = (x_clamped - x_0.float()).unsqueeze(-1) # Shape: (BatchSize, 1)
        
        # Lookup neighboring points
        val_0 = B_weights[x_0] # Shape: (BatchSize, num_components)
        val_1 = B_weights[x_1] # Shape: (BatchSize, num_components)
        
        # Interpolated profile values
        b_warped = (1.0 - w) * val_0 + w * val_1 # Shape: (BatchSize, num_components)
        
        # 4. Reconstruct intensity: y_pred = sum_r a_ir * b_warped_r * c_kr
        y_pred = torch.sum(a * b_warped * c, dim=1)
        return y_pred
