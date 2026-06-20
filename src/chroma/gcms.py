import torch
import torch.nn as nn
from src.chroma.base import BaseChromaPETN

class GCMS_PETN(BaseChromaPETN):
    """
    GC-MS specific Physics-Embedded Tensor Network for Chromatographic Alignment.
    Implements masked loss, L1 spectral sparsity, and sample-specific residual shape matrices delta_B.
    """
    def __init__(self, num_samples, num_time, num_spec, num_components=3, warp_type='linear', num_segments=4,
                 lambda_c=0.01, lambda_res=1.0):
        # Initialize parent first
        super().__init__(num_samples, num_time, num_spec, num_components=num_components, warp_type=warp_type, num_segments=num_segments)
        self.lambda_c = lambda_c
        self.lambda_res = lambda_res
        
        # Residual Shape Matrix (Shape: [I, J, R])
        self.delta_B = nn.Parameter(torch.zeros(num_samples, num_time, num_components))
        self.reset_residual()

    def reset_residual(self):
        with torch.no_grad():
            self.delta_B.zero_()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'delta_B'):
            self.reset_residual()

    def forward(self, sample_idx, time_idx, spec_idx):
        """
        Coordinate-based forward pass for GC-MS.
        """
        a, b_warped, c = self._forward_raw_coo(sample_idx, time_idx, spec_idx)
        # Add residual shape offset delta_B[i, j, r]
        delta_B_val = self.delta_B[sample_idx, time_idx]  # (BatchSize, num_components)
        b_warped_final = b_warped + delta_B_val
        return torch.sum(a * b_warped_final * c, dim=1)

    def forward_grid(self):
        """
        Full grid-based forward pass for GC-MS.
        """
        A, B_warped, C = self._forward_raw_grid()
        # Add residual shape offset delta_B to aligned chromatography profile
        b_warped_final = B_warped + self.delta_B  # (num_samples, num_time, num_components)
        return torch.einsum('ir,ijr,kr->ijk', A, b_warped_final, C)

    def calculate_loss(self, X_pred, X_true):
        """
        Computes masked MSE loss, spectral L1 sparsity loss, and residual L2 loss.
        """
        # 1. Masked MSE (Ignore empty m/z channels)
        mask = X_true > 0
        if torch.sum(mask) == 0:
            mse_loss = torch.tensor(0.0, device=X_pred.device, requires_grad=True)
        else:
            mse_loss = torch.mean((X_pred[mask] - X_true[mask])**2)
        
        # 2. L1 Sparsity on Spectra (C matrix)
        l1_penalty = self.lambda_c * torch.sum(torch.abs(self.C))
        
        # 3. Heavy L2 Penalty on shape residuals
        l2_residual = self.lambda_res * torch.sum(self.delta_B**2)
        
        return mse_loss + l1_penalty + l2_residual
