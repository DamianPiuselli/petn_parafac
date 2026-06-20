import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseChromaPETN(nn.Module, ABC):
    """
    Base class for Physics-Embedded Tensor Network for Chromatographic Alignment (Chroma-PETN).
    Handles core parameter representations (A, B, C), component-specific warping (alpha, beta),
    and area-preserving differentiable interpolation.
    """
    def __init__(self, num_samples, num_time, num_spec, num_components=3, warp_type='linear', num_segments=4):
        super().__init__()
        self.num_samples = num_samples
        self.num_time = num_time
        self.num_spec = num_spec
        self.num_components = num_components
        self.warp_type = warp_type
        self.num_segments = num_segments
        
        # Validate warp type
        if warp_type not in ['linear', 'quadratic', 'spline']:
            raise ValueError(f"Unknown warp_type: {warp_type}. Must be 'linear', 'quadratic', or 'spline'.")
        
        # 1. Core Parameters (constrained to non-negative)
        self.A = nn.Parameter(torch.zeros(num_samples, num_components))  # Scores
        self.B = nn.Parameter(torch.zeros(num_time, num_components))     # Canonical Profiles
        self.C = nn.Parameter(torch.zeros(num_spec, num_components))     # Spectra
        
        # 2. Component-Specific Warping Parameters (Shape: [I, R])
        if self.warp_type == 'linear':
            # alpha (stretch) and beta (shift) per sample and component
            self.alpha = nn.Parameter(torch.zeros(num_samples, num_components))
            self.beta = nn.Parameter(torch.zeros(num_samples, num_components))
        elif self.warp_type == 'quadratic':
            # alpha (quadratic), beta (linear), gamma (shift) per sample and component
            self.alpha = nn.Parameter(torch.zeros(num_samples, num_components))
            self.beta = nn.Parameter(torch.zeros(num_samples, num_components))
            self.gamma = nn.Parameter(torch.zeros(num_samples, num_components))
        elif self.warp_type == 'spline':
            # shift (beta) per sample and component
            self.beta = nn.Parameter(torch.zeros(num_samples, num_components))
            # log-increments per sample, segment, and component
            self.log_increments = nn.Parameter(torch.zeros(num_samples, num_segments, num_components))
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize core parameters with positive values
        nn.init.uniform_(self.A, a=0.01, b=0.5)
        nn.init.uniform_(self.B, a=0.01, b=0.5)
        nn.init.uniform_(self.C, a=0.01, b=0.5)
        
        # Initialize warp offsets to zero (initially aligned)
        if self.warp_type == 'linear':
            nn.init.constant_(self.alpha, 0.0)
            nn.init.constant_(self.beta, 0.0)
        elif self.warp_type == 'quadratic':
            nn.init.constant_(self.alpha, 0.0)
            nn.init.constant_(self.beta, 0.0)
            nn.init.constant_(self.gamma, 0.0)
        elif self.warp_type == 'spline':
            nn.init.constant_(self.beta, 0.0)
            nn.init.constant_(self.log_increments, 0.0)

    @torch.no_grad()
    def project_constraints(self):
        """
        Enforces strict physical constraints: non-negativity on A, B, C,
        monotonicity clamping on alpha, and zero-sum mean centering on beta.
        """
        # A. Apply physical non-negativity constraint
        self.A.clamp_(min=0.0)
        self.B.clamp_(min=0.0)
        self.C.clamp_(min=0.0)
        
        
        # B. Center and clamp warping parameters to resolve translation/scaling degeneracies
        if self.warp_type == 'linear':
            self.alpha.data -= self.alpha.data.mean(dim=0, keepdim=True)
            self.beta.data -= self.beta.data.mean(dim=0, keepdim=True)
            self.alpha.clamp_(min=-0.2, max=0.2)
            self.beta.clamp_(min=-0.15, max=0.15)
        elif self.warp_type == 'quadratic':
            self.alpha.data -= self.alpha.data.mean(dim=0, keepdim=True)
            self.beta.data -= self.beta.data.mean(dim=0, keepdim=True)
            self.gamma.data -= self.gamma.data.mean(dim=0, keepdim=True)
            self.alpha.clamp_(min=-0.2, max=0.2)
            self.beta.clamp_(min=-0.2, max=0.2)
            self.gamma.clamp_(min=-0.15, max=0.15)
        elif self.warp_type == 'spline':
            self.beta.data -= self.beta.data.mean(dim=0, keepdim=True)
            self.log_increments.data -= self.log_increments.data.mean(dim=0, keepdim=True)
            self.beta.clamp_(min=-0.15, max=0.15)
            self.log_increments.clamp_(min=-1.0, max=1.0)

    def _forward_raw_coo(self, sample_idx, time_idx, spec_idx):
        """
        Calculates score lookup, component-specific time warping, area-preserving
        1D interpolation, and spectral lookup for coordinate list inputs.
        """
        # 1. Lookup scores and spectra
        a = self.A[sample_idx]  # (BatchSize, num_components)
        c = self.C[spec_idx]    # (BatchSize, num_components)
        
        # 2. Compute warped time coordinates for each coordinate and component
        t = time_idx.float() / (self.num_time - 1)  # (BatchSize,)
        t = t.unsqueeze(-1)  # Broadcast over components: (BatchSize, 1)
        
        if self.warp_type == 'linear':
            alpha_i = self.alpha[sample_idx]  # (BatchSize, num_components)
            beta_i = self.beta[sample_idx]    # (BatchSize, num_components)
            t_warped = t - (alpha_i * t + beta_i)  # (BatchSize, num_components)
            jacobian = 1.0 - alpha_i
            
        elif self.warp_type == 'quadratic':
            alpha_i = self.alpha[sample_idx]
            beta_i = self.beta[sample_idx]
            gamma_i = self.gamma[sample_idx]
            t_warped = t - (alpha_i * (t ** 2) + beta_i * t + gamma_i)
            jacobian = 1.0 - (2.0 * alpha_i * t + beta_i)
            
        elif self.warp_type == 'spline':
            beta_i = self.beta[sample_idx]  # (BatchSize, num_components)
            log_inc_i = self.log_increments[sample_idx]  # (BatchSize, num_segments, num_components)
            
            # Compute step increments and knot positions
            inc_i = (1.0 / self.num_segments) * torch.exp(log_inc_i)  # (BatchSize, num_segments, num_components)
            zeros = torch.zeros((t.shape[0], 1, self.num_components), device=t.device, dtype=t.dtype)
            cum_inc = torch.cumsum(torch.cat([zeros, inc_i], dim=1), dim=1)  # (BatchSize, num_segments + 1, num_components)
            w = beta_i.unsqueeze(1) + cum_inc  # (BatchSize, num_segments + 1, num_components)
            
            # Find the active segment index k for each query point t
            val = t * self.num_segments
            k = torch.clamp(torch.floor(val).long(), 0, self.num_segments - 1)
            u = val - k.float()
            
            # Gather knot values for interpolation
            k_expanded = k.unsqueeze(-1).expand(-1, -1, self.num_components)
            w_k = torch.gather(w, 1, k_expanded).squeeze(1)  # (BatchSize, num_components)
            w_kp1 = torch.gather(w, 1, k_expanded + 1).squeeze(1)  # (BatchSize, num_components)
            
            # Warped coordinate
            t_warped = (1.0 - u) * w_k + u * w_kp1
            
            # Jacobian slope in the segment
            jacobian = (w_kp1 - w_k) * self.num_segments
            
        # Scale back to continuous index coordinate space [0, J-1]
        x_warped = t_warped * (self.num_time - 1)
        
        # 3. Differentiable 1D Linear Interpolation over B
        # Clamp coordinates to safe range
        x_clamped = torch.clamp(x_warped, 0.0, self.num_time - 1.0 - 1e-3)
        x_0 = torch.floor(x_clamped).long()  # (BatchSize, num_components)
        x_1 = x_0 + 1
        
        # Calculate interpolation weights
        w_interp = x_clamped - x_0.float()  # (BatchSize, num_components)
        
        # Gather along dim 0 of B (time index axis)
        val_0 = torch.gather(self.B, 0, x_0)  # (BatchSize, num_components)
        val_1 = torch.gather(self.B, 0, x_1)  # (BatchSize, num_components)
        
        # Interpolated profile values
        b_interpolated = (1.0 - w_interp) * val_0 + w_interp * val_1  # (BatchSize, num_components)
        
        # Apply Jacobian multiplier to preserve peak area when stretched
        b_warped = b_interpolated * jacobian  # (BatchSize, num_components)
        
        return a, b_warped, c

    def _forward_raw_grid(self):
        """
        Performs continuous time warping and interpolation for full grid reconstruction.
        Supports Component-Specific (CS) warping.
        """
        device = self.B.device
        t_grid = torch.linspace(0.0, 1.0, self.num_time, device=device)
        
        A = self.A  # (num_samples, num_components)
        C = self.C  # (num_spec, num_components)
        
        # 1. Warp time coordinates per sample and component
        if self.warp_type == 'linear':
            # t_grid: (num_time,) -> (1, 1, num_time)
            # alpha: (num_samples, num_components) -> (num_samples, num_components, 1)
            t_warped = t_grid.view(1, 1, -1) - (
                self.alpha.unsqueeze(-1) * t_grid.view(1, 1, -1) + self.beta.unsqueeze(-1)
            )  # (num_samples, num_components, num_time)
            jacobian = 1.0 - self.alpha.unsqueeze(-1)  # (num_samples, num_components, 1)
            
        elif self.warp_type == 'quadratic':
            t_warped = t_grid.view(1, 1, -1) - (
                self.alpha.unsqueeze(-1) * (t_grid.view(1, 1, -1) ** 2) +
                self.beta.unsqueeze(-1) * t_grid.view(1, 1, -1) +
                self.gamma.unsqueeze(-1)
            )  # (num_samples, num_components, num_time)
            jacobian = 1.0 - (
                2.0 * self.alpha.unsqueeze(-1) * t_grid.view(1, 1, -1) + self.beta.unsqueeze(-1)
            )
            
        elif self.warp_type == 'spline':
            inc = (1.0 / self.num_segments) * torch.exp(self.log_increments)  # (num_samples, num_segments, num_components)
            inc = inc.permute(0, 2, 1)  # (num_samples, num_components, num_segments)
            zeros = torch.zeros((self.num_samples, self.num_components, 1), device=device, dtype=inc.dtype)
            cum_inc = torch.cumsum(torch.cat([zeros, inc], dim=2), dim=2)  # (num_samples, num_components, num_segments + 1)
            w = self.beta.unsqueeze(-1) + cum_inc  # (num_samples, num_components, num_segments + 1)
            
            val = t_grid * self.num_segments  # (num_time,)
            k = torch.clamp(torch.floor(val).long(), 0, self.num_segments - 1)  # (num_time,)
            u = val - k.float()  # (num_time,)
            
            w_k = w[:, :, k]  # (num_samples, num_components, num_time)
            w_kp1 = w[:, :, k + 1]  # (num_samples, num_components, num_time)
            t_warped = (1.0 - u.view(1, 1, -1)) * w_k + u.view(1, 1, -1) * w_kp1  # (num_samples, num_components, num_time)
            
            jacobian = (w_kp1 - w_k) * self.num_segments  # (num_samples, num_components, num_time)
            
        # Scale back to continuous index coordinate space [0, J-1]
        x_warped = t_warped * (self.num_time - 1)
        
        # 2. Differentiable 1D Linear Interpolation over B
        B_expanded = self.B.t().unsqueeze(0).expand(self.num_samples, -1, -1)  # (num_samples, num_components, num_time)
        
        x_clamped = torch.clamp(x_warped, 0.0, self.num_time - 1.0 - 1e-3)
        x_0 = torch.floor(x_clamped).long()  # (num_samples, num_components, num_time)
        x_1 = x_0 + 1
        
        w_interp = x_clamped - x_0.float()  # (num_samples, num_components, num_time)
        
        # Gather neighboring values along dim 2 of B_expanded
        val_0 = torch.gather(B_expanded, 2, x_0)  # (num_samples, num_components, num_time)
        val_1 = torch.gather(B_expanded, 2, x_1)  # (num_samples, num_components, num_time)
        
        b_interpolated = (1.0 - w_interp) * val_0 + w_interp * val_1  # (num_samples, num_components, num_time)
        
        # Apply Jacobian multiplier and permute to (num_samples, num_time, num_components)
        b_warped = (b_interpolated * jacobian).permute(0, 2, 1)
        
        return A, b_warped, C

    @abstractmethod
    def calculate_loss(self, X_pred, X_true):
        """
        Calculates subclass-specific losses. Must be implemented by HPLC_PETN and GCMS_PETN.
        """
        pass
