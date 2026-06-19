"""
Chroma-PETN Model Architectures.
Provides separated model architectures for GC-MS and HPLC-DAD.
Implements the base class ChromaPETNBase and subclasses ChromaPETNGCMS and ChromaPETNDAD.
"""
import torch
import torch.nn as nn

class ChromaPETNBase(nn.Module):
    """
    Base class for Physics-Embedded Tensor Network for Chromatographic Alignment (Chroma-PETN).
    Models the trilinear PARAFAC core with a differentiable coordinate warping layer.
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
        
        # 1. Core Embeddings (constrained to non-negative)
        self.sample_embeddings = nn.Embedding(num_samples, num_components)
        self.time_embeddings = nn.Embedding(num_time, num_components)
        self.spec_embeddings = nn.Embedding(num_spec, num_components)
        
        # 2. Warping Parameters
        if self.warp_type == 'linear':
            # stretch (alpha) and shift (beta) per sample
            self.warp_stretch = nn.Parameter(torch.zeros(num_samples))
            self.warp_shift = nn.Parameter(torch.zeros(num_samples))
        elif self.warp_type == 'quadratic':
            # quadratic (alpha), linear (beta), shift (gamma) per sample
            self.warp_alpha = nn.Parameter(torch.zeros(num_samples))
            self.warp_beta = nn.Parameter(torch.zeros(num_samples))
            self.warp_gamma = nn.Parameter(torch.zeros(num_samples))
        elif self.warp_type == 'spline':
            # piecewise linear spline: shift at start + log-increments per segment
            self.warp_shift = nn.Parameter(torch.zeros(num_samples))
            self.warp_log_increments = nn.Parameter(torch.zeros(num_samples, num_segments))
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize embeddings with positive values
        nn.init.uniform_(self.sample_embeddings.weight, a=0.01, b=0.5)
        nn.init.uniform_(self.time_embeddings.weight, a=0.01, b=0.5)
        nn.init.uniform_(self.spec_embeddings.weight, a=0.01, b=0.5)
        
        # Initialize warp offsets to zero (initially aligned)
        if self.warp_type == 'linear':
            nn.init.constant_(self.warp_stretch, 0.0)
            nn.init.constant_(self.warp_shift, 0.0)
        elif self.warp_type == 'quadratic':
            nn.init.constant_(self.warp_alpha, 0.0)
            nn.init.constant_(self.warp_beta, 0.0)
            nn.init.constant_(self.warp_gamma, 0.0)
        elif self.warp_type == 'spline':
            nn.init.constant_(self.warp_shift, 0.0)
            nn.init.constant_(self.warp_log_increments, 0.0)

    @torch.no_grad()
    def project_constraints(self):
        # A. Apply physical non-negativity constraint via parameter clipping
        self.sample_embeddings.weight.clamp_(min=0.0)
        self.time_embeddings.weight.clamp_(min=0.0)
        self.spec_embeddings.weight.clamp_(min=0.0)
        
        # B. Center and clamp warping parameters to resolve translation/scaling degeneracies and ensure monotonicity
        if self.warp_type == 'linear':
            self.warp_stretch.data -= self.warp_stretch.data.mean()
            self.warp_shift.data -= self.warp_shift.data.mean()
            self.warp_stretch.clamp_(min=-0.2, max=0.2)
            self.warp_shift.clamp_(min=-0.15, max=0.15)
        elif self.warp_type == 'quadratic':
            self.warp_alpha.data -= self.warp_alpha.data.mean()
            self.warp_beta.data -= self.warp_beta.data.mean()
            self.warp_gamma.data -= self.warp_gamma.data.mean()
            self.warp_alpha.clamp_(min=-0.2, max=0.2)
            self.warp_beta.clamp_(min=-0.2, max=0.2)
            self.warp_gamma.clamp_(min=-0.15, max=0.15)
        elif self.warp_type == 'spline':
            self.warp_shift.data -= self.warp_shift.data.mean()
            self.warp_log_increments.data -= self.warp_log_increments.data.mean(dim=0, keepdim=True)
            self.warp_shift.clamp_(min=-0.15, max=0.15)
            self.warp_log_increments.clamp_(min=-1.0, max=1.0)

    def _forward_raw(self, sample_idx, time_idx, spec_idx):
        # 1. Lookup scores and spectra
        a = self.sample_embeddings(sample_idx) # Shape: (BatchSize, num_components)
        c = self.spec_embeddings(spec_idx)     # Shape: (BatchSize, num_components)
        
        # 2. Compute warped time coordinates for each coordinate
        t = time_idx.float() / (self.num_time - 1)  # Normalized time in [0, 1]
        
        if self.warp_type == 'linear':
            stretch_i = self.warp_stretch[sample_idx]
            shift_i = self.warp_shift[sample_idx]
            # Warp equation: t' = t - (stretch_i * t + shift_i)
            t_warped = t - (stretch_i * t + shift_i)
            
        elif self.warp_type == 'quadratic':
            alpha_i = self.warp_alpha[sample_idx]
            beta_i = self.warp_beta[sample_idx]
            gamma_i = self.warp_gamma[sample_idx]
            # Warp equation: t' = t - (alpha_i * t^2 + beta_i * t + gamma_i)
            t_warped = t - (alpha_i * (t ** 2) + beta_i * t + gamma_i)
            
        elif self.warp_type == 'spline':
            shift_i = self.warp_shift[sample_idx]
            log_inc_i = self.warp_log_increments[sample_idx] # Shape: (BatchSize, num_segments)
            
            # Compute step increments and knot positions
            inc_i = (1.0 / self.num_segments) * torch.exp(log_inc_i) # Shape: (BatchSize, num_segments)
            zeros = torch.zeros((t.shape[0], 1), device=t.device, dtype=t.dtype)
            cum_inc = torch.cumsum(torch.cat([zeros, inc_i], dim=1), dim=1) # Shape: (BatchSize, num_segments + 1)
            w = shift_i.unsqueeze(-1) + cum_inc # Shape: (BatchSize, num_segments + 1)
            
            # Find the active segment index k for each query point t
            val = t * self.num_segments
            k = torch.clamp(torch.floor(val).long(), 0, self.num_segments - 1)
            u = val - k.float()
            
            # Gather knot values for interpolation
            batch_indices = torch.arange(t.shape[0], device=t.device)
            w_k = w[batch_indices, k]
            w_kp1 = w[batch_indices, k + 1]
            
            # Warped coordinate
            t_warped = (1.0 - u) * w_k + u * w_kp1
        
        # Scale back to continuous index coordinate space [0, J-1]
        x_warped = t_warped * (self.num_time - 1)
        
        # 3. Differentiable 1D Linear Interpolation over canonical B
        B_weights = self.time_embeddings.weight # Shape: (num_time, num_components)
        
        # Clamp coordinates to safe range to prevent out-of-bounds indexing
        x_clamped = torch.clamp(x_warped, 0.0, self.num_time - 1.0 - 1e-3)
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


class ChromaPETNGCMS(ChromaPETNBase):
    """
    GC-MS specific Physics-Embedded Tensor Network for Chromatographic Alignment.
    Implements a standard non-negative raw forward pass without baseline derivatives.
    """
    def forward(self, sample_idx, time_idx, spec_idx):
        return self._forward_raw(sample_idx, time_idx, spec_idx)


class ChromaPETNDAD(ChromaPETNBase):
    """
    HPLC-DAD specific Physics-Embedded Tensor Network for Chromatographic Alignment.
    Allows continuous absorbance modeling and embeds an analytical Savitzky-Golay derivative filter in the forward pass.
    """
    def __init__(self, num_samples, num_time, num_spec, num_components=3, warp_type='linear', num_segments=4,
                 derivative_order=1, sg_window_size=11, sg_polyorder=2):
        super().__init__(num_samples, num_time, num_spec, num_components=num_components, warp_type=warp_type, num_segments=num_segments)
        self.derivative_order = derivative_order
        self.sg_window_size = sg_window_size
        self.sg_polyorder = sg_polyorder
        
        # Register Savitzky-Golay coefficients buffer
        if self.derivative_order > 0:
            if self.sg_window_size is None:
                raise ValueError("sg_window_size must be specified if derivative_order > 0")
            if self.sg_window_size % 2 == 0:
                raise ValueError("sg_window_size must be odd")
            if self.sg_polyorder >= self.sg_window_size:
                raise ValueError("sg_polyorder must be less than sg_window_size")
                
            from scipy.signal import savgol_coeffs
            coeffs = savgol_coeffs(self.sg_window_size, self.sg_polyorder, deriv=self.derivative_order)
            self.register_buffer('sg_kernel', torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1))

    def forward(self, sample_idx, time_idx, spec_idx):
        if self.derivative_order == 0:
            return self._forward_raw(sample_idx, time_idx, spec_idx)
        else:
            return self._forward_derivative(sample_idx, time_idx, spec_idx)

    def _forward_derivative(self, sample_idx, time_idx, spec_idx):
        m = self.sg_window_size // 2
        W = self.sg_window_size
        
        # Build windowed offsets: shape (W,)
        offsets = torch.arange(-m, m + 1, device=time_idx.device)
        
        # Compute windowed time indices: shape (BatchSize, W)
        time_window = time_idx.unsqueeze(-1) + offsets
        time_window_clamped = torch.clamp(time_window, 0, self.num_time - 1)
        
        # Expand sample and spec indices: shape (BatchSize, W)
        sample_window = sample_idx.unsqueeze(-1).expand(-1, W)
        spec_window = spec_idx.unsqueeze(-1).expand(-1, W)
        
        # Flatten to shape (BatchSize * W)
        sample_flat = sample_window.flatten()
        time_flat = time_window_clamped.flatten()
        spec_flat = spec_window.flatten()
        
        # Run standard raw forward pass on flat indices
        y_raw_flat = self._forward_raw(sample_flat, time_flat, spec_flat)
        
        # Reshape back to (BatchSize, 1, W)
        y_raw_window = y_raw_flat.view(-1, 1, W)
        
        # Apply 1D Savitzky-Golay convolution
        y_deriv = torch.nn.functional.conv1d(y_raw_window, self.sg_kernel, padding=0)
        
        # Squeeze to shape (BatchSize,)
        return y_deriv.view(-1)


# Legacy alias/class kept for backward compatibility with external code
class ChromaPETN(ChromaPETNBase):
    """
    General entrypoint class kept for backward compatibility.
    Dynamically routes behavior between raw and derivative forward modes.
    """
    def __init__(self, num_samples, num_time, num_spec, num_components=3, warp_type='linear', num_segments=4,
                 derivative_order=0, sg_window_size=11, sg_polyorder=2):
        super().__init__(num_samples, num_time, num_spec, num_components=num_components, warp_type=warp_type, num_segments=num_segments)
        self.derivative_order = derivative_order
        self.sg_window_size = sg_window_size
        self.sg_polyorder = sg_polyorder
        
        # Register Savitzky-Golay coefficients buffer if needed
        if self.derivative_order > 0:
            if self.sg_window_size is None:
                raise ValueError("sg_window_size must be specified if derivative_order > 0")
            if self.sg_window_size % 2 == 0:
                raise ValueError("sg_window_size must be odd")
            if self.sg_polyorder >= self.sg_window_size:
                raise ValueError("sg_polyorder must be less than sg_window_size")
                
            from scipy.signal import savgol_coeffs
            coeffs = savgol_coeffs(self.sg_window_size, self.sg_polyorder, deriv=self.derivative_order)
            self.register_buffer('sg_kernel', torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1))

    def forward(self, sample_idx, time_idx, spec_idx):
        if self.derivative_order == 0:
            return self._forward_raw(sample_idx, time_idx, spec_idx)
        else:
            return self._forward_derivative(sample_idx, time_idx, spec_idx)

    def _forward_derivative(self, sample_idx, time_idx, spec_idx):
        m = self.sg_window_size // 2
        W = self.sg_window_size
        
        offsets = torch.arange(-m, m + 1, device=time_idx.device)
        time_window = time_idx.unsqueeze(-1) + offsets
        time_window_clamped = torch.clamp(time_window, 0, self.num_time - 1)
        
        sample_window = sample_idx.unsqueeze(-1).expand(-1, W)
        spec_window = spec_idx.unsqueeze(-1).expand(-1, W)
        
        sample_flat = sample_window.flatten()
        time_flat = time_window_clamped.flatten()
        spec_flat = spec_window.flatten()
        
        y_raw_flat = self._forward_raw(sample_flat, time_flat, spec_flat)
        y_raw_window = y_raw_flat.view(-1, 1, W)
        
        y_deriv = torch.nn.functional.conv1d(y_raw_window, self.sg_kernel, padding=0)
        return y_deriv.view(-1)
