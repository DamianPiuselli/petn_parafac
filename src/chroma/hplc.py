import torch
import torch.nn as nn
from src.chroma.base import BaseChromaPETN

class HPLC_PETN(BaseChromaPETN):
    """
    HPLC-DAD specific Physics-Embedded Tensor Network for Chromatographic Alignment.
    Supports continuous absorbance modeling, Savitzky-Golay derivatives, and baseline offset subtraction.
    """
    def __init__(self, num_samples, num_time, num_spec, num_components=3, warp_type='linear', num_segments=4,
                 derivative_order=0, sg_window_size=11, sg_polyorder=2, sample_specific_baseline=False):
        super().__init__(num_samples, num_time, num_spec, num_components=num_components, warp_type=warp_type, num_segments=num_segments)
        self.derivative_order = derivative_order
        self.sg_window_size = sg_window_size
        self.sg_polyorder = sg_polyorder
        
        # 1. Background/Baseline Absorption (sample-specific, rank-1 time-varying polynomial representation)
        self.sample_specific_baseline = sample_specific_baseline
        if self.sample_specific_baseline:
            self.solvent_spectrum = nn.Parameter(torch.ones(num_spec) * 0.05)
            self.baseline_offset = nn.Parameter(torch.zeros(num_samples))
            self.baseline_slope = nn.Parameter(torch.zeros(num_samples))
            self.baseline_quadratic = nn.Parameter(torch.zeros(num_samples))
        else:
            self.baseline_offset = nn.Parameter(torch.zeros(num_spec))
            self.baseline_slope = nn.Parameter(torch.zeros(num_spec))
            self.baseline_quadratic = nn.Parameter(torch.zeros(num_spec))
            
        # 2. Register Savitzky-Golay coefficients buffer if needed
        if self.derivative_order > 0:
            if self.sg_window_size is None:
                raise ValueError("sg_window_size must be specified if derivative_order > 0")
            if self.sg_window_size % 2 == 0:
                raise ValueError("sg_window_size must be odd")
            if self.sg_polyorder >= self.sg_window_size:
                raise ValueError("sg_polyorder must be less than sg_window_size")
                
            from scipy.signal import savgol_coeffs
            coeffs = savgol_coeffs(self.sg_window_size, self.sg_polyorder, deriv=self.derivative_order)
            # Flip the coefficients along the filter length axis to convert cross-correlation (conv1d) to convolution
            coeffs = coeffs[::-1].copy()
            self.register_buffer('sg_kernel', torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1))

    @torch.no_grad()
    def project_constraints(self):
        """
        Applies Base class projections and clamps solvent spectrum to non-negative.
        """
        super().project_constraints()
        if self.sample_specific_baseline:
            self.solvent_spectrum.clamp_(min=0.0)

    def forward(self, sample_idx, time_idx, spec_idx):
        """
        Coordinate-based forward pass for HPLC-DAD.
        """
        if self.derivative_order == 0:
            a, b_warped, c = self._forward_raw_coo(sample_idx, time_idx, spec_idx)
            y_pred = torch.sum(a * b_warped * c, dim=1)
            
            # Add baseline offset and drift
            t = time_idx.float() / (self.num_time - 1)
            if self.sample_specific_baseline:
                poly = (self.baseline_offset[sample_idx] + 
                        self.baseline_slope[sample_idx] * t + 
                        self.baseline_quadratic[sample_idx] * (t ** 2))
                baseline_val = poly * self.solvent_spectrum[spec_idx]
            else:
                baseline_val = (self.baseline_offset[spec_idx] + 
                                self.baseline_slope[spec_idx] * t + 
                                self.baseline_quadratic[spec_idx] * (t ** 2))
            return y_pred + baseline_val
        else:
            # Savitzky-Golay windowed coordinate-based evaluation
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
            
            a, b_warped, c = self._forward_raw_coo(sample_flat, time_flat, spec_flat)
            y_raw_flat = torch.sum(a * b_warped * c, dim=1)
            
            # Add baseline offset and drift
            t_flat = time_flat.float() / (self.num_time - 1)
            if self.sample_specific_baseline:
                poly_flat = (self.baseline_offset[sample_flat] + 
                             self.baseline_slope[sample_flat] * t_flat + 
                             self.baseline_quadratic[sample_flat] * (t_flat ** 2))
                baseline_val_flat = poly_flat * self.solvent_spectrum[spec_flat]
            else:
                baseline_val_flat = (self.baseline_offset[spec_flat] + 
                                     self.baseline_slope[spec_flat] * t_flat + 
                                     self.baseline_quadratic[spec_flat] * (t_flat ** 2))
            y_raw_flat = y_raw_flat + baseline_val_flat
                
            y_raw_window = y_raw_flat.view(-1, 1, W)
            y_deriv = torch.nn.functional.conv1d(y_raw_window, self.sg_kernel, padding=0)
            return y_deriv.view(-1)

    def forward_grid(self):
        """
        Full grid-based forward pass for HPLC-DAD.
        """
        A, B_warped, C = self._forward_raw_grid()
        Y_pred = torch.einsum('ir,ijr,kr->ijk', A, B_warped, C)
        
        # Add sample-specific, time-varying polynomial baseline
        t_grid = torch.linspace(0.0, 1.0, self.num_time, device=Y_pred.device).view(1, -1)
        
        if self.sample_specific_baseline:
            poly = (self.baseline_offset.unsqueeze(1) + 
                    self.baseline_slope.unsqueeze(1) * t_grid + 
                    self.baseline_quadratic.unsqueeze(1) * (t_grid ** 2))
            baseline = torch.einsum('ij,k->ijk', poly, self.solvent_spectrum)
        else:
            t_grid_3d = t_grid.unsqueeze(-1)
            baseline = (self.baseline_offset.view(1, 1, -1) + 
                        self.baseline_slope.view(1, 1, -1) * t_grid_3d + 
                        self.baseline_quadratic.view(1, 1, -1) * (t_grid_3d ** 2))
            
        Y_pred = Y_pred + baseline
            
        # Apply Savitzky-Golay derivative filtering along the time axis (dim 1)
        if self.derivative_order > 0:
            m = self.sg_window_size // 2
            y_raw_window = Y_pred.transpose(1, 2).reshape(self.num_samples * self.num_spec, 1, self.num_time)
            y_padded = torch.nn.functional.pad(y_raw_window, (m, m), mode='replicate')
            y_deriv = torch.nn.functional.conv1d(y_padded, self.sg_kernel, padding=0)
            Y_pred = y_deriv.view(self.num_samples, self.num_spec, self.num_time).transpose(1, 2)
            
        return Y_pred

    def calculate_loss(self, X_pred, X_true):
        """
        Dense MSE loss. No masking required.
        """
        return torch.nn.functional.mse_loss(X_pred, X_true)
