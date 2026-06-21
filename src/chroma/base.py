import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseChromaPETN(nn.Module, ABC):
    """
    Base class for Physics-Embedded Tensor Network for Chromatographic Alignment (Chroma-PETN).
    Handles core parameter representations (A, B, C), component-specific warping (alpha, beta),
    and area-preserving differentiable interpolation.
    """
    def __init__(self, num_samples, num_time, num_spec, num_components=3, warp_type='linear', num_segments=4, component_specific_warp=False):
        super().__init__()
        self.num_samples = num_samples
        self.num_time = num_time
        self.num_spec = num_spec
        self.num_components = num_components
        self.warp_type = warp_type
        self.num_segments = num_segments
        self.component_specific_warp = component_specific_warp
        
        # Validate warp type
        if warp_type not in ['linear', 'quadratic', 'spline']:
            raise ValueError(f"Unknown warp_type: {warp_type}. Must be 'linear', 'quadratic', or 'spline'.")
        
        # 1. Core Parameters (constrained to non-negative)
        self.A = nn.Parameter(torch.zeros(num_samples, num_components))  # Scores
        self.B = nn.Parameter(torch.zeros(num_time, num_components))     # Canonical Profiles
        self.C = nn.Parameter(torch.zeros(num_spec, num_components))     # Spectra
        
        # 2. Warping Parameters (Shape: [I, R] if component-specific, else [I, 1])
        dim2 = num_components if self.component_specific_warp else 1
        if self.warp_type == 'linear':
            # alpha (stretch) and beta (shift) per sample and component
            self.alpha = nn.Parameter(torch.zeros(num_samples, dim2))
            self.beta = nn.Parameter(torch.zeros(num_samples, dim2))
        elif self.warp_type == 'quadratic':
            # alpha (quadratic), beta (linear), gamma (shift) per sample and component
            self.alpha = nn.Parameter(torch.zeros(num_samples, dim2))
            self.beta = nn.Parameter(torch.zeros(num_samples, dim2))
            self.gamma = nn.Parameter(torch.zeros(num_samples, dim2))
        elif self.warp_type == 'spline':
            # shift (beta) per sample and component
            self.beta = nn.Parameter(torch.zeros(num_samples, dim2))
            # log-increments per sample, segment, and component
            self.log_increments = nn.Parameter(torch.zeros(num_samples, num_segments, dim2))
            
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
    def init_from_svd(self, X_tensor, init_warp=True):
        """
        Warm-starts the embedding tables (A, B, C) using unfolded Truncated SVD.
        This gives the network the "average" peak shapes and spectra on Epoch 0,
        meaning the warping head only has to make micro-adjustments.
        
        Parameters
        ----------
        X_tensor : torch.Tensor or numpy.ndarray
            The 3D chromatographic input tensor of shape (I, J, K).
        init_warp : bool, optional
            Whether to also warm-start the warping parameters (beta/gamma) 
            using cross-correlation. Default is True.
        """
        # Ensure input is a PyTorch tensor
        if not isinstance(X_tensor, torch.Tensor):
            X_tensor = torch.tensor(X_tensor, dtype=torch.float32)
            
        # Run SVD on CPU for platform compatibility and numerical stability
        X_cpu = X_tensor.detach().cpu().to(dtype=torch.float32)
        I, J, K = X_cpu.shape
        R = self.num_components
        
        if torch.isnan(X_cpu).any() or torch.isinf(X_cpu).any():
            import warnings
            warnings.warn("X_tensor contains NaN or Inf values. SVD initialization aborted, falling back to random init.")
            self.reset_parameters()
            return
            
        try:
            # Reset all parameters to defaults (aligns warp parameters and zeroes GC-MS residuals)
            self.reset_parameters()
            
            # Mode 1 (Samples): self.A -> shape (I, R)
            X_1 = X_cpu.reshape(I, -1)
            U_1, S_1, _ = torch.linalg.svd(X_1, full_matrices=False)
            R_1 = min(U_1.shape[1], R)
            scale_1 = S_1[:R_1] ** (1.0 / 3.0)
            A_init = torch.zeros((I, R), device=X_cpu.device, dtype=X_cpu.dtype)
            nn.init.uniform_(A_init, a=0.01, b=0.1)
            A_init[:, :R_1] = torch.abs(U_1[:, :R_1]) * scale_1.unsqueeze(0) + 1e-4
            
            # Mode 2 (Time): self.B -> shape (J, R)
            X_2 = X_cpu.transpose(0, 1).reshape(J, -1)
            U_2, S_2, _ = torch.linalg.svd(X_2, full_matrices=False)
            R_2 = min(U_2.shape[1], R)
            scale_2 = S_2[:R_2] ** (1.0 / 3.0)
            B_init = torch.zeros((J, R), device=X_cpu.device, dtype=X_cpu.dtype)
            nn.init.uniform_(B_init, a=0.01, b=0.1)
            B_init[:, :R_2] = torch.abs(U_2[:, :R_2]) * scale_2.unsqueeze(0) + 1e-4
            
            # Mode 3 (Spectra): self.C -> shape (K, R)
            X_3 = X_cpu.transpose(0, 2).reshape(K, -1)
            U_3, S_3, _ = torch.linalg.svd(X_3, full_matrices=False)
            R_3 = min(U_3.shape[1], R)
            scale_3 = S_3[:R_3] ** (1.0 / 3.0)
            C_init = torch.zeros((K, R), device=X_cpu.device, dtype=X_cpu.dtype)
            nn.init.uniform_(C_init, a=0.01, b=0.1)
            C_init[:, :R_3] = torch.abs(U_3[:, :R_3]) * scale_3.unsqueeze(0) + 1e-4
            
            # Copy initialized values to parameters on native device
            self.A.copy_(A_init.to(device=self.A.device, dtype=self.A.dtype))
            self.B.copy_(B_init.to(device=self.B.device, dtype=self.B.dtype))
            self.C.copy_(C_init.to(device=self.C.device, dtype=self.C.dtype))
            
            # Project constraints
            self.project_constraints()
            
            # Warp warm start via cross-correlation
            if init_warp:
                self.init_warp_from_cross_correlation(X_tensor)
            
        except Exception as e:
            import warnings
            warnings.warn(f"SVD warm-start initialization failed: {e}. Falling back to default random initialization.")
            self.reset_parameters()

    @torch.no_grad()
    def init_warp_from_cross_correlation(self, X_tensor):
        """
        Initializes the shift parameter (beta/gamma) using 1D cross-correlation of TICs.
        This provides a coarse initial alignment to place peaks within the gradient "basin of attraction",
        solving the problem of non-overlapping peaks.
        """
        if not isinstance(X_tensor, torch.Tensor):
            X_tensor = torch.tensor(X_tensor, dtype=torch.float32)
            
        X_cpu = X_tensor.detach().cpu().to(dtype=torch.float32)
        I, J, K = X_cpu.shape
        
        # 1. Compute TICs for each sample (sum over spectral dimension K)
        tics = torch.sum(X_cpu, dim=2)  # Shape: (I, J)
        
        # 2. Compute the mean TIC across all samples as the reference template
        mean_tic = torch.mean(tics, dim=0)  # Shape: (J,)
        
        # Normalize and flip the mean TIC for conv1d
        mean_tic_zero_mean = mean_tic - torch.mean(mean_tic)
        mean_tic_std = torch.std(mean_tic_zero_mean) + 1e-8
        mean_tic_norm = mean_tic_zero_mean / (mean_tic_std * J)
        mean_tic_norm_flipped = torch.flip(mean_tic_norm, dims=[0])
        
        # 3. Calculate cross-correlation shifts
        shifts = torch.zeros(I, device=X_cpu.device)
        for i in range(I):
            tic_i = tics[i]
            tic_i_zero_mean = tic_i - torch.mean(tic_i)
            tic_i_std = torch.std(tic_i_zero_mean) + 1e-8
            tic_i_norm = tic_i_zero_mean / tic_i_std
            
            # Cross-correlation via conv1d
            corr = torch.conv1d(
                tic_i_norm.view(1, 1, -1),
                mean_tic_norm_flipped.view(1, 1, -1),
                padding=J - 1
            ).view(-1)
            
            lags = torch.arange(2 * J - 1) - (J - 1)
            best_lag_idx = torch.argmax(corr)
            best_lag = lags[best_lag_idx].float()
            
            # Convert lag in time steps to normalized shift coordinate
            shifts[i] = best_lag / (J - 1)
            
        # 4. Mean-center shifts to satisfy translation constraints
        shifts = shifts - torch.mean(shifts)
        
        # 5. Initialize the shift parameter based on warp type and component_specific_warp
        dim2 = self.num_components if self.component_specific_warp else 1
        shifts_expanded = shifts.unsqueeze(1).expand(-1, dim2).to(device=self.A.device, dtype=self.A.dtype)
        
        if self.warp_type in ['linear', 'spline']:
            self.beta.copy_(shifts_expanded)
        elif self.warp_type == 'quadratic':
            self.gamma.copy_(shifts_expanded)
            
        self.project_constraints()
        print(f"Initialized warping shifts via cross-correlation: {shifts.cpu().numpy()}")

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
            alpha_i = self.alpha[sample_idx]
            beta_i = self.beta[sample_idx]
            if not self.component_specific_warp:
                alpha_i = alpha_i.expand(-1, self.num_components)
                beta_i = beta_i.expand(-1, self.num_components)
            t_warped = t - (alpha_i * t + beta_i)  # (BatchSize, num_components)
            jacobian = 1.0 - alpha_i
            
        elif self.warp_type == 'quadratic':
            alpha_i = self.alpha[sample_idx]
            beta_i = self.beta[sample_idx]
            gamma_i = self.gamma[sample_idx]
            if not self.component_specific_warp:
                alpha_i = alpha_i.expand(-1, self.num_components)
                beta_i = beta_i.expand(-1, self.num_components)
                gamma_i = gamma_i.expand(-1, self.num_components)
            t_warped = t - (alpha_i * (t ** 2) + beta_i * t + gamma_i)
            jacobian = 1.0 - (2.0 * alpha_i * t + beta_i)
            
        elif self.warp_type == 'spline':
            beta_i = self.beta[sample_idx]
            log_inc_i = self.log_increments[sample_idx]
            if not self.component_specific_warp:
                beta_i = beta_i.expand(-1, self.num_components)
                log_inc_i = log_inc_i.expand(-1, -1, self.num_components)
            
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
            alpha = self.alpha
            beta = self.beta
            if not self.component_specific_warp:
                alpha = alpha.expand(-1, self.num_components)
                beta = beta.expand(-1, self.num_components)
            # t_grid: (num_time,) -> (1, 1, num_time)
            # alpha: (num_samples, num_components) -> (num_samples, num_components, 1)
            t_warped = t_grid.view(1, 1, -1) - (
                alpha.unsqueeze(-1) * t_grid.view(1, 1, -1) + beta.unsqueeze(-1)
            )  # (num_samples, num_components, num_time)
            jacobian = 1.0 - alpha.unsqueeze(-1)  # (num_samples, num_components, 1)
            
        elif self.warp_type == 'quadratic':
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            if not self.component_specific_warp:
                alpha = alpha.expand(-1, self.num_components)
                beta = beta.expand(-1, self.num_components)
                gamma = gamma.expand(-1, self.num_components)
            t_warped = t_grid.view(1, 1, -1) - (
                alpha.unsqueeze(-1) * (t_grid.view(1, 1, -1) ** 2) +
                beta.unsqueeze(-1) * t_grid.view(1, 1, -1) +
                gamma.unsqueeze(-1)
            )  # (num_samples, num_components, num_time)
            jacobian = 1.0 - (
                2.0 * alpha.unsqueeze(-1) * t_grid.view(1, 1, -1) + beta.unsqueeze(-1)
            )
            
        elif self.warp_type == 'spline':
            beta = self.beta
            log_increments = self.log_increments
            if not self.component_specific_warp:
                beta = beta.expand(-1, self.num_components)
                log_increments = log_increments.expand(-1, -1, self.num_components)
            inc = (1.0 / self.num_segments) * torch.exp(log_increments)  # (num_samples, num_segments, num_components)
            inc = inc.permute(0, 2, 1)  # (num_samples, num_components, num_segments)
            zeros = torch.zeros((self.num_samples, self.num_components, 1), device=device, dtype=inc.dtype)
            cum_inc = torch.cumsum(torch.cat([zeros, inc], dim=2), dim=2)  # (num_samples, num_components, num_segments + 1)
            w = beta.unsqueeze(-1) + cum_inc  # (num_samples, num_components, num_segments + 1)
            
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
