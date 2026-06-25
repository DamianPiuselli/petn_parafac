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
    Supports softplus reparameterization, learnable/dynamic background CDOM solvent profiles,
    and automatic component rank selection.
    """
    def __init__(self, num_samples, num_ex, num_em, ex_wavelens, em_wavelens,
                 ex_bg=None, em_bg=None, num_components=3,
                 use_softplus=True, learnable_bg=False, dynamic_bg=False,
                 enable_rank_selection=False):
        super().__init__()
        self.num_samples = num_samples
        self.num_ex = num_ex
        self.num_em = num_em
        self.num_components = num_components
        self.use_softplus = use_softplus
        self.learnable_bg = learnable_bg
        self.dynamic_bg = dynamic_bg
        self.enable_rank_selection = enable_rank_selection
        
        # 1. Trilinear core embedding layers
        self.sample_embeddings = nn.Embedding(num_samples, num_components)
        self.ex_embeddings = nn.Embedding(num_ex, num_components)
        self.em_embeddings = nn.Embedding(num_em, num_components)
        
        # 2. IFE Molar absorptivity scaling parameters (non-negative)
        self.alpha = nn.Parameter(torch.ones(num_components) * 0.1)
        
        # 3. Optional automatic component rank selection salience weights
        if self.enable_rank_selection:
            self.comp_weights = nn.Parameter(torch.ones(num_components))
        
        # 4. Register/parameterize physical background CDOM profiles
        if ex_bg is None:
            ex_bg = torch.zeros(num_ex)
        else:
            ex_bg = torch.as_tensor(ex_bg, dtype=torch.float32)
            
        if em_bg is None:
            em_bg = torch.zeros(num_em)
        else:
            em_bg = torch.as_tensor(em_bg, dtype=torch.float32)
            
        if self.learnable_bg:
            self.ex_bg = nn.Parameter(ex_bg.clone())
            self.em_bg = nn.Parameter(em_bg.clone())
        else:
            self.register_buffer('ex_bg', ex_bg)
            self.register_buffer('em_bg', em_bg)
            
        # Optional sample-specific CDOM drift profiles (must be smooth)
        if self.dynamic_bg:
            self.ex_bg_drift = nn.Embedding(num_samples, num_ex)
            self.em_bg_drift = nn.Embedding(num_samples, num_em)
            
        # Register physical wavelengths
        self.register_buffer('ex_wavelens', torch.tensor(ex_wavelens, dtype=torch.float32))
        self.register_buffer('em_wavelens', torch.tensor(em_wavelens, dtype=torch.float32))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes all embeddings with positive values, taking softplus into account if enabled."""
        if self.use_softplus:
            # We initialize raw weights such that softplus(weight) matches target uniform initialization
            # Since softplus(x) = log(1 + exp(x)), the inverse is x = log(exp(y) - 1)
            with torch.no_grad():
                y_sample = torch.empty_like(self.sample_embeddings.weight).uniform_(0.1, 1.0)
                self.sample_embeddings.weight.copy_(torch.log(torch.exp(y_sample) - 1.0))
                
                y_ex = torch.empty_like(self.ex_embeddings.weight).uniform_(0.1, 1.0)
                self.ex_embeddings.weight.copy_(torch.log(torch.exp(y_ex) - 1.0))
                
                y_em = torch.empty_like(self.em_embeddings.weight).uniform_(0.1, 1.0)
                self.em_embeddings.weight.copy_(torch.log(torch.exp(y_em) - 1.0))
                
                y_alpha = torch.empty_like(self.alpha).uniform_(0.01, 0.20)
                self.alpha.copy_(torch.log(torch.exp(y_alpha) - 1.0))
                
                if self.enable_rank_selection:
                    y_comp = torch.empty_like(self.comp_weights).uniform_(0.8, 1.2)
                    self.comp_weights.copy_(torch.log(torch.exp(y_comp) - 1.0))
        else:
            nn.init.uniform_(self.sample_embeddings.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.ex_embeddings.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.em_embeddings.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.alpha.data, a=0.01, b=0.20)
            
            if self.enable_rank_selection:
                nn.init.uniform_(self.comp_weights.data, a=0.8, b=1.2)

        if self.dynamic_bg:
            nn.init.zeros_(self.ex_bg_drift.weight)
            nn.init.zeros_(self.em_bg_drift.weight)

    @torch.no_grad()
    def project_constraints(self):
        """
        Applies the physical non-negativity constraint.
        Forces concentrations, spectral profiles, and absorptivity scaling to be non-negative.
        Under use_softplus=True, the softplus activation itself handles the constraint,
        so this function only clamps parameters if use_softplus=False.
        """
        if not self.use_softplus:
            self.sample_embeddings.weight.clamp_(min=0.0)
            self.ex_embeddings.weight.clamp_(min=0.0)
            self.em_embeddings.weight.clamp_(min=0.0)
            self.alpha.clamp_(min=0.0)
            if self.enable_rank_selection:
                self.comp_weights.clamp_(min=0.0)
                
        if self.learnable_bg:
            if isinstance(self.ex_bg, nn.Parameter):
                self.ex_bg.clamp_(min=0.0)
            if isinstance(self.em_bg, nn.Parameter):
                self.em_bg.clamp_(min=0.0)

    @torch.no_grad()
    def get_resolved_factors(self):
        """
        Returns the resolved, non-negative scores, loadings, and other parameters.
        Applies softplus transformation if use_softplus=True, or clamping if use_softplus=False.
        
        Returns:
            A: numpy array of shape (num_samples, num_components)
            B: numpy array of shape (num_ex, num_components)
            C: numpy array of shape (num_em, num_components)
            alpha: numpy array of shape (num_components,)
            comp_weights: numpy array of shape (num_components,) or None
        """
        if self.use_softplus:
            A = torch.nn.functional.softplus(self.sample_embeddings.weight)
            B = torch.nn.functional.softplus(self.ex_embeddings.weight)
            C = torch.nn.functional.softplus(self.em_embeddings.weight)
            alpha = torch.nn.functional.softplus(self.alpha)
            comp_weights = torch.nn.functional.softplus(self.comp_weights) if self.enable_rank_selection else None
        else:
            A = self.sample_embeddings.weight
            B = self.ex_embeddings.weight
            C = self.em_embeddings.weight
            alpha = self.alpha
            comp_weights = self.comp_weights if self.enable_rank_selection else None
            
        A_np = A.cpu().numpy()
        B_np = B.cpu().numpy()
        C_np = C.cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        comp_weights_np = comp_weights.cpu().numpy() if comp_weights is not None else None
        
        return A_np, B_np, C_np, alpha_np, comp_weights_np

    @torch.no_grad()
    def get_learned_absorptivities(self):
        """
        Extracts the resolved excitation and emission molar absorptivities.
        
        Returns:
            E: 2D numpy array of shape (num_ex, num_components)
            M: 2D numpy array of shape (num_em, num_components)
        """
        import numpy as np
        _, B_np, _, alpha_np, comp_weights_np = self.get_resolved_factors()
        
        if self.enable_rank_selection:
            E = alpha_np * B_np * comp_weights_np
        else:
            E = alpha_np * B_np
            
        M = np.zeros((self.num_em, self.num_components))
        return E, M

    def get_background_smoothness_loss(self):
        """
        Computes the L2 smoothness penalty on background CDOM profiles.
        Penalizes the first difference of the background and background drift profiles.
        """
        loss = torch.tensor(0.0)
        
        if self.learnable_bg:
            # Make sure to compute device-compatible loss
            loss = loss.to(self.ex_bg.device)
            diff_ex_base = self.ex_bg[1:] - self.ex_bg[:-1]
            loss = loss + torch.mean(diff_ex_base ** 2)
            
            diff_em_base = self.em_bg[1:] - self.em_bg[:-1]
            loss = loss + torch.mean(diff_em_base ** 2)
            
        if self.dynamic_bg:
            loss = loss.to(self.ex_bg_drift.weight.device)
            diff_ex_drift = self.ex_bg_drift.weight[:, 1:] - self.ex_bg_drift.weight[:, :-1]
            loss = loss + torch.mean(diff_ex_drift ** 2)
            
            diff_em_drift = self.em_bg_drift.weight[:, 1:] - self.em_bg_drift.weight[:, :-1]
            loss = loss + torch.mean(diff_em_drift ** 2)
            
        return loss

    def get_sparsity_loss(self):
        """
        Computes the L1 sparsity loss on the component salience weights for automatic rank selection.
        """
        if not self.enable_rank_selection:
            return torch.tensor(0.0)
            
        if self.use_softplus:
            weights = torch.nn.functional.softplus(self.comp_weights)
        else:
            weights = self.comp_weights
            
        return torch.sum(torch.abs(weights))

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
        # 1. Trilinear core lookup & activation
        if self.use_softplus:
            a = torch.nn.functional.softplus(self.sample_embeddings(sample_idx))
            b = torch.nn.functional.softplus(self.ex_embeddings(ex_idx))
            c = torch.nn.functional.softplus(self.em_embeddings(em_idx))
            alpha = torch.nn.functional.softplus(self.alpha)
            if self.enable_rank_selection:
                comp_weights = torch.nn.functional.softplus(self.comp_weights)
        else:
            a = self.sample_embeddings(sample_idx)
            b = self.ex_embeddings(ex_idx)
            c = self.em_embeddings(em_idx)
            alpha = self.alpha
            if self.enable_rank_selection:
                comp_weights = self.comp_weights
                
        # 2. Calculate unattenuated true intensity
        if self.enable_rank_selection:
            I_true = torch.sum(a * b * c * comp_weights, dim=1)
        else:
            I_true = torch.sum(a * b * c, dim=1)
            
        # 3. Calculate total absorbances
        if self.enable_rank_selection:
            Abs_ex = torch.sum(a * (alpha * b) * comp_weights, dim=1)
        else:
            Abs_ex = torch.sum(a * (alpha * b), dim=1)
            
        if self.dynamic_bg:
            ex_drift = self.ex_bg_drift(sample_idx).gather(1, ex_idx.unsqueeze(1)).squeeze(1)
            em_drift = self.em_bg_drift(sample_idx).gather(1, em_idx.unsqueeze(1)).squeeze(1)
            Abs_ex = Abs_ex + self.ex_bg[ex_idx] + ex_drift
            Abs_em = self.em_bg[em_idx] + em_drift
        else:
            Abs_ex = Abs_ex + self.ex_bg[ex_idx]
            Abs_em = self.em_bg[em_idx]
            
        # 4. Calculate attenuation
        gamma = torch.pow(10.0, -(Abs_ex + Abs_em))
        
        # Combine
        y_pred = I_true * gamma
        return y_pred
