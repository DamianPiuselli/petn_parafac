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
    def __init__(self, num_samples, num_ex, num_em, ex_wavelens, em_wavelens, num_components=3):
        super().__init__()
        self.num_samples = num_samples
        self.num_ex = num_ex
        self.num_em = num_em
        self.num_components = num_components
        
        # Initialize three independent embedding layers
        self.sample_embeddings = nn.Embedding(num_samples, num_components)
        self.ex_embeddings = nn.Embedding(num_ex, num_components)
        self.em_embeddings = nn.Embedding(num_em, num_components)
        
        # Register physical wavelengths and their bounds for normalized MLP inputs
        self.register_buffer('ex_wavelens', torch.tensor(ex_wavelens, dtype=torch.float32))
        self.register_buffer('em_wavelens', torch.tensor(em_wavelens, dtype=torch.float32))
        self.register_buffer('ex_min', torch.tensor(ex_wavelens[0], dtype=torch.float32))
        self.register_buffer('ex_max', torch.tensor(ex_wavelens[-1], dtype=torch.float32))
        self.register_buffer('em_min', torch.tensor(em_wavelens[0], dtype=torch.float32))
        self.register_buffer('em_max', torch.tensor(em_wavelens[-1], dtype=torch.float32))
        
        # Parallel Feedforward Dense Network (Black-Box head) for IFE matrix attenuation coefficient
        self.ife_network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Forces output attenuation gamma to sit strictly between 0 and 1
        )
        
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

    @torch.no_grad()
    def get_learned_ife_matrix(self):
        """
        Computes the full learned 2D IFE matrix of shape (num_ex, num_em).
        
        Returns:
            gamma: 2D numpy array of shape (num_ex, num_em)
        """
        # Create 2D grid of normalized wavelengths
        ex_grid = (self.ex_wavelens - self.ex_min) / (self.ex_max - self.ex_min)
        em_grid = (self.em_wavelens - self.em_min) / (self.em_max - self.em_min)
        
        EX, EM = torch.meshgrid(ex_grid, em_grid, indexing='ij')
        inputs = torch.stack([EX.reshape(-1), EM.reshape(-1)], dim=1)
        
        gamma = self.ife_network(inputs).reshape(self.num_ex, self.num_em)
        return gamma.cpu().numpy()

    def forward(self, sample_idx, ex_idx, em_idx):
        """
        Computes the forward pass.
        Args:
            sample_idx: Tensor of shape (BatchSize,) containing sample indices
            ex_idx: Tensor of shape (BatchSize,) containing excitation wavelength indices
            em_idx: Tensor of shape (BatchSize,) containing emission wavelength indices
        Returns:
            y_pred: Tensor of shape (BatchSize,) containing the predicted observed fluorescence intensity
        """
        # 1. Look up embeddings: shape (BatchSize, num_components)
        a = self.sample_embeddings(sample_idx)
        b = self.ex_embeddings(ex_idx)
        c = self.em_embeddings(em_idx)
        
        # Calculate trilinear core ideal intensity (I_true)
        # I_true = sum_r A(i,r) * B(j,r) * C(k,r)
        I_true = torch.sum(a * b * c, dim=1)
        
        # 2. Look up continuous wavelengths and normalize
        ex_val = self.ex_wavelens[ex_idx]
        em_val = self.em_wavelens[em_idx]
        
        ex_norm = (ex_val - self.ex_min) / (self.ex_max - self.ex_min)
        em_norm = (em_val - self.em_min) / (self.em_max - self.em_min)
        
        # 3. Stack inputs and run through IFE network
        inputs_dense = torch.stack([ex_norm, em_norm], dim=1)
        gamma = self.ife_network(inputs_dense).squeeze(1)
        
        # 4. Combine: I_obs = I_true * gamma
        y_pred = I_true * gamma
        return y_pred
