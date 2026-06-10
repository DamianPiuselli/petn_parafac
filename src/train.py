"""
Training and Validation Loops.
Handles dataset preparation, optimizer stepping, constraints projection, and model evaluation.
"""
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.generator import EEMGenerator
from src.model import PINNParafac
from src.loss import masked_mse_loss

class EEMDataset(Dataset):
    """
    Converts a 3D EEM tensor of shape (num_samples, num_ex, num_em)
    into a flat coordinate dataset of (sample_idx, ex_idx, em_idx) triplets
    with their corresponding target intensity and optional mask values.
    """
    def __init__(self, X, mask=None):
        """
        Args:
            X: numpy array of shape (num_samples, num_ex, num_em)
            mask: numpy array of shape (num_samples, num_ex, num_em) or (num_ex, num_em).
                  If None, a mask of all ones is assumed.
        """
        self.num_samples, self.num_ex, self.num_em = X.shape
        self.X = torch.tensor(X, dtype=torch.float32)
        
        # Determine mask shape and expand if necessary
        if mask is None:
            self.mask = torch.ones_like(self.X)
        else:
            mask_t = torch.tensor(mask, dtype=torch.float32)
            if mask_t.ndim == 2:
                # Broadcast across samples: (num_ex, num_em) -> (num_samples, num_ex, num_em)
                self.mask = mask_t.unsqueeze(0).expand(self.num_samples, -1, -1)
            else:
                self.mask = mask_t
                
        # Pre-calculate flat indices to avoid 3D index conversion in __getitem__
        # Create a grid of indices
        sample_grid, ex_grid, em_grid = torch.meshgrid(
            torch.arange(self.num_samples),
            torch.arange(self.num_ex),
            torch.arange(self.num_em),
            indexing='ij'
        )
        
        self.sample_indices = sample_grid.reshape(-1)
        self.ex_indices = ex_grid.reshape(-1)
        self.em_indices = em_grid.reshape(-1)
        self.intensities = self.X.reshape(-1)
        self.mask_values = self.mask.reshape(-1)
        
    def __len__(self):
        return len(self.intensities)
        
    def __getitem__(self, idx):
        return (
            self.sample_indices[idx],
            self.ex_indices[idx],
            self.em_indices[idx],
            self.intensities[idx],
            self.mask_values[idx]
        )


def match_and_align_components(true_A, true_B, true_C, pred_A, pred_B, pred_C):
    """
    Resolves permutation and scaling ambiguities of PARAFAC decomposition.
    Normalizes profile vectors so they have a maximum value of 1.0 (magnitude absorbed in A),
    then uses linear sum assignment on the correlation matrix to align predictions with true components.
    
    Args:
        true_A, true_B, true_C: Ground truth score and spectral loading matrices
        pred_A, pred_B, pred_C: Predicted score and spectral loading matrices
        
    Returns:
        aligned_pred_A, aligned_pred_B, aligned_pred_C: Aligned and scaled predictions
        r2_scores: Dictionary of R2 similarity scores for A, B, and C components
    """
    num_components = true_A.shape[1]
    
    # 1. Normalize predicted spectral profiles to max = 1.0 and scale scores accordingly
    norm_pred_A = pred_A.copy()
    norm_pred_B = pred_B.copy()
    norm_pred_C = pred_C.copy()
    
    for r in range(num_components):
        max_b = np.max(pred_B[:, r])
        max_c = np.max(pred_C[:, r])
        
        # Avoid division by zero
        max_b = max_b if max_b > 1e-8 else 1.0
        max_c = max_c if max_c > 1e-8 else 1.0
        
        norm_pred_B[:, r] /= max_b
        norm_pred_C[:, r] /= max_c
        norm_pred_A[:, r] *= (max_b * max_c)
        
    # Same normalization on true components (though excitation/emission Gaussians already peak at 1.0)
    norm_true_A = true_A.copy()
    norm_true_B = true_B.copy()
    norm_true_C = true_C.copy()
    for r in range(num_components):
        max_b = np.max(true_B[:, r])
        max_c = np.max(true_C[:, r])
        max_b = max_b if max_b > 1e-8 else 1.0
        max_c = max_c if max_c > 1e-8 else 1.0
        norm_true_B[:, r] /= max_b
        norm_true_C[:, r] /= max_c
        norm_true_A[:, r] *= (max_b * max_c)
        
    # 2. Build cost matrix for permutation alignment using correlation coefficients of excitation profiles
    # Cost = 1 - correlation_coefficient (to minimize cost)
    cost_matrix = np.zeros((num_components, num_components))
    for i in range(num_components):
        for j in range(num_components):
            # Compute correlation between true excitation component i and predicted excitation component j
            corr_b = np.corrcoef(norm_true_B[:, i], norm_pred_B[:, j])[0, 1]
            corr_c = np.corrcoef(norm_true_C[:, i], norm_pred_C[:, j])[0, 1]
            # Average similarity score (1.0 is perfect correlation)
            similarity = 0.5 * (corr_b + corr_c)
            cost_matrix[i, j] = 1.0 - similarity
            
    # 3. Solve the assignment problem
    true_ind, pred_ind = linear_sum_assignment(cost_matrix)
    
    # 4. Reorder predicted matrices to match true components
    aligned_pred_A = norm_pred_A[:, pred_ind]
    aligned_pred_B = norm_pred_B[:, pred_ind]
    aligned_pred_C = norm_pred_C[:, pred_ind]
    
    # 5. Calculate R2 similarity for each component (true vs. aligned pred)
    r2_A = []
    r2_B = []
    r2_C = []
    for r in range(num_components):
        # Apply optimal linear scale matching on scores to resolve global scaling ambiguity
        # s_r = sum(true * pred) / sum(pred^2)
        true_col = norm_true_A[:, r]
        pred_col = aligned_pred_A[:, r]
        s_r = np.sum(true_col * pred_col) / (np.sum(pred_col ** 2) + 1e-8)
        scaled_pred_col = s_r * pred_col
        aligned_pred_A[:, r] = scaled_pred_col
        
        # Now compute R^2
        ss_res_a = np.sum((true_col - scaled_pred_col) ** 2)
        ss_tot_a = np.sum((true_col - np.mean(true_col)) ** 2)
        r2_a = 1.0 - (ss_res_a / ss_tot_a) if ss_tot_a > 1e-8 else 0.0
        r2_A.append(r2_a)
        
        # Excitation R^2
        ss_res_b = np.sum((norm_true_B[:, r] - aligned_pred_B[:, r]) ** 2)
        ss_tot_b = np.sum((norm_true_B[:, r] - np.mean(norm_true_B[:, r])) ** 2)
        r2_b = 1.0 - (ss_res_b / ss_tot_b) if ss_tot_b > 1e-8 else 0.0
        r2_B.append(r2_b)
        
        # Emission R^2
        ss_res_c = np.sum((norm_true_C[:, r] - aligned_pred_C[:, r]) ** 2)
        ss_tot_c = np.sum((norm_true_C[:, r] - np.mean(norm_true_C[:, r])) ** 2)
        r2_c = 1.0 - (ss_res_c / ss_tot_c) if ss_tot_c > 1e-8 else 0.0
        r2_C.append(r2_c)
        
    return aligned_pred_A, aligned_pred_B, aligned_pred_C, {
        'r2_A': r2_A,
        'r2_B': r2_B,
        'r2_C': r2_C
    }


def train_pinn_mvp(epochs=600, lr=0.01, batch_size=512, seed=42):
    """
    Runs the full Phase 1 training pipeline:
    1. Generates synthetic data.
    2. Builds the model.
    3. Trains on coordinates.
    4. Evaluates component alignment and spectral recovery.
    """
    # Fix random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Generate data
    print("Generating synthetic EEM data with scattering and IFE non-linear corruption...")
    generator = EEMGenerator(num_samples=20, num_ex=60, num_em=100, num_components=3, seed=seed)
    data = generator.generate_dataset(noise_std=0.005, corrupt_scatter=True, corrupt_ife=True)
    
    X = data['X']
    X_true = data['X_true']
    true_A = data['A']
    true_B = data['B']
    true_C = data['C']
    mask_2d = data['mask']
    gamma_true = data['gamma']
    
    # Pre-flatten coordinate vectors for fast full-batch training on CPU
    sample_grid, ex_grid, em_grid = np.meshgrid(
        np.arange(generator.num_samples),
        np.arange(generator.num_ex),
        np.arange(generator.num_em),
        indexing='ij'
    )
    sample_indices = torch.tensor(sample_grid.reshape(-1), dtype=torch.long)
    ex_indices = torch.tensor(ex_grid.reshape(-1), dtype=torch.long)
    em_indices = torch.tensor(em_grid.reshape(-1), dtype=torch.long)
    intensities = torch.tensor(X.reshape(-1), dtype=torch.float32)
    
    # Broadcast 2D mask to 3D for coordinates
    mask_3d = mask_2d[np.newaxis, :, :].repeat(generator.num_samples, axis=0)
    mask_values = torch.tensor(mask_3d.reshape(-1), dtype=torch.float32)
    
    # 2. Instantiate model
    print("Building PINN model...")
    model = PINNParafac(
        num_samples=generator.num_samples,
        num_ex=generator.num_ex,
        num_em=generator.num_em,
        ex_wavelens=generator.ex_wavelens,
        em_wavelens=generator.em_wavelens,
        num_components=generator.num_components
    )
    
    # Configure optimizer with separate learning rates and L2 regularization for the MLP parameters
    optimizer = optim.Adam([
        {'params': [model.sample_embeddings.weight, model.ex_embeddings.weight, model.em_embeddings.weight], 'lr': 0.008},
        {'params': model.ife_network.parameters(), 'lr': 0.001, 'weight_decay': 1e-4}
    ])
    
    # 3. Training loop
    epochs = 1200
    print(f"Training model in full-batch mode for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(sample_indices, ex_indices, em_indices)
        loss = masked_mse_loss(predictions, intensities, mask_values)
        
        loss.backward()
        optimizer.step()
        
        # Enforce positive constraints
        model.project_constraints()
        
        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{epochs} - Loss: {loss.item():.6f}")
            
    # 4. Extract trained weights
    model.eval()
    with torch.no_grad():
        pred_A = model.sample_embeddings.weight.cpu().numpy()
        pred_B = model.ex_embeddings.weight.cpu().numpy()
        pred_C = model.em_embeddings.weight.cpu().numpy()
        
    # 5. Match and align components
    aligned_A, aligned_B, aligned_C, metrics = match_and_align_components(
        true_A, true_B, true_C, pred_A, pred_B, pred_C
    )
    
    # 6. Save comparison plot
    from src.utils import plot_resolved_vs_true_profiles, plot_eem_heatmaps, plot_ife_comparison
    plot_resolved_vs_true_profiles(
        true_B, true_C, aligned_B, aligned_C,
        generator.ex_wavelens, generator.em_wavelens,
        save_path='notebooks/phase3_resolved_profiles.png'
    )
    
    # Retrieve the learned 2D IFE matrix
    gamma_learned = model.get_learned_ife_matrix()
    
    # Compute full reconstructed observed tensor to visualize heatmaps
    X_reconstructed = np.einsum('ir,jr,kr->ijk', aligned_A, aligned_B, aligned_C) * gamma_learned[np.newaxis, :, :]
    
    # Save heatmap plot
    plot_eem_heatmaps(
        X_clean=X_true[0],
        X_corrupted=X[0],
        mask=mask_2d,
        X_reconstructed=X_reconstructed[0],
        ex_wavelens=generator.ex_wavelens,
        em_wavelens=generator.em_wavelens,
        save_path='notebooks/phase3_eem_heatmaps.png'
    )
    
    # Save IFE comparison plot
    plot_ife_comparison(
        true_gamma=gamma_true,
        pred_gamma=gamma_learned,
        ex_wavelens=generator.ex_wavelens,
        em_wavelens=generator.em_wavelens,
        save_path='notebooks/phase3_ife_comparison.png'
    )
    
    # 7. Print and return metrics
    print("\n--- Model Evaluation Results ---")
    print(f"Sample scores (A) R2 scores:      {metrics['r2_A']}")
    print(f"Excitation loadings (B) R2 scores: {metrics['r2_B']}")
    print(f"Emission loadings (C) R2 scores:   {metrics['r2_C']}")
    
    avg_r2_B = np.mean(metrics['r2_B'])
    avg_r2_C = np.mean(metrics['r2_C'])
    print(f"Average Excitation Loading R2:    {avg_r2_B:.4f}")
    print(f"Average Emission Loading R2:      {avg_r2_C:.4f}")
    
    return {
        'model': model,
        'generator': generator,
        'aligned_A': aligned_A,
        'aligned_B': aligned_B,
        'aligned_C': aligned_C,
        'metrics': metrics
    }

if __name__ == '__main__':
    train_pinn_mvp()
