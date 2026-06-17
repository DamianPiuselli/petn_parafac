"""
Training and Evaluation script for Chroma-PETN.
Implements the training loop, parameter updates, alignment projection, and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.chroma.model import ChromaPETN
from src.chroma.generator import ChromatographicDataGenerator

def train_chroma_petn(dataset, epochs=1200, lr=0.01, warp_reg_coef=0.001):
    """
    Trains the Chroma-PETN model on the provided dataset.
    
    Args:
        dataset: Dictionary containing the data matrix 'X' of shape (I, J, K)
        epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
        warp_reg_coef: Weight for warp parameter regularization
        
    Returns:
        model: Trained ChromaPETN model instance
    """
    X = torch.tensor(dataset['X'], dtype=torch.float32)
    I, J, K = X.shape
    
    # Instantiate model
    model = ChromaPETN(num_samples=I, num_time=J, num_spec=K, num_components=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Generate complete coordinate triplets for full-batch training
    coords_i, coords_j, coords_k = torch.meshgrid(
        torch.arange(I), torch.arange(J), torch.arange(K), indexing='ij'
    )
    coords_i = coords_i.flatten()
    coords_j = coords_j.flatten()
    coords_k = coords_k.flatten()
    y_target = X[coords_i, coords_j, coords_k]
    
    print(f"Training Chroma-PETN model for {epochs} epochs...")
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(coords_i, coords_j, coords_k)
        
        # MSE Reconstruction Loss
        loss_mse = nn.functional.mse_loss(y_pred, y_target)
        
        # Warp parameter regularization
        loss_warp_reg = warp_reg_coef * (torch.mean(model.warp_stretch**2) + torch.mean(model.warp_shift**2))
        
        # Total loss
        loss = loss_mse + loss_warp_reg
        
        loss.backward()
        optimizer.step()
        
        # Apply physical constraints (non-negativity clipping and warp centering)
        model.project_constraints()
        
        if epoch % 300 == 0:
            print(f"Epoch {epoch:4d} | MSE Loss: {loss_mse.item():.6f} | Warp Reg: {loss_warp_reg.item():.6f}")
            
    return model

def calculate_cosine_similarity(v1, v2):
    """Calculates cosine similarity (correlation coefficient) between two vectors."""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    return np.max([np.dot(v1_norm, v2_norm), np.dot(v1_norm, -v2_norm)])

def evaluate_chroma_alignment(model, dataset):
    """
    Evaluates profile recovery and shift recovery.
    Resolves permutation/scaling ambiguities and prints correlation metrics.
    
    Args:
        model: Trained ChromaPETN model instance
        dataset: Dictionary containing ground truth matrices and shifts
    Returns:
        metrics: Dictionary containing alignment similarity scores and shift recovery metrics
    """
    A_pred = model.sample_embeddings.weight.detach().cpu().numpy()
    B_pred = model.time_embeddings.weight.detach().cpu().numpy()
    C_pred = model.spec_embeddings.weight.detach().cpu().numpy()
    
    A_true = dataset['A'].copy()
    B_true = dataset['B'].copy()
    C_true = dataset['C'].copy()
    
    R = A_true.shape[1]
    
    # Compute spectral similarity matrix to match components
    sim_matrix_C = np.zeros((R, R))
    for r_pred in range(R):
        for r_true in range(R):
            sim_matrix_C[r_pred, r_true] = calculate_cosine_similarity(C_pred[:, r_pred], C_true[:, r_true])
            
    # Match predicted components to true components based on spectra
    perm = []
    used = set()
    for r in range(R):
        best_sim = -1.0
        best_idx = 0
        for r_true in range(R):
            if r_true in used:
                continue
            sim = sim_matrix_C[r, r_true]
            if sim > best_sim:
                best_sim = sim
                best_idx = r_true
        perm.append(best_idx)
        used.add(best_idx)
        
    print("\nPermutation mapping resolved (pred component index -> true component index):", perm)
    
    # Reorder predicted profiles
    A_pred_ordered = np.zeros_like(A_pred)
    B_pred_ordered = np.zeros_like(B_pred)
    C_pred_ordered = np.zeros_like(C_pred)
    
    for r in range(R):
        true_idx = perm[r]
        A_pred_ordered[:, true_idx] = A_pred[:, r]
        B_pred_ordered[:, true_idx] = B_pred[:, r]
        C_pred_ordered[:, true_idx] = C_pred[:, r]
        
    # Scale ambiguity resolution: normalize profiles to unit length and transfer scale to scores
    for r in range(R):
        norm_b = np.linalg.norm(B_pred_ordered[:, r])
        norm_c = np.linalg.norm(C_pred_ordered[:, r])
        B_pred_ordered[:, r] /= norm_b
        C_pred_ordered[:, r] /= norm_c
        A_pred_ordered[:, r] *= (norm_b * norm_c)
        
        norm_b_true = np.linalg.norm(B_true[:, r])
        norm_c_true = np.linalg.norm(C_true[:, r])
        B_true[:, r] /= norm_b_true
        C_true[:, r] /= norm_c_true
        A_true[:, r] *= (norm_b_true * norm_c_true)
        
    # Calculate recovery metrics
    b_sims = [calculate_cosine_similarity(B_pred_ordered[:, r], B_true[:, r]) for r in range(R)]
    c_sims = [calculate_cosine_similarity(C_pred_ordered[:, r], C_true[:, r]) for r in range(R)]
    a_sims = [calculate_cosine_similarity(A_pred_ordered[:, r], A_true[:, r]) for r in range(R)]
    
    # Evaluate shifts
    shifts_pred = model.warp_shift.detach().cpu().numpy()
    stretches_pred = model.warp_stretch.detach().cpu().numpy()
    
    shifts_true_mapped = dataset['shifts'] / (1.0 + dataset['stretches'])
    stretches_true_mapped = dataset['stretches'] / (1.0 + dataset['stretches'])
    
    shift_corr = np.corrcoef(shifts_pred, shifts_true_mapped)[0, 1]
    stretch_corr = np.corrcoef(stretches_pred, stretches_true_mapped)[0, 1]
    mean_shift_error = np.mean(np.abs(shifts_pred - shifts_true_mapped))
    
    # Calculate the fully aligned (unshifted) reconstructed tensor
    X_aligned = np.einsum('ir,jr,kr->ijk', A_pred_ordered, B_pred_ordered, C_pred_ordered)
    
    metrics = {
        'b_similarities': b_sims,
        'c_similarities': c_sims,
        'a_similarities': a_sims,
        'shift_correlation': shift_corr,
        'stretch_correlation': stretch_corr,
        'mean_shift_error': mean_shift_error
    }
    
    print("\n--- Chroma-PETN Model Recovery Evaluation ---")
    for r in range(R):
        print(f"Component {r+1}:")
        print(f"  Chromatography profile similarity: {b_sims[r]:.4f}")
        print(f"  Spectral profile similarity:       {c_sims[r]:.4f}")
        print(f"  Concentration score similarity:    {a_sims[r]:.4f}")
        
    print("\n--- Shift & Alignment Recovery ---")
    print(f"  Shift parameter correlation:  {shift_corr:.4f}")
    print(f"  Stretch parameter correlation: {stretch_corr:.4f}")
    print(f"  Mean absolute shift error:    {mean_shift_error:.4f} (normalized time units)")
    
    # Generate and save comparison plots in notebooks/chroma/
    import os
    os.makedirs('notebooks/chroma', exist_ok=True)
    
    from src.common.utils import (
        plot_chroma_resolved_vs_true_profiles,
        plot_chroma_alignment_comparison,
        plot_chroma_warp_parameters
    )
    
    time_grid = np.linspace(0.0, 1.0, B_true.shape[0])
    spec_grid = np.linspace(200.0, 400.0, C_true.shape[0])
    
    # 1. Resolved profiles comparison
    plot_chroma_resolved_vs_true_profiles(
        B_true, C_true, B_pred_ordered, C_pred_ordered,
        time_grid, spec_grid,
        save_path='notebooks/chroma/chroma_resolved_profiles.png'
    )
    
    # 2. Alignment comparison (Observed vs Aligned Chromatograms)
    plot_chroma_alignment_comparison(
        time_grid, dataset['X'], X_aligned,
        save_path='notebooks/chroma/chroma_alignment_comparison.png'
    )
    
    # 3. Warp parameters recovery plot
    plot_chroma_warp_parameters(
        shifts_true_mapped, stretches_true_mapped, shifts_pred, stretches_pred,
        save_path='notebooks/chroma/chroma_warp_parameters.png'
    )
    
    return metrics

if __name__ == "__main__":
    print("Generating synthetic chromatographic data...")
    generator = ChromatographicDataGenerator(num_samples=15, num_time=100, num_spec=80, num_components=3)
    dataset = generator.generate_dataset(noise_std=0.015, max_shift=0.05, max_stretch=0.08)
    
    model = train_chroma_petn(dataset, epochs=1200, lr=0.01)
    evaluate_chroma_alignment(model, dataset)
