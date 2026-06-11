"""
Training and Validation on Real-World Amino Acids EEM Benchmark Dataset.
Loads amino.mat, creates custom scattering masks, trains the PETN cuvette model,
and evaluates resolved concentration scores and loading spectra.
"""
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

from src.model import PETNParafac
from src.loss import masked_mse_loss

def generate_aminoacids_scattering_mask(ex_wavelens, em_wavelens):
    """
    Generates a 2D binary mask of shape (num_ex, num_em) to mask out
    Rayleigh and Raman scattering for the amino acids wavelength grid.
    """
    num_ex = len(ex_wavelens)
    num_em = len(em_wavelens)
    mask = np.ones((num_ex, num_em))
    
    # 1st-order Rayleigh scattering band: em = ex (width: +/- 12 nm)
    # Solvent Raman scattering: em_raman = ex / (1.0 - 3.4e-4 * ex) (width: +/- 10 nm)
    for j in range(num_ex):
        ex = ex_wavelens[j]
        em_raman = ex / (1.0 - 3.4e-4 * ex)
        for k in range(num_em):
            em = em_wavelens[k]
            
            # Mask 1st-order Rayleigh
            if abs(em - ex) <= 12.0:
                mask[j, k] = 0.0
                
            # Mask solvent Raman
            if abs(em - em_raman) <= 10.0:
                mask[j, k] = 0.0
                
    return mask

def train_aminoacids_dataset(epochs=3000, lr=0.008, seed=43):
    """
    Loads amino.mat, trains PETNParafac, and prints/plots evaluation results.
    """
    # 1. Load data
    mat_path = 'data/raw/amino.mat'
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Raw dataset not found at {mat_path}. Run src/download_aminoacids.py first.")
        
    print(f"Loading raw dataset from {mat_path}...")
    mat = loadmat(mat_path)
    
    # Extract variables
    ex_wavelens = mat['ExAx'].squeeze() # (61,) from 240 to 300
    em_wavelens = mat['EmAx'].squeeze() # (201,) from 250 to 450
    X_flat = mat['X'] # (5, 12261)
    y_true = mat['y'] # (5, 3) concentrations of Trp, Tyr, Phe
    
    # Reshape X from flat to (5, 61, 201) matching Fortran column-major order to C-order
    num_samples = X_flat.shape[0]
    num_ex = len(ex_wavelens)
    num_em = len(em_wavelens)
    X = X_flat.reshape(num_samples, num_ex, num_em)
    
    print(f"Dataset shape: {X.shape} (Samples: {num_samples}, Ex: {num_ex}, Em: {num_em})")
    
    # 2. Generate custom scattering mask
    mask_2d = generate_aminoacids_scattering_mask(ex_wavelens, em_wavelens)
    
    # Flat coordinate index grid
    sample_grid, ex_grid, em_grid = np.meshgrid(
        np.arange(num_samples),
        np.arange(num_ex),
        np.arange(num_em),
        indexing='ij'
    )
    sample_indices = torch.tensor(sample_grid.reshape(-1), dtype=torch.long)
    ex_indices = torch.tensor(ex_grid.reshape(-1), dtype=torch.long)
    em_indices = torch.tensor(em_grid.reshape(-1), dtype=torch.long)
    intensities = torch.tensor(X.reshape(-1), dtype=torch.float32)
    
    # Broadcast mask to 3D
    mask_3d = mask_2d[np.newaxis, :, :].repeat(num_samples, axis=0)
    mask_values = torch.tensor(mask_3d.reshape(-1), dtype=torch.float32)
    
    # 3. Instantiate model with zero background absorbances (pure water solvent)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ex_bg = torch.zeros(num_ex)
    em_bg = torch.zeros(num_em)
    
    print("Building PETN cuvette model...")
    model = PETNParafac(
        num_samples=num_samples,
        num_ex=num_ex,
        num_em=num_em,
        ex_wavelens=ex_wavelens,
        em_wavelens=em_wavelens,
        ex_bg=ex_bg,
        em_bg=em_bg,
        num_components=3
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training loop
    print(f"Training jointly from scratch in full-batch mode for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(sample_indices, ex_indices, em_indices)
        loss = masked_mse_loss(predictions, intensities, mask_values)
        
        loss.backward()
        optimizer.step()
        model.project_constraints()
        
        if epoch % 300 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{epochs} - Loss: {loss.item():.6f}")
            
    # 5. Extract trained weights
    model.eval()
    with torch.no_grad():
        pred_A = model.sample_embeddings.weight.cpu().numpy()
        pred_B = model.ex_embeddings.weight.cpu().numpy()
        pred_C = model.em_embeddings.weight.cpu().numpy()
        pred_E, pred_M = model.get_learned_absorptivities()
        
    # 6. Align predicted components to Tryptophan, Tyrosine, and Phenylalanine
    # based on the correlation of concentrations to the true concentrations matrix y
    cost_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # Compute correlation between true concentration i and predicted scores j
            corr = np.corrcoef(y_true[:, i], pred_A[:, j])[0, 1]
            # Handle possible NaNs in correlation calculation
            if np.isnan(corr):
                corr = -1.0
            cost_matrix[i, j] = 1.0 - corr
            
    true_ind, pred_ind = linear_sum_assignment(cost_matrix)
    
    aligned_A = pred_A[:, pred_ind]
    aligned_B = pred_B[:, pred_ind]
    aligned_C = pred_C[:, pred_ind]
    aligned_E = pred_E[:, pred_ind]
    
    # Normalize loading profiles to peak at 1.0
    for r in range(3):
        max_b = np.max(aligned_B[:, r])
        max_c = np.max(aligned_C[:, r])
        aligned_B[:, r] /= (max_b if max_b > 1e-8 else 1.0)
        aligned_C[:, r] /= (max_c if max_c > 1e-8 else 1.0)
        
    # Optimal least-squares scale alignment of concentrations
    r2_scores = []
    scale_factors = []
    for r in range(3):
        true_col = y_true[:, r]
        pred_col = aligned_A[:, r]
        s_r = np.sum(true_col * pred_col) / (np.sum(pred_col ** 2) + 1e-8)
        scaled_pred_col = s_r * pred_col
        aligned_A[:, r] = scaled_pred_col
        scale_factors.append(s_r)
        
        ss_res = np.sum((true_col - scaled_pred_col) ** 2)
        ss_tot = np.sum((true_col - np.mean(true_col)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
        r2_scores.append(r2)
        
    print("\n--- Real-World Validation Results ---")
    print("True Concentrations (y_true):\n", y_true)
    print("Raw Predicted Scores (pred_A):\n", pred_A)
    print("Alignment indices (pred_ind):", pred_ind)
    print("Cost matrix:\n", cost_matrix)
    print("Aligned Scores (aligned_A):\n", aligned_A)
    
    names = ["Tryptophan (Trp)", "Tyrosine (Tyr)", "Phenylalanine (Phe)"]
    for r in range(3):
        print(f"Component {r+1} ({names[r]}):")
        print(f"  Concentration (A) R2: {r2_scores[r]:.4f}")
        
        # Verify peaks
        peak_ex = ex_wavelens[np.argmax(aligned_B[:, r])]
        peak_em = em_wavelens[np.argmax(aligned_C[:, r])]
        print(f"  Resolved Ex Peak: {peak_ex:.1f} nm, Em Peak: {peak_em:.1f} nm")
        
    avg_r2 = np.mean(r2_scores)
    print(f"\nAverage Concentration Recovery R2: {avg_r2:.4f}")
    
    # 7. Save resolved loading profiles comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for r in range(3):
        axes[0].plot(ex_wavelens, aligned_B[:, r], label=names[r], color=colors[r], linewidth=2.5)
        axes[1].plot(em_wavelens, aligned_C[:, r], label=names[r], color=colors[r], linewidth=2.5)
        
    axes[0].set_title('Resolved Excitation Loadings (B)')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Normalized Intensity')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    axes[1].set_title('Resolved Emission Loadings (C)')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Normalized Intensity')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    # Normalized concentration scores comparison
    max_y = np.max(y_true, axis=0, keepdims=True)
    max_y = np.where(max_y == 0, 1.0, max_y)
    norm_y_true = y_true / max_y
    norm_aligned_A = aligned_A / max_y

    axes[2].set_title('Concentration Recovery (Scores A)')
    x = np.arange(num_samples)
    width = 0.12
    
    axes[2].bar(x - 2.5 * width, norm_y_true[:, 0], width, label='Trp (True)', color='#1f77b4', alpha=0.4)
    axes[2].bar(x - 1.5 * width, norm_aligned_A[:, 0], width, label='Trp (Pred)', color='#1f77b4', edgecolor='#1f77b4', linewidth=1.5)
    
    axes[2].bar(x - 0.5 * width, norm_y_true[:, 1], width, label='Tyr (True)', color='#ff7f0e', alpha=0.4)
    axes[2].bar(x + 0.5 * width, norm_aligned_A[:, 1], width, label='Tyr (Pred)', color='#ff7f0e', edgecolor='#ff7f0e', linewidth=1.5)
    
    axes[2].bar(x + 1.5 * width, norm_y_true[:, 2], width, label='Phe (True)', color='#2ca02c', alpha=0.4)
    axes[2].bar(x + 2.5 * width, norm_aligned_A[:, 2], width, label='Phe (Pred)', color='#2ca02c', edgecolor='#2ca02c', linewidth=1.5)
    
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Normalized Concentration')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"S{i+1}" for i in range(num_samples)])
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(fontsize='small', ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    
    plt.tight_layout()
    os.makedirs('notebooks', exist_ok=True)
    plot_path = 'notebooks/aminoacids_resolved_profiles.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nResolved profiles plot saved to {plot_path}")
    
    return {
        'r2_scores': r2_scores,
        'avg_r2': avg_r2
    }

if __name__ == '__main__':
    train_aminoacids_dataset()
