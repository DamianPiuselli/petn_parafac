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

import pandas as pd
from src.eem.model import PETNParafac
from src.eem.loss import masked_mse_loss
from src.common.utils import EarlyStopping, plot_resolved_vs_true_profiles

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

def train_aminoacids_dataset(epochs=3000, lr=0.008, seed=43, patience=150, tol=1e-5):
    """
    Loads amino.mat, trains PETNParafac, and prints/plots evaluation results.
    """
    # 1. Load data
    mat_path = 'data/eem/aminoacids/amino.mat'
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Raw dataset not found at {mat_path}. Run src/eem/download_aminoacids.py first.")
        
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
    early_stopping = EarlyStopping(patience=patience, tol=tol, min_epochs=100)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(sample_indices, ex_indices, em_indices)
        loss = masked_mse_loss(predictions, intensities, mask_values)
        
        loss.backward()
        optimizer.step()
        model.project_constraints()
        
        loss_val = loss.item()
        if early_stopping(epoch, loss_val, intensities):
            break
            
        if epoch % 300 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{epochs} - Loss: {loss_val:.6f}")
            
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
    
    # 7. Save resolved loading profiles and scores comparison
    save_dir = 'notebooks/eem/experiments/aminoacids'
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Resolved Loadings in multi-row grid (excitation left, emission right)
    plot_resolved_vs_true_profiles(
        true_B=None,
        true_C=None,
        pred_B=aligned_B,
        pred_C=aligned_C,
        ex_wavelens=ex_wavelens,
        em_wavelens=em_wavelens,
        component_names=names,
        save_path=os.path.join(save_dir, 'aminoacids_resolved_profiles.png')
    )
    
    # Plot 2: Standalone Scores Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    max_y = np.max(y_true, axis=0, keepdims=True)
    max_y = np.where(max_y == 0, 1.0, max_y)
    norm_y_true = y_true / max_y
    norm_aligned_A = aligned_A / max_y

    ax.set_title('Concentration Recovery (Scores A) - Amino Acids', fontsize=12, fontweight='bold')
    x = np.arange(num_samples)
    width = 0.12
    
    ax.bar(x - 2.5 * width, norm_y_true[:, 0], width, label='Trp (True)', color='#1f77b4', alpha=0.4)
    ax.bar(x - 1.5 * width, norm_aligned_A[:, 0], width, label='Trp (Pred)', color='#1f77b4', edgecolor='#1f77b4', linewidth=1.5)
    
    ax.bar(x - 0.5 * width, norm_y_true[:, 1], width, label='Tyr (True)', color='#ff7f0e', alpha=0.4)
    ax.bar(x + 0.5 * width, norm_aligned_A[:, 1], width, label='Tyr (Pred)', color='#ff7f0e', edgecolor='#ff7f0e', linewidth=1.5)
    
    ax.bar(x + 1.5 * width, norm_y_true[:, 2], width, label='Phe (True)', color='#2ca02c', alpha=0.4)
    ax.bar(x + 2.5 * width, norm_aligned_A[:, 2], width, label='Phe (Pred)', color='#2ca02c', edgecolor='#2ca02c', linewidth=1.5)
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Normalized Concentration')
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in range(num_samples)])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize='small', ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    
    plt.tight_layout()
    scores_path = os.path.join(save_dir, 'aminoacids_scores.png')
    plt.savefig(scores_path, dpi=300)
    plt.close()
    print(f"Saved resolved scores to {scores_path}")
    
    # Plot 3: Resolved Molar Absorptivities
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for r in range(3):
        ax.plot(ex_wavelens, aligned_E[:, r], label=names[r], color=colors[r], linewidth=2.5)
    ax.set_title("Resolved Molar Absorptivities (E) - Amino Acids", fontsize=12, fontweight='bold')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorptivity")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    abs_path = os.path.join(save_dir, 'aminoacids_resolved_absorptivities.png')
    plt.savefig(abs_path, dpi=300)
    plt.close()
    print(f"Saved resolved absorptivities to {abs_path}")
    
    # 8. Export CSV files
    comp_cols = ["Tryptophan", "Tyrosine", "Phenylalanine"]
    df_scores = pd.DataFrame(aligned_A, index=[f"Sample_{i+1}" for i in range(num_samples)], columns=comp_cols)
    df_ex_loadings = pd.DataFrame(aligned_B, index=ex_wavelens, columns=comp_cols)
    df_em_loadings = pd.DataFrame(aligned_C, index=em_wavelens, columns=comp_cols)
    df_abs = pd.DataFrame(aligned_E, index=ex_wavelens, columns=comp_cols)
    
    df_scores.to_csv(os.path.join(save_dir, "resolved_scores.csv"))
    df_ex_loadings.to_csv(os.path.join(save_dir, "resolved_excitation_loadings.csv"))
    df_em_loadings.to_csv(os.path.join(save_dir, "resolved_emission_loadings.csv"))
    df_abs.to_csv(os.path.join(save_dir, "resolved_absorptivities.csv"))
    print(f"CSVs exported to: {save_dir}/")
    
    # 9. Write aminoacids_experiment_report.md
    report_content = f"""# EEM-PETN Real-World Amino Acids Experiment Report

## 1. Summary of Recovered Component Concentrations & Peak Locations
Below are the recovery metrics (R² scores) for resolved concentrations and recovered peak wavelengths for the 3 amino acids (Tryptophan, Tyrosine, and Phenylalanine).

| Component | Amino Acid Label | Concentration Recovery (A) R² | Resolved Ex Peak (nm) | Resolved Em Peak (nm) |
|---|---|---|---|---|
"""
    for r in range(3):
        peak_ex = ex_wavelens[np.argmax(aligned_B[:, r])]
        peak_em = em_wavelens[np.argmax(aligned_C[:, r])]
        report_content += f"| **Component {r+1}** | {names[r]} | {r2_scores[r]:.6f} | {peak_ex:.1f} | {peak_em:.1f} |\n"

    report_content += f"""
### Key Averages:
- **Average Concentration Score R² (A):** {avg_r2:.6f}

## 2. Visualization Artifacts
The following plots have been generated and saved to the EEM output folder:
1. **[Resolved Profiles](aminoacids_resolved_profiles.png)**: Visualizes resolved excitation (B) and emission (C) loadings separated by component.
2. **[Scores Recovery](aminoacids_scores.png)**: Bar chart comparing true vs. predicted normalized concentration scores for each component across all samples.
3. **[Resolved Absorptivities](aminoacids_resolved_absorptivities.png)**: Visualizes resolved excitation (E) molar absorptivity curves.
"""
    
    report_path = os.path.join(save_dir, 'aminoacids_experiment_report.md')
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"Diagnostics: Amino Acids Report written to: {report_path}")
    
    return {
        'r2_scores': r2_scores,
        'avg_r2': avg_r2
    }

if __name__ == '__main__':
    train_aminoacids_dataset()

