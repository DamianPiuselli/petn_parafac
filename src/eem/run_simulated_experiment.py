"""
Training and Validation Loops.
Handles dataset preparation, optimizer stepping, constraints projection, and model evaluation.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.eem.generator import EEMGenerator
from src.eem.model import PETNParafac
from src.eem.loss import masked_mse_loss
from src.common.utils import EarlyStopping


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
    scale_factors = []
    for r in range(num_components):
        # Apply optimal linear scale matching on scores to resolve global scaling ambiguity
        # s_r = sum(true * pred) / sum(pred^2)
        true_col = norm_true_A[:, r]
        pred_col = aligned_pred_A[:, r]
        s_r = np.sum(true_col * pred_col) / (np.sum(pred_col ** 2) + 1e-8)
        scaled_pred_col = s_r * pred_col
        aligned_pred_A[:, r] = scaled_pred_col
        scale_factors.append(s_r)
        
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
        'r2_C': r2_C,
        'pred_ind': pred_ind,
        'scale_factors': scale_factors
    }


def train_petn_mvp(epochs=3000, lr=0.008, batch_size=512, seed=43, patience=150, tol=1e-5):
    """
    Runs the full Phase 3 training pipeline:
    1. Generates synthetic EEM data with combined scattering and IFE.
    2. Builds the physical cuvette model with registered background buffers.
    3. Trains on coordinates in full-batch mode.
    4. Evaluates component alignment, spectral recovery, and absorption profile resolution.
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
    E_true = data['E']
    M_true = data['M']
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Pre-flatten coordinate vectors for fast full-batch training
    sample_grid, ex_grid, em_grid = np.meshgrid(
        np.arange(generator.num_samples),
        np.arange(generator.num_ex),
        np.arange(generator.num_em),
        indexing='ij'
    )
    sample_indices = torch.tensor(sample_grid.reshape(-1), dtype=torch.long, device=device)
    ex_indices = torch.tensor(ex_grid.reshape(-1), dtype=torch.long, device=device)
    em_indices = torch.tensor(em_grid.reshape(-1), dtype=torch.long, device=device)
    intensities = torch.tensor(X.reshape(-1), dtype=torch.float32, device=device)
    
    # Broadcast 2D mask to 3D for coordinates
    mask_3d = mask_2d[np.newaxis, :, :].repeat(generator.num_samples, axis=0)
    mask_values = torch.tensor(mask_3d.reshape(-1), dtype=torch.float32, device=device)
    
    # Compute true background CDOM absorbances to register as buffers
    lambda_0 = 240.0
    A_bg_ex = 0.10 * np.exp(-0.010 * (generator.ex_wavelens - lambda_0))
    A_bg_em = 0.10 * np.exp(-0.010 * (generator.em_wavelens - lambda_0))
    
    # 2. Instantiate model
    print("Building PETN model...")
    model = PETNParafac(
        num_samples=generator.num_samples,
        num_ex=generator.num_ex,
        num_em=generator.num_em,
        ex_wavelens=generator.ex_wavelens,
        em_wavelens=generator.em_wavelens,
        ex_bg=A_bg_ex,
        em_bg=A_bg_em,
        num_components=generator.num_components
    ).to(device)
    
    # Simple Adam optimizer for all parameters since they are all physical embeddings
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Training loop
    print(f"Training model in full-batch mode for {epochs} epochs...")
    early_stopping = EarlyStopping(patience=patience, tol=tol, min_epochs=100)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(sample_indices, ex_indices, em_indices)
        loss = masked_mse_loss(predictions, intensities, mask_values)
        
        loss.backward()
        optimizer.step()
        
        # Enforce positive constraints
        model.project_constraints()
        
        loss_val = loss.item()
        if early_stopping(epoch, loss_val, intensities):
            break
            
        if epoch % 300 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{epochs} - Loss: {loss_val:.6f}")
            
    # 4. Extract trained weights
    model.eval()
    with torch.no_grad():
        pred_A, pred_B, pred_C, _, _ = model.get_resolved_factors()
        pred_E, pred_M = model.get_learned_absorptivities()
        pred_ex_bg = model.ex_bg.cpu().numpy() if not model.learnable_bg else model.ex_bg.detach().cpu().numpy()
        pred_em_bg = model.em_bg.cpu().numpy() if not model.learnable_bg else model.em_bg.detach().cpu().numpy()
        
    # 5. Match and align components
    aligned_A, aligned_B, aligned_C, metrics = match_and_align_components(
        true_A, true_B, true_C, pred_A, pred_B, pred_C
    )
    
    # 6. Re-align and scale the molar absorptivity profiles using the same permutation index
    # and inverse scale factors to preserve the absorbance scaling (Abs = A * E)
    pred_ind = metrics['pred_ind']
    s_factors = metrics['scale_factors']
    aligned_E = pred_E[:, pred_ind]
    aligned_M = pred_M[:, pred_ind]
    for r in range(generator.num_components):
        s_r = s_factors[r]
        # pred_ind[r] is the predicted component index matched to true component r
        max_c = np.max(pred_C[:, pred_ind[r]])
        max_c = max_c if max_c > 1e-8 else 1.0
        aligned_E[:, r] = aligned_E[:, r] / (s_r * max_c)
        aligned_M[:, r] = aligned_M[:, r] / (s_r * max_c)
        
    # Compute R^2 metrics for resolved absorptivities
    r2_E = []
    r2_M = []
    for r in range(generator.num_components):
        ss_res_e = np.sum((E_true[:, r] - aligned_E[:, r]) ** 2)
        ss_tot_e = np.sum((E_true[:, r] - np.mean(E_true[:, r])) ** 2)
        r2_e = 1.0 - (ss_res_e / ss_tot_e) if ss_tot_e > 1e-8 else 0.0
        r2_E.append(r2_e)
        
        ss_res_m = np.sum((M_true[:, r] - aligned_M[:, r]) ** 2)
        ss_tot_m = np.sum((M_true[:, r] - np.mean(M_true[:, r])) ** 2)
        r2_m = 1.0 - (ss_res_m / ss_tot_m) if ss_tot_m > 1e-8 else 0.0
        r2_M.append(r2_m)
        
    metrics['r2_E'] = r2_E
    metrics['r2_M'] = r2_M
    
    # 7. Save comparison plots and CSVs
    from src.common.utils import (
        plot_resolved_vs_true_profiles,
        plot_eem_heatmaps,
        plot_resolved_absorptivities,
        plot_scores_comparison
    )
    save_dir = 'notebooks/eem/experiments/simulated'
    os.makedirs(save_dir, exist_ok=True)
    
    fluorophore_names = [f"Fluorophore Component {r+1}" for r in range(generator.num_components)]
    
    plot_resolved_vs_true_profiles(
        true_B, true_C, aligned_B, aligned_C,
        generator.ex_wavelens, generator.em_wavelens,
        component_names=fluorophore_names,
        save_path=os.path.join(save_dir, 'simulated_resolved_profiles.png')
    )
    
    plot_resolved_absorptivities(
        E_true, M_true, aligned_E, aligned_M,
        generator.ex_wavelens, generator.em_wavelens,
        save_path=os.path.join(save_dir, 'simulated_resolved_absorptivities.png')
    )
    
    # Save scores comparison plot
    plot_scores_comparison(
        true_A, aligned_A,
        component_names=fluorophore_names,
        save_path=os.path.join(save_dir, 'simulated_scores.png')
    )
    
    # Calculate reconstructed observed EEM for heatmap display
    # Abs_ex shape: (num_samples, num_ex)
    Abs_ex = np.dot(aligned_A, aligned_E.T) + pred_ex_bg[np.newaxis, :]
    # Abs_em shape: (num_samples, num_em)
    Abs_em = np.dot(aligned_A, aligned_M.T) + pred_em_bg[np.newaxis, :]
    gamma_reconstructed = 10.0 ** (-(Abs_ex[:, :, np.newaxis] + Abs_em[:, np.newaxis, :]))
    X_reconstructed = np.einsum('ir,jr,kr->ijk', aligned_A, aligned_B, aligned_C) * gamma_reconstructed
    
    # Save heatmap plot
    plot_eem_heatmaps(
        X_clean=X_true[0],
        X_corrupted=X[0],
        mask=mask_2d,
        X_reconstructed=X_reconstructed[0],
        ex_wavelens=generator.ex_wavelens,
        em_wavelens=generator.em_wavelens,
        save_path=os.path.join(save_dir, 'simulated_eem_heatmaps.png')
    )
    
    # Save resolved CSV files
    comp_names = [f"Fluorophore_Component_{r+1}" for r in range(generator.num_components)]
    df_A = pd.DataFrame(aligned_A, index=[f"Sample_{i+1}" for i in range(generator.num_samples)], columns=comp_names)
    df_B = pd.DataFrame(aligned_B, index=generator.ex_wavelens, columns=comp_names)
    df_C = pd.DataFrame(aligned_C, index=generator.em_wavelens, columns=comp_names)
    df_E = pd.DataFrame(aligned_E, index=generator.ex_wavelens, columns=comp_names)
    df_M = pd.DataFrame(aligned_M, index=generator.em_wavelens, columns=comp_names)
    
    df_A.to_csv(os.path.join(save_dir, "resolved_scores.csv"))
    df_B.to_csv(os.path.join(save_dir, "resolved_excitation_loadings.csv"))
    df_C.to_csv(os.path.join(save_dir, "resolved_emission_loadings.csv"))
    df_E.to_csv(os.path.join(save_dir, "resolved_excitation_absorptivities.csv"))
    df_M.to_csv(os.path.join(save_dir, "resolved_emission_absorptivities.csv"))
    print(f"CSVs exported to: {save_dir}/")
    
    # Write report.md
    a_sims = [np.corrcoef(true_A[:, r], aligned_A[:, r])[0, 1] for r in range(generator.num_components)]
    b_sims = [np.corrcoef(true_B[:, r], aligned_B[:, r])[0, 1] for r in range(generator.num_components)]
    c_sims = [np.corrcoef(true_C[:, r], aligned_C[:, r])[0, 1] for r in range(generator.num_components)]
    
    report_content = f"""# EEM-PETN Model Calibration & Recovery Report
 
 ## 1. Summary of Recovered Component Loadings & Absorptivities
 Below are the recovery metrics (R² scores and Cosine Similarities) between the ground truth and EEM-PETN resolved profiles for each of the {generator.num_components} chemical components.
 
 | Component | Fluorophore Label | Score (A) R² | Score (A) CosSim | Excitation (B) R² | Excitation (B) CosSim | Emission (C) R² | Emission (C) CosSim | Excitation Abs (E) R² | Emission Abs (M) R² |
 |---|---|---|---|---|---|---|---|---|---|
"""
    for r in range(generator.num_components):
        a_r2 = metrics['r2_A'][r]
        a_sim = a_sims[r]
        b_r2 = metrics['r2_B'][r]
        b_sim = b_sims[r]
        c_r2 = metrics['r2_C'][r]
        c_sim = c_sims[r]
        e_r2 = metrics['r2_E'][r]
        m_r2 = metrics['r2_M'][r]
        lbl = fluorophore_names[r]
        report_content += f"| **Component {r+1}** | {lbl} | {a_r2:.6f} | {a_sim:.6f} | {b_r2:.6f} | {b_sim:.6f} | {c_r2:.6f} | {c_sim:.6f} | {e_r2:.6f} | {m_r2:.6f} |\n"
        
    report_content += f"""
 ### Key Averages:
 - **Average Concentration Score R² (A):** {np.mean(metrics['r2_A']):.6f}
 - **Average Excitation Loading R² (B):** {np.mean(metrics['r2_B']):.6f}
 - **Average Emission Loading R² (C):** {np.mean(metrics['r2_C']):.6f}
 - **Average Excitation Absorptivity R² (E):** {np.mean(metrics['r2_E']):.6f}
 - **Average Emission Absorptivity R² (M):** {np.mean(metrics['r2_M']):.6f}
 
 ## 2. Visualization Artifacts
 The following plots have been generated and saved to the EEM output folder:
 1. **[Resolved Profiles](simulated_resolved_profiles.png)**: Overlays true vs. recovered excitation (B) and emission (C) profiles.
 2. **[Resolved Absorptivities](simulated_resolved_absorptivities.png)**: Overlays true vs. recovered excitation (E) and emission (M) molar absorptivity curves.
 3. **[EEM Heatmaps](simulated_eem_heatmaps.png)**: Visualizes the true EEM, corrupted EEM, scatter mask, and reconstructed EEM.
 4. **[Scores Comparison](simulated_scores.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
 """
    
    report_path = 'notebooks/eem/experiments/simulated/simulated_experiment_report.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"Diagnostics: EEM Report written to: {report_path}")
    
    # 8. Print and return metrics
    print("\n--- Model Evaluation Results ---")
    print(f"Sample scores (A) R2 scores:         {metrics['r2_A']}")
    print(f"Excitation loadings (B) R2 scores:    {metrics['r2_B']}")
    print(f"Emission loadings (C) R2 scores:      {metrics['r2_C']}")
    print(f"Excitation absorptivities (E) R2:     {metrics['r2_E']}")
    print(f"Emission absorptivities (M) R2:       {metrics['r2_M']}")
    
    avg_r2_B = np.mean(metrics['r2_B'])
    avg_r2_C = np.mean(metrics['r2_C'])
    avg_r2_E = np.mean(metrics['r2_E'])
    avg_r2_M = np.mean(metrics['r2_M'])
    print(f"Average Excitation Loading R2:       {avg_r2_B:.4f}")
    print(f"Average Emission Loading R2:         {avg_r2_C:.4f}")
    print(f"Average Excitation Absorptivity R2:  {avg_r2_E:.4f}")
    print(f"Average Emission Absorptivity R2:    {avg_r2_M:.4f}")
    
    return {
        'model': model,
        'generator': generator,
        'aligned_A': aligned_A,
        'aligned_B': aligned_B,
        'aligned_C': aligned_C,
        'aligned_E': aligned_E,
        'aligned_M': aligned_M,
        'metrics': metrics
    }

if __name__ == '__main__':
    train_petn_mvp()
