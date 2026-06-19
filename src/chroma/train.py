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

def train_chroma_petn(dataset, epochs=1200, lr=0.01, warp_reg_coef=0.001, warp_type='linear',
                      num_segments=4, tol=1e-6, patience=50, num_components=3,
                      derivative_order=0, sg_window_size=11, sg_polyorder=2, batch_size=None,
                      compile_model=True):

    """
    Trains the Chroma-PETN model on the provided dataset.
    
    Args:
        dataset: Dictionary containing data matrix 'X' (or raw numpy array) of shape (I, J, K)
        epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
        warp_reg_coef: Weight for warp parameter regularization
        warp_type: Warping model type ('linear', 'quadratic', 'spline')
        num_segments: Number of uniform segments for spline warp type
        tol: Tolerance for relative change in MSE loss to define convergence
        patience: Number of epochs to wait for improvement before early stopping
        num_components: Number of chemical components to resolve (default: 3)
        derivative_order: Savitzky-Golay derivative order (0 for raw, >0 for derivatives)
        sg_window_size: Savitzky-Golay filter window size
        sg_polyorder: Savitzky-Golay polynomial order
        batch_size: If specified, chunk coordinate evaluation to prevent OOM
        compile_model: If True, compile the model graph using torch.compile
        
    Returns:
        model: Trained ChromaPETN model instance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    if isinstance(dataset, dict):
        if 'X' not in dataset:
            raise KeyError("dataset dictionary must contain key 'X'")
        X_np = np.array(dataset['X'], dtype=np.float32)
    else:
        X_np = np.array(dataset, dtype=np.float32)
        
    if X_np.ndim != 3:
        raise ValueError(f"Input data must be 3-dimensional (shape: I x J x K), got {X_np.ndim}D")
        
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    I, J, K = X.shape
    if num_components < 1:
        raise ValueError(f"num_components must be >= 1, got {num_components}")
    
    # Instantiate model
    model = ChromaPETN(
        num_samples=I, 
        num_time=J, 
        num_spec=K, 
        num_components=num_components, 
        warp_type=warp_type, 
        num_segments=num_segments,
        derivative_order=derivative_order,
        sg_window_size=sg_window_size,
        sg_polyorder=sg_polyorder
    ).to(device)

    # 3. Model Compilation Optimization (optional)
    if compile_model:
        if hasattr(torch, 'compile'):
            try:
                print("Compiling model graph using torch.compile...")
                model = torch.compile(model)
            except Exception as e:
                print(f"Model compilation failed: {e}. Falling back to uncompiled model.")
        else:
            print("torch.compile is not supported on this PyTorch version. Using uncompiled model.")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if batch_size is None:
        # Full-grid training mode (Optimization 2)
        print(f"Training Chroma-PETN model ({warp_type} warp, R={num_components}) in grid-based mode on {device}...")
        
        # Prepare targets
        if derivative_order > 0:
            from scipy.signal import savgol_filter
            X_deriv = savgol_filter(X_np, window_length=sg_window_size, polyorder=sg_polyorder, deriv=derivative_order, axis=1)
            y_target = torch.tensor(X_deriv, dtype=torch.float32, device=device)
        else:
            y_target = X
            
        y_target_var = torch.var(y_target).item()
        
        t_grid = torch.linspace(0.0, 1.0, J, device=device)
        m = sg_window_size // 2
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 1. Warp time coordinates per sample: shape (I, J)
            if model.warp_type == 'linear':
                stretch = model.warp_stretch.unsqueeze(-1)
                shift = model.warp_shift.unsqueeze(-1)
                t_warped = t_grid.unsqueeze(0) - (stretch * t_grid.unsqueeze(0) + shift)
            elif model.warp_type == 'quadratic':
                alpha = model.warp_alpha.unsqueeze(-1)
                beta = model.warp_beta.unsqueeze(-1)
                gamma = model.warp_gamma.unsqueeze(-1)
                t_warped = t_grid.unsqueeze(0) - (alpha * (t_grid.unsqueeze(0)**2) + beta * t_grid.unsqueeze(0) + gamma)
            elif model.warp_type == 'spline':
                shift = model.warp_shift.unsqueeze(-1)
                inc = (1.0 / model.num_segments) * torch.exp(model.warp_log_increments)
                zeros = torch.zeros((I, 1), device=device, dtype=inc.dtype)
                cum_inc = torch.cumsum(torch.cat([zeros, inc], dim=1), dim=1)
                w = shift + cum_inc
                
                val = t_grid * model.num_segments
                k = torch.clamp(torch.floor(val).long(), 0, model.num_segments - 1)
                u = val - k.float()
                
                w_k = w[:, k]
                w_kp1 = w[:, k + 1]
                t_warped = (1.0 - u.unsqueeze(0)) * w_k + u.unsqueeze(0) * w_kp1
                
            # 2. Differentiable 1D Linear Interpolation over canonical B
            x_warped = t_warped * (J - 1)
            x_clamped = torch.clamp(x_warped, 0.0, J - 1.0 - 1e-3)
            x_0 = torch.floor(x_clamped).long()
            x_1 = x_0 + 1
            
            w_interp = (x_clamped - x_0.float()).unsqueeze(-1) # (I, J, 1)
            
            B_weights = model.time_embeddings.weight
            val_0 = B_weights[x_0] # (I, J, R)
            val_1 = B_weights[x_1] # (I, J, R)
            b_warped = (1.0 - w_interp) * val_0 + w_interp * val_1 # (I, J, R)
            
            # 3. Reconstruct raw predicted intensities Y_pred: shape (I, J, K)
            A_weights = model.sample_embeddings.weight
            C_weights = model.spec_embeddings.weight
            Y_pred = torch.einsum('ir,ijr,kr->ijk', A_weights, b_warped, C_weights)
            
            # 4. Apply Savitzky-Golay derivative along time axis if needed
            if derivative_order > 0:
                y_raw_window = Y_pred.transpose(1, 2).reshape(I * K, 1, J)
                y_padded = torch.nn.functional.pad(y_raw_window, (m, m), mode='replicate')
                y_deriv = torch.nn.functional.conv1d(y_padded, model.sg_kernel, padding=0)
                Y_pred = y_deriv.view(I, K, J).transpose(1, 2)
                
            loss_mse = nn.functional.mse_loss(Y_pred, y_target)
            
            # Warp parameter regularization
            if model.warp_type == 'linear':
                loss_warp_reg = warp_reg_coef * (torch.mean(model.warp_stretch**2) + torch.mean(model.warp_shift**2))
            elif model.warp_type == 'quadratic':
                loss_warp_reg = warp_reg_coef * (torch.mean(model.warp_alpha**2) + torch.mean(model.warp_beta**2) + torch.mean(model.warp_gamma**2))
            elif model.warp_type == 'spline':
                loss_warp_reg = warp_reg_coef * (torch.mean(model.warp_shift**2) + torch.mean(model.warp_log_increments**2))
                
            loss = loss_mse + loss_warp_reg
            loss.backward()
            optimizer.step()
            model.project_constraints()
            
            # Convergence checks
            loss_val = loss_mse.item()
            if loss_val < 1e-7 or loss_val < 1e-5 * y_target_var:
                print(f"Convergence reached at epoch {epoch:4d} (MSE Loss < target threshold). Final MSE: {loss_val:.3e}")
                break
                
            if epoch > 0:
                change_abs = best_loss - loss_val
                change_rel = change_abs / (best_loss + 1e-10)
                
                if change_rel > tol and change_abs > tol * y_target_var:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch:4d} (MSE did not decrease significantly for {patience} epochs). Final MSE: {loss_val:.3e}")
                    break
            else:
                best_loss = loss_val
                
            if (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1:4d}/{epochs} | MSE Loss: {loss_val:.3e} | Reg: {loss_warp_reg.item():.3e}")
                
        return model
    else:
        # Coordinate-based batched training mode (Optimization 1)
        print(f"Training Chroma-PETN model ({warp_type} warp, R={num_components}) in coordinate-based mode (batch_size={batch_size}) on {device}...")
        
        # Generate complete coordinate triplets
        coords_i, coords_j, coords_k = torch.meshgrid(
            torch.arange(I, device=device), torch.arange(J, device=device), torch.arange(K, device=device), indexing='ij'
        )
        coords_i = coords_i.flatten()
        coords_j = coords_j.flatten()
        coords_k = coords_k.flatten()
        
        if derivative_order > 0:
            from scipy.signal import savgol_filter
            X_deriv = savgol_filter(X_np, window_length=sg_window_size, polyorder=sg_polyorder, deriv=derivative_order, axis=1)
            y_target = torch.tensor(X_deriv, dtype=torch.float32, device=device)[coords_i, coords_j, coords_k]
        else:
            y_target = X[coords_i, coords_j, coords_k]
            
        y_target_var = torch.var(y_target).item()
        num_coords = coords_i.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Accumulate MSE loss and backprop in chunks if batching is used
            loss_mse_val = 0.0
            for start_idx in range(0, num_coords, batch_size):
                end_idx = min(start_idx + batch_size, num_coords)
                batch_i = coords_i[start_idx:end_idx]
                batch_j = coords_j[start_idx:end_idx]
                batch_k = coords_k[start_idx:end_idx]
                
                y_pred_batch = model(batch_i, batch_j, batch_k)
                y_target_batch = y_target[start_idx:end_idx]
                
                batch_loss = nn.functional.mse_loss(y_pred_batch, y_target_batch) * (len(batch_i) / num_coords)
                batch_loss.backward()
                loss_mse_val += batch_loss.item()
                
            # Add regularization loss and backward
            if model.warp_type == 'linear':
                loss_warp_reg = warp_reg_coef * (torch.mean(model.warp_stretch**2) + torch.mean(model.warp_shift**2))
            elif model.warp_type == 'quadratic':
                loss_warp_reg = warp_reg_coef * (torch.mean(model.warp_alpha**2) + torch.mean(model.warp_beta**2) + torch.mean(model.warp_gamma**2))
            elif model.warp_type == 'spline':
                loss_warp_reg = warp_reg_coef * (torch.mean(model.warp_shift**2) + torch.mean(model.warp_log_increments**2))
                
            loss_warp_reg.backward()
            optimizer.step()
            model.project_constraints()
            
            # Convergence checks
            if loss_mse_val < 1e-7 or loss_mse_val < 1e-5 * y_target_var:
                print(f"Convergence reached at epoch {epoch:4d} (MSE Loss < target threshold). Final MSE: {loss_mse_val:.3e}")
                break
                
            if epoch > 0:
                change_abs = best_loss - loss_mse_val
                change_rel = change_abs / (best_loss + 1e-10)
                
                if change_rel > tol and change_abs > tol * y_target_var:
                    best_loss = loss_mse_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch:4d} (MSE did not decrease significantly for {patience} epochs). Final MSE: {loss_mse_val:.3e}")
                    break
            else:
                best_loss = loss_mse_val
                
            if (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1:4d}/{epochs} | MSE Loss: {loss_mse_val:.3e} | Reg: {loss_warp_reg.item():.3e}")
                
        return model

def calculate_cosine_similarity(v1, v2):
    """Calculates cosine similarity (correlation coefficient) between two vectors."""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    return np.max([np.dot(v1_norm, v2_norm), np.dot(v1_norm, -v2_norm)])

def evaluate_chroma_alignment(model, dataset, save_dir='notebooks/chroma'):
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
    
    I, J, K = dataset['X'].shape
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
    
    # Evaluate generalized coordinate alignment error across all samples
    t_observed = np.linspace(0.0, 1.0, J)
    mae_list = []
    for i in range(I):
        # True warped time
        ds_warp_type = dataset.get('warp_type', 'linear')
        if ds_warp_type == 'linear':
            t_true = (t_observed - dataset['shifts'][i]) / (1.0 + dataset['stretches'][i])
        elif ds_warp_type == 'quadratic':
            alpha = dataset['alphas'][i]
            beta = dataset['betas'][i]
            gamma = dataset['gammas'][i]
            t_true = t_observed - (alpha * (t_observed ** 2) + beta * t_observed + gamma)
        elif ds_warp_type == 'spline':
            shift = dataset['shifts'][i]
            stretch = dataset['stretches'][i] * 0.5
            t_true = t_observed - (shift + stretch * np.sin(np.pi * t_observed))
        else:
            t_true = (t_observed - dataset['shifts'][i]) / (1.0 + dataset['stretches'][i])
        
        # Predicted warped time
        if model.warp_type == 'linear':
            stretch_pred = model.warp_stretch[i].item()
            shift_pred = model.warp_shift[i].item()
            t_pred = t_observed - (stretch_pred * t_observed + shift_pred)
        elif model.warp_type == 'quadratic':
            alpha_pred = model.warp_alpha[i].item()
            beta_pred = model.warp_beta[i].item()
            gamma_pred = model.warp_gamma[i].item()
            t_pred = t_observed - (alpha_pred * (t_observed**2) + beta_pred * t_observed + gamma_pred)
        elif model.warp_type == 'spline':
            shift_pred = model.warp_shift[i].item()
            log_inc_pred = model.warp_log_increments[i].detach().cpu().numpy()
            inc_pred = (1.0 / model.num_segments) * np.exp(log_inc_pred)
            w_pred = shift_pred + np.cumsum(np.concatenate([[0.0], inc_pred]))
            
            # Interpolate to find t_pred
            val = t_observed * model.num_segments
            k = np.clip(np.floor(val).astype(int), 0, model.num_segments - 1)
            u = val - k
            t_pred = (1.0 - u) * w_pred[k] + u * w_pred[k + 1]
            
        t_pred_centered = t_pred - t_pred.mean()
        t_true_centered = t_true - t_true.mean()
        mae_list.append(np.mean(np.abs(t_pred_centered - t_true_centered)))
        
    mean_coord_mae = np.mean(mae_list)
    
    # Evaluate shifts for linear warp
    shift_corr = 0.0
    stretch_corr = 0.0
    mean_shift_error = mean_coord_mae
    
    if model.warp_type == 'linear':
        shifts_pred = model.warp_shift.detach().cpu().numpy()
        stretches_pred = model.warp_stretch.detach().cpu().numpy()
        
        shifts_pred_centered = shifts_pred - shifts_pred.mean()
        stretches_pred_centered = stretches_pred - stretches_pred.mean()
        
        shifts_true_mapped = dataset['shifts'] / (1.0 + dataset['stretches'])
        stretches_true_mapped = dataset['stretches'] / (1.0 + dataset['stretches'])
        
        shifts_true_centered = shifts_true_mapped - shifts_true_mapped.mean()
        stretches_true_centered = stretches_true_mapped - stretches_true_mapped.mean()
        
        shift_corr = np.corrcoef(shifts_pred_centered, shifts_true_centered)[0, 1]
        stretch_corr = np.corrcoef(stretches_pred_centered, stretches_true_centered)[0, 1]
        mean_shift_error = np.mean(np.abs(shifts_pred_centered - shifts_true_centered))
        
    # Calculate the fully aligned (unshifted) reconstructed tensor
    X_aligned = np.einsum('ir,jr,kr->ijk', A_pred_ordered, B_pred_ordered, C_pred_ordered)
    
    metrics = {
        'b_similarities': b_sims,
        'c_similarities': c_sims,
        'a_similarities': a_sims,
        'shift_correlation': shift_corr,
        'stretch_correlation': stretch_corr,
        'mean_shift_error': mean_shift_error,
        'mean_coordinate_error': mean_coord_mae
    }
    
    print("\n--- Chroma-PETN Model Recovery Evaluation ---")
    for r in range(R):
        print(f"Component {r+1}:")
        print(f"  Chromatography profile similarity: {b_sims[r]:.4f}")
        print(f"  Spectral profile similarity:       {c_sims[r]:.4f}")
        print(f"  Concentration score similarity:    {a_sims[r]:.4f}")
        
    print("\n--- Shift & Alignment Recovery ---")
    if model.warp_type == 'linear':
        print(f"  Shift parameter correlation:  {shift_corr:.4f}")
        print(f"  Stretch parameter correlation: {stretch_corr:.4f}")
        print(f"  Mean absolute shift error:    {mean_shift_error:.4f} (normalized time units)")
    else:
        print(f"  Warp Model:                   {model.warp_type}")
        print(f"  Mean absolute coordinate error: {mean_coord_mae:.4f} (normalized time units)")
    
    if save_dir:
        # Generate and save comparison plots in save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        from src.common.utils import (
            plot_chroma_resolved_vs_true_profiles,
            plot_chroma_alignment_comparison,
            plot_chroma_warp_parameters
        )
        
        time_grid = np.linspace(0.0, 1.0, B_true.shape[0])
        spec_grid = np.linspace(200.0, 400.0, C_true.shape[0])
        
        synthetic_names = [
            "Component 1 (Baseline Interference)",
            "Component 2 (Peak 1)",
            "Component 3 (Peak 2)"
        ]

        # 1. Resolved profiles comparison
        plot_chroma_resolved_vs_true_profiles(
            B_true, C_true, B_pred_ordered, C_pred_ordered,
            time_grid, spec_grid, component_names=synthetic_names,
            plot_type='dad',
            save_path=os.path.join(save_dir, 'chroma_resolved_profiles.png')
        )
        
        # 2. Alignment comparison (Observed vs Aligned Chromatograms)
        plot_chroma_alignment_comparison(
            time_grid, dataset['X'], X_aligned,
            save_path=os.path.join(save_dir, 'chroma_alignment_comparison.png')
        )
        
        # 3. Warp parameters recovery plot (only for linear warp model)
        if model.warp_type == 'linear':
            plot_chroma_warp_parameters(
                shifts_true_centered, stretches_true_centered, shifts_pred_centered, stretches_pred_centered,
                save_path=os.path.join(save_dir, 'chroma_warp_parameters.png')
            )
        else:
            print("  Warp parameter plotting skipped for non-linear warp models.")

    
    return metrics

if __name__ == "__main__":
    print("Generating synthetic chromatographic data...")
    generator = ChromatographicDataGenerator(num_samples=15, num_time=100, num_spec=80, num_components=3)
    dataset = generator.generate_dataset(noise_std=0.015, max_shift=0.05, max_stretch=0.08)
    
    # Run with default linear warp
    model = train_chroma_petn(dataset, epochs=1200, lr=0.01)
    evaluate_chroma_alignment(model, dataset)
