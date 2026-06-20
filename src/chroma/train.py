"""
Training and Evaluation script for Chroma-PETN.
Implements the training loop, parameter updates, alignment projection, and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.chroma.hplc import HPLC_PETN
from src.chroma.gcms import GCMS_PETN
from src.chroma.generator import ChromatographicDataGenerator
from src.common.utils import EarlyStopping


def train_chroma_petn(dataset, epochs=1200, lr=0.01, warp_reg_coef=0.001, warp_type='linear',
                      num_segments=4, tol=1e-6, patience=50, num_components=3,
                      derivative_order=0, sg_window_size=11, sg_polyorder=2, batch_size=None,
                      compile_model=True, threshold=None, lambda_res=10.0, lambda_c=1e-4,
                      lambda_raw=0.0, lambda_smooth_B=0.0, model_type=None, init_svd=True):
    """
    Trains the Chroma-PETN model on the provided dataset.
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
    
    # Resolve model type if not explicitly set
    if model_type is None:
        model_type = 'hplc' if derivative_order > 0 else 'gcms'
    
    # Instantiate specific model subclass
    if model_type == 'hplc':
        model = HPLC_PETN(
            num_samples=I,
            num_time=J,
            num_spec=K,
            num_components=num_components,
            warp_type=warp_type,
            num_segments=num_segments,
            derivative_order=derivative_order,
            sg_window_size=sg_window_size,
            sg_polyorder=sg_polyorder,
            sample_specific_baseline=True
        ).to(device)
    elif model_type == 'gcms':
        model = GCMS_PETN(
            num_samples=I,
            num_time=J,
            num_spec=K,
            num_components=num_components,
            warp_type=warp_type,
            num_segments=num_segments,
            lambda_c=lambda_c,
            lambda_res=lambda_res
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Warm-start embedding tables using Truncated SVD if enabled
    if init_svd:
        print("Warm-starting embedding tables using Truncated SVD...")
        model.init_from_svd(X)

    # Compile model graph
    if compile_model:
        if hasattr(torch, 'compile'):
            try:
                print("Compiling model graph using torch.compile...")
                model = torch.compile(model)
            except Exception as e:
                print(f"Model compilation failed: {e}. Falling back to uncompiled model.")
        else:
            print("torch.compile is not supported. Using uncompiled model.")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Get raw model for checking non-compiled attributes
    raw_model = getattr(model, '_orig_mod', model)

    if batch_size is None:
        # Full-grid training mode
        print(f"Training Chroma-PETN model ({warp_type} warp, R={num_components}) in grid-based mode on {device}...")
        
        # Prepare target
        if derivative_order > 0:
            from scipy.signal import savgol_filter
            X_deriv = savgol_filter(X_np, window_length=sg_window_size, polyorder=sg_polyorder, deriv=derivative_order, axis=1)
            y_target = torch.tensor(X_deriv, dtype=torch.float32, device=device)
        else:
            y_target = X
            
        early_stopping = EarlyStopping(patience=patience, tol=tol, min_epochs=50)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            Y_pred = model.forward_grid()
            
            # Subclass loss function
            loss_physics = raw_model.calculate_loss(Y_pred, y_target)
            
            # Warp parameter regularization
            loss_warp_reg = 0.0
            if warp_reg_coef > 0.0:
                if raw_model.warp_type == 'linear':
                    loss_warp_reg = warp_reg_coef * (torch.mean(raw_model.alpha**2) + torch.mean(raw_model.beta**2))
                elif raw_model.warp_type == 'quadratic':
                    loss_warp_reg = warp_reg_coef * (torch.mean(raw_model.alpha**2) + torch.mean(raw_model.beta**2) + torch.mean(raw_model.gamma**2))
                elif raw_model.warp_type == 'spline':
                    loss_warp_reg = warp_reg_coef * (torch.mean(raw_model.beta**2) + torch.mean(raw_model.log_increments**2))
            
            # Smoothness penalty on B
            loss_smooth = 0.0
            if lambda_smooth_B > 0.0:
                diff1 = raw_model.B[1:] - raw_model.B[:-1]
                diff2 = diff1[1:] - diff1[:-1]
                loss_smooth = lambda_smooth_B * torch.mean(diff2 ** 2)
                
            # Raw loss term
            loss_raw_term = 0.0
            if lambda_raw > 0.0:
                A_p, B_warped_p, C_p = raw_model._forward_raw_grid()
                Y_pred_raw = torch.einsum('ir,ijr,kr->ijk', A_p, B_warped_p, C_p)
                
                # Reconstruct baseline
                t_grid = torch.linspace(0.0, 1.0, J, device=device).view(1, -1)
                if raw_model.sample_specific_baseline:
                    poly = (raw_model.baseline_offset.unsqueeze(1) + 
                            raw_model.baseline_slope.unsqueeze(1) * t_grid + 
                            raw_model.baseline_quadratic.unsqueeze(1) * (t_grid ** 2))
                    baseline = torch.einsum('ij,k->ijk', poly, raw_model.solvent_spectrum)
                else:
                    t_grid_3d = t_grid.unsqueeze(-1)
                    baseline = (raw_model.baseline_offset.view(1, 1, -1) + 
                                raw_model.baseline_slope.view(1, 1, -1) * t_grid_3d + 
                                raw_model.baseline_quadratic.view(1, 1, -1) * (t_grid_3d ** 2))
                Y_pred_raw = Y_pred_raw + baseline
                loss_raw_term = lambda_raw * torch.nn.functional.mse_loss(Y_pred_raw, X)
                
            loss = loss_physics + loss_warp_reg + loss_smooth + loss_raw_term
            loss.backward()
            optimizer.step()
            raw_model.project_constraints()
            
            # Convergence check using reusable EarlyStopping on main target loss
            loss_val = loss_physics.item()
            if early_stopping(epoch, loss_val, y_target):
                break
                
            if (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1:4d}/{epochs} | Model Loss: {loss_val:.3e} | Raw Loss: {loss_raw_term.item() if isinstance(loss_raw_term, torch.Tensor) else loss_raw_term:.3e} | Warp Reg: {loss_warp_reg.item() if isinstance(loss_warp_reg, torch.Tensor) else loss_warp_reg:.3e}")
                
        return model
    else:
        # Coordinate-based batched training mode
        print(f"Training Chroma-PETN model ({warp_type} warp, R={num_components}) in coordinate-based mode (batch_size={batch_size}) on {device}...")
        
        # Prepare target signal block on CPU
        if derivative_order > 0:
            from scipy.signal import savgol_filter
            X_deriv = savgol_filter(X_np, window_length=sg_window_size, polyorder=sg_polyorder, deriv=derivative_order, axis=1)
            X_target = torch.tensor(X_deriv, dtype=torch.float32)
        else:
            X_target = torch.tensor(X_np, dtype=torch.float32)
            
        # Get DataLoader
        from src.chroma.dataset import get_chroma_dataloader
        use_pin = device.type == 'cuda'
        dataloader, dataset_coo = get_chroma_dataloader(
            X=X_target, 
            batch_size=batch_size, 
            shuffle=True, 
            threshold=threshold, 
            pin_memory=use_pin
        )

        
        early_stopping = EarlyStopping(patience=patience, tol=tol, min_epochs=50)
        
        for epoch in range(epochs):
            loss_physics_val = 0.0
            for batch_coords, batch_targets in dataloader:
                optimizer.zero_grad()
                
                batch_coords = batch_coords.to(device)
                batch_targets = batch_targets.to(device)
                
                batch_i = batch_coords[:, 0]
                batch_j = batch_coords[:, 1]
                batch_k = batch_coords[:, 2]
                
                y_pred_batch = model(batch_i, batch_j, batch_k)
                
                # Scaled by batch size relative to total size to maintain consistent gradient scale
                batch_loss = raw_model.calculate_loss(y_pred_batch, batch_targets) * (len(batch_coords) / len(dataset_coo))
                
                # Regularize warp parameters
                loss_warp_reg = 0.0
                if warp_reg_coef > 0.0:
                    if raw_model.warp_type == 'linear':
                        loss_warp_reg = warp_reg_coef * (torch.mean(raw_model.alpha**2) + torch.mean(raw_model.beta**2))
                    elif raw_model.warp_type == 'quadratic':
                        loss_warp_reg = warp_reg_coef * (torch.mean(raw_model.alpha**2) + torch.mean(raw_model.beta**2) + torch.mean(raw_model.gamma**2))
                    elif raw_model.warp_type == 'spline':
                        loss_warp_reg = warp_reg_coef * (torch.mean(raw_model.beta**2) + torch.mean(raw_model.log_increments**2))
                    loss_warp_reg = loss_warp_reg * (len(batch_coords) / len(dataset_coo))
                    
                # Smoothness penalty on B
                loss_smooth = 0.0
                if lambda_smooth_B > 0.0:
                    diff1 = raw_model.B[1:] - raw_model.B[:-1]
                    diff2 = diff1[1:] - diff1[:-1]
                    loss_smooth = lambda_smooth_B * torch.mean(diff2 ** 2) * (len(batch_coords) / len(dataset_coo))
                    
                # Raw loss term
                loss_raw_term = 0.0
                if lambda_raw > 0.0:
                    a, b_warped, c = raw_model._forward_raw_coo(batch_i, batch_j, batch_k)
                    y_pred_raw_batch = torch.sum(a * b_warped * c, dim=1)
                    
                    # Add baseline
                    t_batch = batch_j.float() / (J - 1)
                    if raw_model.sample_specific_baseline:
                        poly_batch = (raw_model.baseline_offset[batch_i] + 
                                      raw_model.baseline_slope[batch_i] * t_batch + 
                                      raw_model.baseline_quadratic[batch_i] * (t_batch ** 2))
                        baseline_batch = poly_batch * raw_model.solvent_spectrum[batch_k]
                    else:
                        baseline_batch = (raw_model.baseline_offset[batch_k] + 
                                          raw_model.baseline_slope[batch_k] * t_batch + 
                                          raw_model.baseline_quadratic[batch_k] * (t_batch ** 2))
                    y_pred_raw_batch = y_pred_raw_batch + baseline_batch
                    
                    batch_targets_raw = X[batch_i, batch_j, batch_k]
                    loss_raw_batch = torch.nn.functional.mse_loss(y_pred_raw_batch, batch_targets_raw) * (len(batch_coords) / len(dataset_coo))
                    loss_raw_term = lambda_raw * loss_raw_batch
                    
                total_loss = batch_loss + loss_warp_reg + loss_smooth + loss_raw_term
                total_loss.backward()
                optimizer.step()
                raw_model.project_constraints()
                
                loss_physics_val += batch_loss.item()
            
            # Convergence check using reusable EarlyStopping
            if early_stopping(epoch, loss_physics_val, dataset_coo.targets):
                break
                
            if (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1:4d}/{epochs} | Model Loss: {loss_physics_val:.3e}")
                
        return model


def calculate_cosine_similarity(v1, v2):
    """Calculates cosine similarity (correlation coefficient) between two vectors."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    return np.max([np.dot(v1_norm, v2_norm), np.dot(v1_norm, -v2_norm)])

def evaluate_chroma_alignment(model, dataset, save_dir='notebooks/chroma'):
    """
    Evaluates profile recovery and shift recovery.
    Resolves permutation/scaling ambiguities and prints correlation metrics.
    """
    from src.chroma.plots import extract_loadings
    loadings = extract_loadings(model)
    A_pred, B_pred, C_pred = loadings['A'], loadings['B'], loadings['C']
    
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
        
    # Scale ambiguity resolution
    for r in range(R):
        norm_b = np.linalg.norm(B_pred_ordered[:, r]) + 1e-10
        norm_c = np.linalg.norm(C_pred_ordered[:, r]) + 1e-10
        B_pred_ordered[:, r] /= norm_b
        C_pred_ordered[:, r] /= norm_c
        A_pred_ordered[:, r] *= (norm_b * norm_c)
        
        norm_b_true = np.linalg.norm(B_true[:, r]) + 1e-10
        norm_c_true = np.linalg.norm(C_true[:, r]) + 1e-10
        B_true[:, r] /= norm_b_true
        C_true[:, r] /= norm_c_true
        A_true[:, r] *= (norm_b_true * norm_c_true)
        
    # Calculate recovery metrics
    b_sims = [calculate_cosine_similarity(B_pred_ordered[:, r], B_true[:, r]) for r in range(R)]
    c_sims = [calculate_cosine_similarity(C_pred_ordered[:, r], C_true[:, r]) for r in range(R)]
    a_sims = [calculate_cosine_similarity(A_pred_ordered[:, r], A_true[:, r]) for r in range(R)]
    
    # Evaluate coordinate alignment error across all samples
    t_observed = np.linspace(0.0, 1.0, J)
    mae_list = []
    
    raw_model = getattr(model, '_orig_mod', model)
    
    for i in range(I):
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
        
        # Predicted warped time (mean over components)
        if raw_model.warp_type == 'linear':
            stretch_pred = raw_model.alpha[i].mean().item()
            shift_pred = raw_model.beta[i].mean().item()
            t_pred = t_observed - (stretch_pred * t_observed + shift_pred)
        elif raw_model.warp_type == 'quadratic':
            alpha_pred = raw_model.alpha[i].mean().item()
            beta_pred = raw_model.beta[i].mean().item()
            gamma_pred = raw_model.gamma[i].mean().item()
            t_pred = t_observed - (alpha_pred * (t_observed**2) + beta_pred * t_observed + gamma_pred)
        elif raw_model.warp_type == 'spline':
            shift_pred = raw_model.beta[i].mean().item()
            log_inc_pred = raw_model.log_increments[i].mean(dim=-1).detach().cpu().numpy()
            inc_pred = (1.0 / raw_model.num_segments) * np.exp(log_inc_pred)
            w_pred = shift_pred + np.cumsum(np.concatenate([[0.0], inc_pred]))
            
            val = t_observed * raw_model.num_segments
            k = np.clip(np.floor(val).astype(int), 0, raw_model.num_segments - 1)
            u = val - k
            t_pred = (1.0 - u) * w_pred[k] + u * w_pred[k + 1]
            
        t_pred_centered = t_pred - t_pred.mean()
        t_true_centered = t_true - t_true.mean()
        mae_list.append(np.mean(np.abs(t_pred_centered - t_true_centered)))
        
    mean_coord_mae = np.mean(mae_list)
    
    shift_corr = 0.0
    stretch_corr = 0.0
    mean_shift_error = mean_coord_mae
    
    if raw_model.warp_type == 'linear':
        shifts_pred = raw_model.beta.mean(dim=-1).detach().cpu().numpy()
        stretches_pred = raw_model.alpha.mean(dim=-1).detach().cpu().numpy()
        
        shifts_pred_centered = shifts_pred - shifts_pred.mean()
        stretches_pred_centered = stretches_pred - stretches_pred.mean()
        
        shifts_true_mapped = dataset['shifts'] / (1.0 + dataset['stretches'])
        stretches_true_mapped = dataset['stretches'] / (1.0 + dataset['stretches'])
        
        shifts_true_centered = shifts_true_mapped - shifts_true_mapped.mean()
        stretches_true_centered = stretches_true_mapped - stretches_true_mapped.mean()
        
        shift_corr = np.corrcoef(shifts_pred_centered, shifts_true_centered)[0, 1]
        stretch_corr = np.corrcoef(stretches_pred_centered, stretches_true_centered)[0, 1]
        mean_shift_error = np.mean(np.abs(shifts_pred_centered - shifts_true_centered))
        
    # Calculate fully aligned reconstructed tensor
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
    if raw_model.warp_type == 'linear':
        print(f"  Shift parameter correlation:  {shift_corr:.4f}")
        print(f"  Stretch parameter correlation: {stretch_corr:.4f}")
        print(f"  Mean absolute shift error:    {mean_shift_error:.4f} (normalized time units)")
    else:
        print(f"  Warp Model:                   {raw_model.warp_type}")
        print(f"  Mean absolute coordinate error: {mean_coord_mae:.4f} (normalized time units)")
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        from src.common.utils import (
            plot_chroma_resolved_vs_true_profiles,
            plot_chroma_alignment_comparison,
            plot_chroma_warp_parameters,
            plot_scores_comparison
        )
        
        time_grid = np.linspace(0.0, 1.0, B_true.shape[0])
        spec_grid = np.linspace(200.0, 400.0, C_true.shape[0])
        
        is_gcms = 'GCMS' in raw_model.__class__.__name__
        plot_type = 'ms' if is_gcms else 'dad'
        
        if is_gcms:
            synthetic_names = [f"Analyte Component {r+1}" for r in range(R)]
        else:
            synthetic_names = [
                "Component 1 (Baseline Interference)",
                "Component 2 (Peak 1)",
                "Component 3 (Peak 2)"
            ] if R == 3 else [f"Component {r+1}" for r in range(R)]

        plot_chroma_resolved_vs_true_profiles(
            B_true, C_true, B_pred_ordered, C_pred_ordered,
            time_grid, spec_grid, component_names=synthetic_names,
            plot_type=plot_type,
            save_path=os.path.join(save_dir, 'chroma_resolved_profiles.png')
        )
        
        plot_chroma_alignment_comparison(
            time_grid, dataset['X'], X_aligned,
            save_path=os.path.join(save_dir, 'chroma_alignment_comparison.png')
        )
        
        # Save scores comparison plot
        plot_scores_comparison(
            A_true, A_pred_ordered,
            component_names=synthetic_names,
            save_path=os.path.join(save_dir, 'scores_comparison.png')
        )
        
        if raw_model.warp_type == 'linear':
            plot_chroma_warp_parameters(
                shifts_true_centered, stretches_true_centered, shifts_pred_centered, stretches_pred_centered,
                save_path=os.path.join(save_dir, 'chroma_warp_parameters.png')
            )
            
        # Write report.md
        model_name = "GCMS-PETN" if is_gcms else "HPLC-PETN"
        report_content = f"""# {model_name} Model Calibration & Recovery Report

## 1. Summary of Recovered Component Loadings
Below are the cosine similarities (correlation coefficients) between the ground truth and PETN-resolved profiles for each of the {R} chemical components.

| Component | Component Label | Concentration Score (A) | Chromatography Profile (B) | Spectral Profile (C) |
|---|---|---|---|---|
"""
        for r in range(R):
            a_sim = a_sims[r]
            b_sim = b_sims[r]
            c_sim = c_sims[r]
            lbl = synthetic_names[r]
            report_content += f"| **Component {r+1}** | {lbl} | {a_sim:.6f} | {b_sim:.6f} | {c_sim:.6f} |\n"
            
        report_content += f"""
### Key Averages:
- **Mean Score Similarity:** {np.mean(a_sims):.6f}
- **Mean Elution Profile Similarity:** {np.mean(b_sims):.6f}
- **Mean Spectral Profile Similarity:** {np.mean(c_sims):.6f}

## 2. Retention Time Warping & Alignment Performance
The warping head aligns sample-specific shifting and stretching to the canonical time grid.

- **Mean Coordinate Alignment Error (MAE):** {mean_coord_mae:.6e} (normalized time units)
"""
        if raw_model.warp_type == 'linear':
            report_content += f"- **Shift Parameter (beta) Correlation:** {shift_corr:.6f}\n"
            report_content += f"- **Stretch Parameter (alpha) Correlation:** {stretch_corr:.6f}\n"
            report_content += f"- **Mean Absolute Shift Parameter Error:** {mean_shift_error:.6e}\n"
            
        report_content += """
## 3. Visualization Artifacts
The following plots have been generated and saved to the experiment folder:
1. **[Resolved Profiles](chroma_resolved_profiles.png)**: Overlays true vs. recovered elution profiles (B) and absorption/mass spectra (C).
2. **[Alignment Comparison](chroma_alignment_comparison.png)**: Shows total elution profiles (TIC) across all samples before and after alignment.
3. **[Scores Comparison](scores_comparison.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
"""
        if raw_model.warp_type == 'linear':
            report_content += "4. **[Warp Parameter Recovery](chroma_warp_parameters.png)**: True vs. predicted shift and stretch parameters for each sample.\n"
            
        report_path = os.path.join(save_dir, 'report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"Diagnostics: Report written to: {report_path}")
            
    return metrics
