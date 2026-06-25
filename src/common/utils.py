import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_resolved_vs_true_profiles(true_B, true_C, pred_B, pred_C, ex_wavelens, em_wavelens, component_names=None, save_path=None):
    """
    Plots true vs. resolved excitation (B) and emission (C) loadings side by side, 
    separated by component in a multi-row grid.
    
    Args:
        true_B: shape (num_ex, num_components)
        true_C: shape (num_em, num_components)
        pred_B: shape (num_ex, num_components)
        pred_C: shape (num_em, num_components)
        ex_wavelens: array of excitation wavelengths
        em_wavelens: array of emission wavelengths
        component_names: optional list of component labels (length num_components)
        save_path: path to save the generated image, or None to display it.
    """
    num_components = pred_B.shape[1]
    
    fig, axes = plt.subplots(num_components, 2, figsize=(14, 3.2 * num_components))
    if num_components == 1:
        axes = np.expand_dims(axes, axis=0)
        
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for r in range(num_components):
        color = colors[r % len(colors)]
        comp_name = component_names[r] if component_names is not None else f'Component {r+1}'
        
        # 1. Excitation Loading (Left column)
        ax_b = axes[r, 0]
        if true_B is not None:
            ax_b.plot(ex_wavelens, true_B[:, r], label='True', color='gray', linestyle='--', alpha=0.7)
        ax_b.plot(ex_wavelens, pred_B[:, r], label='Resolved', color=color, linewidth=2)
        
        ax_b.set_title(f'{comp_name} - Excitation Loading (B)')
        ax_b.set_xlabel('Wavelength (nm)')
        ax_b.set_ylabel('Normalized Intensity')
        ax_b.grid(True, linestyle=':', alpha=0.6)
        ax_b.legend()
        
        # 2. Emission Loading (Right column)
        ax_c = axes[r, 1]
        if true_C is not None:
            ax_c.plot(em_wavelens, true_C[:, r], label='True', color='gray', linestyle='--', alpha=0.7)
        ax_c.plot(em_wavelens, pred_C[:, r], label='Resolved', color=color, linewidth=2)
        
        ax_c.set_title(f'{comp_name} - Emission Loading (C)')
        ax_c.set_xlabel('Wavelength (nm)')
        ax_c.set_ylabel('Normalized Intensity')
        ax_c.grid(True, linestyle=':', alpha=0.6)
        ax_c.legend()
        
    plt.tight_layout()
    if save_path:
        dir_name = os.path.dirname(os.path.abspath(save_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_eem_heatmaps(X_clean, X_corrupted, mask, X_reconstructed, ex_wavelens, em_wavelens, save_path=None):
    """
    Plots a 4-panel comparison of EEM heatmaps:
    1. Clean EEM
    2. Corrupted EEM (with scattering lines)
    3. Binary Mask W
    4. Reconstructed EEM from the model
    
    Args:
        X_clean: 2D numpy array of shape (num_ex, num_em)
        X_corrupted: 2D numpy array of shape (num_ex, num_em)
        mask: 2D numpy array of shape (num_ex, num_em)
        X_reconstructed: 2D numpy array of shape (num_ex, num_em)
        ex_wavelens: array of excitation wavelengths
        em_wavelens: array of emission wavelengths
        save_path: path to save the generated image, or None to display it.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Grid for contour plotting
    EM, EX = np.meshgrid(em_wavelens, ex_wavelens)
    
    # 1. Clean EEM
    c1 = axes[0, 0].contourf(EM, EX, X_clean, levels=50, cmap='viridis')
    fig.colorbar(c1, ax=axes[0, 0], label='Intensity')
    axes[0, 0].set_title('True Clean EEM')
    axes[0, 0].set_xlabel('Emission Wavelength (nm)')
    axes[0, 0].set_ylabel('Excitation Wavelength (nm)')
    
    # 2. Corrupted EEM
    max_clean = np.max(X_clean)
    c2 = axes[0, 1].contourf(EM, EX, X_corrupted, levels=50, cmap='viridis', vmax=max_clean * 1.5)
    fig.colorbar(c2, ax=axes[0, 1], label='Intensity (Saturated)')
    axes[0, 1].set_title('Corrupted EEM (with Scattering)')
    axes[0, 1].set_xlabel('Emission Wavelength (nm)')
    axes[0, 1].set_ylabel('Excitation Wavelength (nm)')
    
    # 3. Binary Mask W
    c3 = axes[1, 0].contourf(EM, EX, mask, levels=2, cmap='gray')
    fig.colorbar(c3, ax=axes[1, 0], label='Mask Value (0=Ignore, 1=Train)')
    axes[1, 0].set_title('Binary Mask W')
    axes[1, 0].set_xlabel('Emission Wavelength (nm)')
    axes[1, 0].set_ylabel('Excitation Wavelength (nm)')
    
    # 4. Reconstructed EEM
    c4 = axes[1, 1].contourf(EM, EX, X_reconstructed, levels=50, cmap='viridis', vmax=max_clean)
    fig.colorbar(c4, ax=axes[1, 1], label='Intensity')
    axes[1, 1].set_title('PINN Reconstructed EEM (Scattering Interpolated)')
    axes[1, 1].set_xlabel('Emission Wavelength (nm)')
    axes[1, 1].set_ylabel('Excitation Wavelength (nm)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"EEM heatmap plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_resolved_absorptivities(true_E, true_M, pred_E, pred_M, ex_wavelens, em_wavelens, save_path=None):
    """
    Plots a comparison of true vs. resolved excitation and emission molar absorptivities.
    
    Args:
        true_E: shape (num_ex, num_components)
        true_M: shape (num_em, num_components)
        pred_E: shape (num_ex, num_components)
        pred_M: shape (num_em, num_components)
        ex_wavelens: array of excitation wavelengths
        em_wavelens: array of emission wavelengths
        save_path: path to save the generated image, or None to display it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    num_components = true_E.shape[1]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Excitation Absorptivities
    for r in range(num_components):
        label_true = f'True Abs Comp {r+1}'
        label_pred = f'Resolved Abs Comp {r+1}'
        axes[0].plot(ex_wavelens, true_E[:, r], label=label_true, color=colors[r % len(colors)], linestyle='--', alpha=0.7)
        axes[0].plot(ex_wavelens, pred_E[:, r], label=label_pred, color=colors[r % len(colors)], linewidth=2)
        
    axes[0].set_title('Excitation Molar Absorptivities (E)')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Absorptivity')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # 2. Emission Absorptivities
    for r in range(num_components):
        label_true = f'True Abs Comp {r+1}'
        label_pred = f'Resolved Abs Comp {r+1}'
        axes[1].plot(em_wavelens, true_M[:, r], label=label_true, color=colors[r % len(colors)], linestyle='--', alpha=0.7)
        axes[1].plot(em_wavelens, pred_M[:, r], label=label_pred, color=colors[r % len(colors)], linewidth=2)
        
    axes[1].set_title('Emission Molar Absorptivities (M)')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Absorptivity')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Molar absorptivities comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_chroma_resolved_vs_true_profiles(true_B, true_C, pred_B, pred_C, time_grid, spec_grid, component_names=None, plot_type='ms', save_path=None):
    """
    Plots chromatography profiles (B) on the left and spectral profiles (C) on the right.
    Both B and C are plotted separately for each component.
    
    Args:
        true_B: true chromatography profiles
        true_C: true spectral profiles
        pred_B: resolved chromatography profiles
        pred_C: resolved spectral profiles
        time_grid: array of time grid points
        spec_grid: array of spectral channel points (m/z or wavelength)
        component_names: list of component label strings
        plot_type: 'ms' (mass spectrometry, vertical spikes) or 'dad' (diode array detector / UV-Vis, continuous curves)
        save_path: path to save the generated figure
    """
    num_components = pred_B.shape[1]
    
    fig, axes = plt.subplots(num_components, 2, figsize=(14, 3.2 * num_components))
    if num_components == 1:
        axes = np.expand_dims(axes, axis=0)
        
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for r in range(num_components):
        color = colors[r % len(colors)]
        comp_name = component_names[r] if component_names is not None else f'Component {r+1}'
        
        # 1. Chromatography profile (Left column)
        ax_b = axes[r, 0]
        if true_B is not None:
            ax_b.plot(time_grid, true_B[:, r], label='True', color='gray', linestyle='--', alpha=0.7)
        ax_b.plot(time_grid, pred_B[:, r], label='Resolved', color=color, linewidth=2)
        
        ax_b.set_title(f'{comp_name} - Chromatography Profile (B)')
        ax_b.set_xlabel('Time')
        ax_b.set_ylabel('Normalized Intensity')
        ax_b.grid(True, linestyle=':', alpha=0.6)
        ax_b.legend()
        
        # 2. Spectral profile (Right column)
        ax_c = axes[r, 1]
        
        if plot_type == 'ms':
            # Plot predicted spectra as discrete spikes (vlines)
            ax_c.vlines(spec_grid, 0.0, pred_C[:, r], colors=color, linewidth=1.5, label='Resolved')
            
            # Plot true spectra (literature) if available as reference thin lines
            if true_C is not None:
                ax_c.vlines(spec_grid, 0.0, true_C[:, r], colors='gray', alpha=0.5, linewidth=1.0, label='Literature' if 'applewine' in str(save_path) else 'True')
                
            ax_c.set_title(f'{comp_name} - Mass Spectrum (C)')
            ax_c.set_xlabel('m/z')
            ax_c.set_ylabel('Relative Intensity')
            ax_c.set_ylim(0.0, 1.1)
        else:
            # Plot predicted spectra as continuous curves
            if true_C is not None:
                ax_c.plot(spec_grid, true_C[:, r], label='True', color='gray', linestyle='--', alpha=0.7)
            ax_c.plot(spec_grid, pred_C[:, r], label='Resolved', color=color, linewidth=2)
            
            ax_c.set_title(f'{comp_name} - Absorbance Spectrum (C)')
            ax_c.set_xlabel('Wavelength (nm)')
            ax_c.set_ylabel('Normalized Absorbance')
            
        ax_c.grid(True, linestyle=':', alpha=0.6)
        ax_c.legend()
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Resolved profiles comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_chroma_alignment_comparison(time_grid, X_unaligned, X_aligned, save_path=None):
    """
    Plots unaligned vs. aligned chromatograms across all samples.
    We sum along the spectral dimension to obtain the Total Ion Chromatogram (TIC) equivalent.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    num_samples = X_unaligned.shape[0]
    
    # Sum over spectra to get TIC
    tic_unaligned = np.sum(X_unaligned, axis=2)
    tic_aligned = np.sum(X_aligned, axis=2)
    
    # 1. Unaligned
    for i in range(num_samples):
        axes[0].plot(time_grid, tic_unaligned[i], alpha=0.7)
    axes[0].set_title('Un-aligned Chromatograms (Observed)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Total Intensity (TIC)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. Aligned
    for i in range(num_samples):
        axes[1].plot(time_grid, tic_aligned[i], alpha=0.7)
    axes[1].set_title('Chroma-PETN Aligned Chromatograms')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Total Intensity (TIC)')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Alignment comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_chroma_warp_parameters(true_shifts, true_stretches, pred_shifts, pred_stretches, save_path=None):
    """
    Plots true vs predicted shift and stretch factors across all samples.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    num_samples = len(true_shifts)
    sample_indices = np.arange(num_samples)
    
    # 1. Shifts
    axes[0].scatter(sample_indices, true_shifts, label='True Shift (Mean-Centered)', color='#d62728', marker='o', s=100)
    axes[0].scatter(sample_indices, pred_shifts, label='Recovered Shift (Mean-Centered)', color='#1f77b4', marker='x', s=100, linewidths=2)
    axes[0].set_title('Shift Parameter Recovery (beta - Mean-Centered)')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Mean-Centered Shift (Time units)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # 2. Stretches
    axes[1].scatter(sample_indices, true_stretches, label='True Stretch (Mean-Centered)', color='#d62728', marker='o', s=100)
    axes[1].scatter(sample_indices, pred_stretches, label='Recovered Stretch (Mean-Centered)', color='#1f77b4', marker='x', s=100, linewidths=2)
    axes[1].set_title('Stretch Parameter Recovery (alpha - Mean-Centered)')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Mean-Centered Stretch Factor')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Warp parameters plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_scores_parity(true_A, pred_A, num_calib=5, components_to_plot=None, component_names=None, save_path=None):
    """
    Calibrates the predicted scores using the first `num_calib` samples as standards,
    and plots true vs. calibrated concentrations as a parity scatter plot with a 1:1 line.
    
    Args:
        true_A: true score matrix of shape (I, R)
        pred_A: predicted score matrix of shape (I, R)
        num_calib: number of standard samples to fit calibration slope
        components_to_plot: list of component indices to plot (0-indexed). 
                             If None, plots all components.
        component_names: list of string names for the components.
    """
    total_components = true_A.shape[1]
    if components_to_plot is None:
        components_to_plot = list(range(total_components))
        
    num_to_plot = len(components_to_plot)
    fig, axes = plt.subplots(1, num_to_plot, figsize=(5 * num_to_plot, 4.5))
    if num_to_plot == 1:
        axes = [axes]
        
    for idx, r in enumerate(components_to_plot):
        ax = axes[idx]
        t_val = true_A[:, r]
        p_val = pred_A[:, r]
        
        # 1. Fit calibration slope using the first `num_calib` samples as standards
        # We solve: true = slope * pred (no intercept, assuming zero blank)
        t_cal = t_val[:num_calib]
        p_cal = p_val[:num_calib]
        
        slope = np.sum(t_cal * p_cal) / (np.sum(p_cal ** 2) + 1e-12)
        
        # Apply calibration to all samples
        p_val_calibrated = slope * p_val
        
        # 2. Calculate R2 correlation for validation samples (indices >= num_calib)
        t_val_val = t_val[num_calib:]
        p_val_val = p_val_calibrated[num_calib:]
        
        corr_val = np.corrcoef(t_val_val, p_val_val)[0, 1]
        r2_val = corr_val ** 2 if not np.isnan(corr_val) else 0.0
        
        # 3. Plot calibration standards
        ax.scatter(t_val[:num_calib], p_val_calibrated[:num_calib], 
                   color='#d62728', marker='o', edgecolors='k', s=80, label='Cal. Standards')
        
        # 4. Plot validation samples
        ax.scatter(t_val[num_calib:], p_val_calibrated[num_calib:], 
                   color='#1f77b4', marker='.', alpha=0.7, s=50, label=f'Validation (R² = {r2_val:.4f})')
        
        # 1:1 line range
        min_val = min(np.min(t_val), np.min(p_val_calibrated))
        max_val = max(np.max(t_val), np.max(p_val_calibrated))
        margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
        line_range = np.linspace(min_val - margin, max_val + margin, 100)
        
        ax.plot(line_range, line_range, color='gray', linestyle='--', alpha=0.7, label='1:1 Line')
        
        comp_name = component_names[r] if component_names is not None else f'Component {r+1}'
        ax.set_title(f'{comp_name} Calibration')
        ax.set_xlabel('True Concentration')
        ax.set_ylabel('Calibrated Concentration')
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Scores parity plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_scores_comparison(A_true, A_pred, component_names=None, save_path=None):
    """
    Plots true vs. predicted scores (concentrations) for each component.
    Plots a 1:1 diagonal line and calculates correlation R2 and similarity scores.
    
    Args:
        A_true: shape (num_samples, num_components)
        A_pred: shape (num_samples, num_components)
        component_names: list of component names (length num_components)
        save_path: path to save the generated image, or None to display it.
    """
    import os
    num_components = A_true.shape[1]
    fig, axes = plt.subplots(1, num_components, figsize=(5 * num_components, 4.5))
    if num_components == 1:
        axes = [axes]
        
    for r in range(num_components):
        ax = axes[r]
        y_true = A_true[:, r]
        y_pred = A_pred[:, r]
        
        # Calculate R^2 and correlation
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        r2 = corr ** 2 if not np.isnan(corr) else 0.0
        
        # Calculate cosine similarity
        v1_norm = y_true / (np.linalg.norm(y_true) + 1e-10)
        v2_norm = y_pred / (np.linalg.norm(y_pred) + 1e-10)
        cos_sim = np.dot(v1_norm, v2_norm)
        
        # Scatter plot
        ax.scatter(y_true, y_pred, color='#1f77b4', edgecolors='k', s=80, alpha=0.8, zorder=3)
        
        # 1:1 diagonal line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
        line_range = np.linspace(min_val - margin, max_val + margin, 100)
        ax.plot(line_range, line_range, color='gray', linestyle='--', alpha=0.7, label='1:1 Line', zorder=2)
        
        comp_name = component_names[r] if component_names is not None else f'Component {r+1}'
        ax.set_title(f'{comp_name}\nSimilarity: {cos_sim:.4f} | R² = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('True Score (Concentration)', fontsize=10)
        ax.set_ylabel('Predicted Score (Concentration)', fontsize=10)
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.grid(True, linestyle=':', alpha=0.6, zorder=1)
        ax.legend(loc='upper left')
        
    plt.tight_layout()
    if save_path:
        dir_name = os.path.dirname(os.path.abspath(save_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostics: Scores comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

class EarlyStopping:
    """
    Scale-independent Early Stopping monitor.
    Uses Fraction of Variance Unexplained (FVU = MSE / Var(y)) to normalize loss across scales,
    and monitors relative improvement to prevent premature stopping on slow learning phases.
    """
    def __init__(self, patience=50, tol=1e-5, min_epochs=None):
        self.patience = patience
        self.tol = tol
        if min_epochs is None:
            # Allow min_epochs to adapt if patience is small (for testing)
            self.min_epochs = max(0, patience - 1)
        else:
            self.min_epochs = max(0, min_epochs)
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.var_y = None
        
    def __call__(self, epoch, loss_val, y_target):
        # 1. Initialize variance of target to normalize loss (scale independence)
        if self.var_y is None:
            if isinstance(y_target, torch.Tensor):
                self.var_y = torch.var(y_target).item()
            else:
                self.var_y = np.var(y_target)
                
            if self.var_y < 1e-10:
                self.var_y = 1.0  # Avoid division by zero for constant target
                
        # 2. Compute Fraction of Variance Unexplained (FVU)
        fvu = loss_val / self.var_y
        
        # 3. Check for hard scale-independent convergence (R² > 99.9999%)
        if fvu < 1e-6:
            print(f"Convergence reached at epoch {epoch:4d} (Loss < target threshold). Final loss: {loss_val:.3e}")
            return True
            
        if epoch < self.min_epochs:
            # Update best loss during warmup period without checking patience
            if loss_val < self.best_loss:
                self.best_loss = loss_val
            return False
            
        # 4. Check relative improvement
        # We require a relative decrease of at least `tol` (e.g. 1e-5)
        improvement = (self.best_loss - loss_val) / (self.best_loss + 1e-10)
        
        if improvement > self.tol:
            self.best_loss = loss_val
            self.patience_counter = 0  # Significant improvement, reset patience
        else:
            if loss_val < self.best_loss:
                self.best_loss = loss_val  # Update baseline, but don't reset patience
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            print(f"Early stopping at epoch {epoch:4d} (Loss did not decrease significantly for {self.patience} epochs). Final loss: {loss_val:.3e}")
            return True
            
        return False





