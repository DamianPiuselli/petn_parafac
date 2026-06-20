import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.common.utils import plot_scores_comparison


def extract_loadings(model):
    """
    Extracts the core loading matrices A, B, and C from the PyTorch model graph,
    detaches them from the device, and converts them to standard NumPy arrays.
    
    Args:
        model: Trained HPLC_PETN or GCMS_PETN model instance.
        
    Returns:
        dict: A dictionary containing:
            'A': scores (num_samples, num_components)
            'B': canonical chromatographic profiles (num_time, num_components)
            'C': spectral loadings (num_spec, num_components)
    """
    model.eval()
    with torch.no_grad():
        # Get compiled base model if using torch.compile
        raw_model = getattr(model, '_orig_mod', model)
        A = raw_model.A.detach().cpu().numpy()
        B = raw_model.B.detach().cpu().numpy()
        C = raw_model.C.detach().cpu().numpy()
    return {'A': A, 'B': B, 'C': C}

def extract_loadings_df(model, sample_names=None, time_points=None, spectral_channels=None):
    """
    Extracts the core loading matrices and formats them as standard pandas DataFrames
    for easy downstream chemometric analysis.
    
    Args:
        model: Trained HPLC_PETN or GCMS_PETN model instance.
        sample_names: Optional list of sample indices/labels (length I)
        time_points: Optional list/array of time points (length J)
        spectral_channels: Optional list/array of spectral channels (length K)
        
    Returns:
        dict: A dictionary of pandas DataFrames for 'A', 'B', and 'C'.
    """
    loadings = extract_loadings(model)
    A, B, C = loadings['A'], loadings['B'], loadings['C']
    num_comp = A.shape[1]
    comp_names = [f"Component_{r+1}" for r in range(num_comp)]
    
    if sample_names is None:
        sample_names = [f"Sample_{i+1}" for i in range(A.shape[0])]
    if time_points is None:
        time_points = np.arange(B.shape[0])
    if spectral_channels is None:
        spectral_channels = np.arange(C.shape[0])
        
    df_A = pd.DataFrame(A, index=sample_names, columns=comp_names)
    df_B = pd.DataFrame(B, index=time_points, columns=comp_names)
    df_C = pd.DataFrame(C, index=spectral_channels, columns=comp_names)
    
    return {'A': df_A, 'B': df_B, 'C': df_C}

def plot_alignment_verification(model, X_true, save_path=None):
    """
    Alignment Verifier: Isolates the highest intensity spectral channel,
    and plots the raw Total Ion Chromatograms (TICs) and channel profiles
    overlaid against the model's aligned predictions.
    
    Args:
        model: Trained HPLC_PETN or GCMS_PETN model instance.
        X_true: Observed/true data matrix of shape (I, J, K) - NumPy array or PyTorch Tensor.
        save_path: Optional path to save the generated image.
    """
    # 1. Convert X_true to NumPy array
    if isinstance(X_true, torch.Tensor):
        X_np = X_true.detach().cpu().numpy()
    else:
        X_np = np.array(X_true)
        
    I, J, K = X_np.shape
    
    # 2. Find the highest intensity spectral channel
    channel_intensities = np.sum(X_np, axis=(0, 1))
    k_max = np.argmax(channel_intensities)
    
    # 3. Generate un-derived predictions for direct physical overlay against raw observed data
    model.eval()
    with torch.no_grad():
        raw_model = getattr(model, '_orig_mod', model)
        A = raw_model.A
        C = raw_model.C
        _, B_warped_t, _ = raw_model._forward_raw_grid()
        
        if hasattr(raw_model, 'delta_B'):
            # GC-MS: incorporate shape residuals
            B_warped_t = B_warped_t + raw_model.delta_B
            
        Y_pred_tensor = torch.einsum('ir,ijr,kr->ijk', A, B_warped_t, C)
        
        if hasattr(raw_model, 'baseline_offset'):
            # HPLC: incorporate baseline offset and polynomial drift
            if hasattr(raw_model, 'sample_specific_baseline') and raw_model.sample_specific_baseline:
                t_grid = torch.linspace(0.0, 1.0, J, device=Y_pred_tensor.device).view(1, -1)
                poly = (raw_model.baseline_offset.unsqueeze(1) + 
                        raw_model.baseline_slope.unsqueeze(1) * t_grid + 
                        raw_model.baseline_quadratic.unsqueeze(1) * (t_grid ** 2))
                baseline = torch.einsum('ij,k->ijk', poly, raw_model.solvent_spectrum)
                Y_pred_tensor = Y_pred_tensor + baseline
            else:
                if raw_model.baseline_offset.ndim == 1:
                    Y_pred_tensor = Y_pred_tensor + raw_model.baseline_offset.view(1, 1, -1)
                else:
                    Y_pred_tensor = Y_pred_tensor + raw_model.baseline_offset.unsqueeze(1)
                
        Y_pred = Y_pred_tensor.detach().cpu().numpy()
            
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Highest Intensity Spectral Channel Overlay
    for i in range(I):
        label_raw = f"Sample {i+1} (Obs)" if i < 3 else ""
        label_pred = f"Sample {i+1} (Model)" if i < 3 else ""
        axes[0].plot(X_np[i, :, k_max], alpha=0.4, linewidth=1.5, label=label_raw)
        axes[0].plot(Y_pred[i, :, k_max], linestyle='--', alpha=0.8, linewidth=1.5, label=label_pred)
        
    axes[0].set_ylabel("Intensity", fontsize=12)
    axes[0].set_title(f"Alignment Verification at Peak Channel (m/z or Wavelength index: {k_max})", fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc="upper right", framealpha=0.9)
    
    # Plot 2: Total Ion Chromatograms (TIC) / Summed Wavelength Absorbance Overlay
    raw_tic = np.sum(X_np, axis=2)
    pred_tic = np.sum(Y_pred, axis=2)
    
    for i in range(I):
        axes[1].plot(raw_tic[i], alpha=0.4, linewidth=1.5)
        axes[1].plot(pred_tic[i], linestyle='--', alpha=0.8, linewidth=1.5)
        
    axes[1].set_xlabel("Time (Scan/Point Index)", fontsize=12)
    axes[1].set_ylabel("Total Summed Intensity", fontsize=12)
    axes[1].set_title("Total Ion Chromatogram (TIC) Alignment Verification", fontsize=14, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        dir_name = os.path.dirname(os.path.abspath(save_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostics: Alignment verification plot saved to: {save_path}")
    else:
        plt.show()
        
    plt.close()



