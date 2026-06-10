"""
Utility Functions.
Provides helpers for coordinate conversion, plotting EEM surfaces, and resolved loadings comparison.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_resolved_vs_true_profiles(true_B, true_C, pred_B, pred_C, ex_wavelens, em_wavelens, save_path=None):
    """
    Plots true vs. resolved excitation and emission loadings side by side.
    
    Args:
        true_B: shape (num_ex, num_components)
        true_C: shape (num_em, num_components)
        pred_B: shape (num_ex, num_components)
        pred_C: shape (num_em, num_components)
        ex_wavelens: array of excitation wavelengths
        em_wavelens: array of emission wavelengths
        save_path: path to save the generated image, or None to display it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    num_components = true_B.shape[1]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Plot Excitation Loadings
    for r in range(num_components):
        label_true = f'True Comp {r+1}'
        label_pred = f'Resolved Comp {r+1}'
        axes[0].plot(ex_wavelens, true_B[:, r], label=label_true, color=colors[r % len(colors)], linestyle='--', alpha=0.7)
        axes[0].plot(ex_wavelens, pred_B[:, r], label=label_pred, color=colors[r % len(colors)], linewidth=2)
        
    axes[0].set_title('Excitation Loadings (B)')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Normalized Intensity')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # 2. Plot Emission Loadings
    for r in range(num_components):
        label_true = f'True Comp {r+1}'
        label_pred = f'Resolved Comp {r+1}'
        axes[1].plot(em_wavelens, true_C[:, r], label=label_true, color=colors[r % len(colors)], linestyle='--', alpha=0.7)
        axes[1].plot(em_wavelens, pred_C[:, r], label=label_pred, color=colors[r % len(colors)], linewidth=2)
        
    axes[1].set_title('Emission Loadings (C)')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Normalized Intensity')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
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

def plot_ife_comparison(true_gamma, pred_gamma, ex_wavelens, em_wavelens, save_path=None):
    """
    Plots a 2-panel comparison of the true vs. learned IFE matrix (gamma).
    
    Args:
        true_gamma: 2D numpy array of shape (num_ex, num_em)
        pred_gamma: 2D numpy array of shape (num_ex, num_em)
        ex_wavelens: array of excitation wavelengths
        em_wavelens: array of emission wavelengths
        save_path: path to save the generated image, or None to display it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    EM, EX = np.meshgrid(em_wavelens, ex_wavelens)
    
    # 1. True Attenuation
    c1 = axes[0].contourf(EM, EX, true_gamma, levels=50, cmap='inferno', vmin=0.0, vmax=1.0)
    fig.colorbar(c1, ax=axes[0], label='Attenuation Factor (gamma)')
    axes[0].set_title('True IFE Attenuation Matrix')
    axes[0].set_xlabel('Emission Wavelength (nm)')
    axes[0].set_ylabel('Excitation Wavelength (nm)')
    
    # 2. Learned Attenuation
    c2 = axes[1].contourf(EM, EX, pred_gamma, levels=50, cmap='inferno', vmin=0.0, vmax=1.0)
    fig.colorbar(c2, ax=axes[1], label='Attenuation Factor (gamma)')
    axes[1].set_title('Learned IFE Attenuation Matrix')
    axes[1].set_xlabel('Emission Wavelength (nm)')
    axes[1].set_ylabel('Excitation Wavelength (nm)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"IFE comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


