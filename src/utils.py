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
