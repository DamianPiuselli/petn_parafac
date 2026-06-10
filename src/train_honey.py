"""
Training and Validation of PINNParafac on the Copenhagen Honey EEM Dataset.
Loads HoneyEEM.mat, handles NaN masking, creates custom scattering masks,
trains the model, and validates by visualizing resolved profiles and PCA score clusters.
"""
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from src.model import PINNParafac
from src.loss import masked_mse_loss

def generate_honey_scattering_mask(ex_wavelens, em_wavelens):
    """
    Generates a 2D binary mask of shape (num_em, num_ex) to mask out
    1st and 2nd order Rayleigh scattering and water Raman scattering.
    """
    num_ex = len(ex_wavelens)
    num_em = len(em_wavelens)
    mask = np.ones((num_em, num_ex))
    
    # 1st-order Rayleigh: em = ex (width +/- 15 nm)
    # 2nd-order Rayleigh: em = 2*ex (width +/- 15 nm)
    # Solvent Raman: em_raman = ex / (1.0 - 3.4e-4 * ex) (width +/- 12 nm)
    for j in range(num_ex):
        ex = ex_wavelens[j]
        em_raman = ex / (1.0 - 3.4e-4 * ex)
        for k in range(num_em):
            em = em_wavelens[k]
            
            # Mask 1st-order Rayleigh
            if abs(em - ex) <= 15.0:
                mask[k, j] = 0.0
                
            # Mask 2nd-order Rayleigh
            if abs(em - 2 * ex) <= 15.0:
                mask[k, j] = 0.0
                
            # Mask Raman
            if abs(em - em_raman) <= 12.0:
                mask[k, j] = 0.0
                
    return mask

def train_honey_dataset(num_components=6, epochs=1500, lr=0.02, seed=42):
    """
    Loads HoneyEEM.mat, builds masks, trains the PINN model, and evaluates botanical separation.
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # 1. Load data
    mat_path = 'data/raw/honey/HoneyEEM.mat'
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Raw dataset not found at {mat_path}. Run src/download_honey.py first.")
        
    print(f"Loading raw honey dataset from {mat_path}...")
    from scipy.io import loadmat
    mat = loadmat(mat_path)
    dataset = mat['X'][0, 0]
    
    # Extract variables
    X_raw = dataset['data'] # (110, 741, 52)
    em_wavelens = dataset['axisscale'][1, 0].squeeze() # (741,)
    ex_wavelens = dataset['axisscale'][2, 0].squeeze() # (52,)

    # Downsample emission grid by 4 to speed up training without losing spectral shape information
    X_raw = X_raw[:, ::4, :]
    em_wavelens = em_wavelens[::4]
    
    # Normalize globally to [0.0, 1.0] for stable gradients and consistent initialization scaling
    global_max = np.nanmax(X_raw)
    X_raw = X_raw / global_max
    
    num_samples = X_raw.shape[0]
    num_em = len(em_wavelens)
    num_ex = len(ex_wavelens)
    
    print(f"Dataset downsampled & normalized: {num_samples} samples, {num_em} emission channels, {num_ex} excitation channels.")
    
    # Extract botanical classes
    class_ids = dataset['class'][0, 0].squeeze() # (110,)
    class_lookup_raw = dataset['classlookup'][0, 0]
    class_names = {}
    for row in class_lookup_raw:
        cid = int(row[0][0][0])
        cname = str(row[1][0])
        class_names[cid] = cname
        
    print("Class mapping:")
    for cid, name in sorted(class_names.items()):
        count = np.sum(class_ids == cid)
        print(f"  ID {cid}: {name} ({count} samples)")
        
    # 2. Build masks
    # Create sample-specific NaN mask
    nan_mask = ~np.isnan(X_raw)
    print(f"Total NaNs in raw data: {np.isnan(X_raw).sum()} ({np.isnan(X_raw).mean()*100:.2f}% of pixels)")
    
    # Replace NaNs with 0.0 in data
    X = np.nan_to_num(X_raw, nan=0.0)
    
    # Create 2D instrument scattering mask
    scatter_mask_2d = generate_honey_scattering_mask(ex_wavelens, em_wavelens) # (num_em, num_ex)
    scatter_mask_3d = scatter_mask_2d[np.newaxis, :, :].repeat(num_samples, axis=0) # (num_samples, num_em, num_ex)
    
    # Combine masks
    combined_mask = scatter_mask_3d * nan_mask
    print(f"Masked pixels (scatter + NaNs): {np.sum(combined_mask == 0)} / {combined_mask.size} ({np.mean(combined_mask == 0)*100:.2f}%)")
    
    # Flatten grid to coordinate triplets matching the shape (num_samples, num_em, num_ex)
    sample_grid, em_grid, ex_grid = np.meshgrid(
        np.arange(num_samples),
        np.arange(num_em),
        np.arange(num_ex),
        indexing='ij'
    )
    
    # Tensors moved to device
    sample_indices = torch.tensor(sample_grid.reshape(-1), dtype=torch.long, device=device)
    em_indices = torch.tensor(em_grid.reshape(-1), dtype=torch.long, device=device)
    ex_indices = torch.tensor(ex_grid.reshape(-1), dtype=torch.long, device=device)
    intensities = torch.tensor(X.reshape(-1), dtype=torch.float32, device=device)
    mask_values = torch.tensor(combined_mask.reshape(-1), dtype=torch.float32, device=device)
    
    # 3. Instantiate model
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # No baseline solvent absorbance (water blank corrected or zero-buffer reference)
    ex_bg = torch.zeros(num_ex)
    em_bg = torch.zeros(num_em)
    
    print(f"Building PINN model for Honey (num_components = {num_components})...")
    model = PINNParafac(
        num_samples=num_samples,
        num_ex=num_ex,
        num_em=num_em,
        ex_wavelens=ex_wavelens,
        em_wavelens=em_wavelens,
        ex_bg=ex_bg,
        em_bg=em_bg,
        num_components=num_components
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    print(f"Training jointly for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # PINNParafac forward signature: forward(sample_idx, ex_idx, em_idx)
        predictions = model(sample_indices, ex_indices, em_indices)
        loss = masked_mse_loss(predictions, intensities, mask_values)
        
        loss.backward()
        optimizer.step()
        model.project_constraints()
        
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{epochs} - Loss: {loss.item():.6f}")
            
    # 5. Extract resolved parameters
    model.eval()
    with torch.no_grad():
        pred_A = model.sample_embeddings.weight.cpu().numpy()
        pred_B = model.ex_embeddings.weight.cpu().numpy()
        pred_C = model.em_embeddings.weight.cpu().numpy()
        pred_E, pred_M = model.get_learned_absorptivities()
        
    # Sort resolved components consistently by peak emission wavelength
    peak_em_indices = np.argmax(pred_C, axis=0)
    sort_idx = np.argsort(em_wavelens[peak_em_indices])
    
    pred_A = pred_A[:, sort_idx]
    pred_B = pred_B[:, sort_idx]
    pred_C = pred_C[:, sort_idx]
    pred_E = pred_E[:, sort_idx]
    
    # Normalize loading profiles to peak at 1.0
    for r in range(num_components):
        max_b = np.max(pred_B[:, r])
        max_c = np.max(pred_C[:, r])
        pred_B[:, r] /= (max_b if max_b > 1e-8 else 1.0)
        pred_C[:, r] /= (max_c if max_c > 1e-8 else 1.0)
        
    # 6. Evaluation: Class separability analysis
    # PCA on sample scores A using SVD from scratch
    # We will do PCA on both raw and row-normalized scores
    
    # 1. Raw scores PCA & Classification
    A_centered = pred_A - np.mean(pred_A, axis=0)
    U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
    scores_pca = U[:, :2] * S[:2]
    
    correct_raw = 0
    for i in range(num_samples):
        dists = np.sum((pred_A - pred_A[i])**2, axis=1)
        dists[i] = np.inf
        nearest_idx = np.argmin(dists)
        if class_ids[nearest_idx] == class_ids[i]:
            correct_raw += 1
    classification_acc = correct_raw / num_samples
    print(f"Resolved sample scores 1-NN Leave-One-Out Classification Accuracy (Raw): {classification_acc * 100:.2f}%")
    
    # 2. Row-normalized scores PCA & Classification
    # To avoid division by zero, we add a tiny epsilon to sum
    pred_A_norm = pred_A / (np.sum(pred_A, axis=1, keepdims=True) + 1e-12)
    A_norm_centered = pred_A_norm - np.mean(pred_A_norm, axis=0)
    U_n, S_n, Vt_n = np.linalg.svd(A_norm_centered, full_matrices=False)
    scores_norm_pca = U_n[:, :2] * S_n[:2]
    
    correct_norm = 0
    for i in range(num_samples):
        dists = np.sum((pred_A_norm - pred_A_norm[i])**2, axis=1)
        dists[i] = np.inf
        nearest_idx = np.argmin(dists)
        if class_ids[nearest_idx] == class_ids[i]:
            correct_norm += 1
    classification_acc_norm = correct_norm / num_samples
    print(f"Resolved sample scores 1-NN Leave-One-Out Classification Accuracy (Row-Normalized): {classification_acc_norm * 100:.2f}%")
    
    # 7. Plotting and Visualizations
    os.makedirs('notebooks', exist_ok=True)
    
    # Plot 1: Resolved Loadings and Molar Absorptivities
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Excitation Loadings
    for r in range(num_components):
        axes[0].plot(ex_wavelens, pred_B[:, r], label=f'Comp {r+1}', linewidth=2.5)
    axes[0].set_title('Resolved Excitation Loadings')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Normalized Intensity')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(ncol=2)
    
    # Emission Loadings
    for r in range(num_components):
        axes[1].plot(em_wavelens, pred_C[:, r], label=f'Comp {r+1}', linewidth=2.5)
    axes[1].set_title('Resolved Emission Loadings')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Normalized Intensity')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(ncol=2)
    
    # Molar Absorptivities
    for r in range(num_components):
        axes[2].plot(ex_wavelens, pred_E[:, r], label=f'Comp {r+1}', linewidth=2.5)
    axes[2].set_title('Resolved Molar Absorptivities (E)')
    axes[2].set_xlabel('Wavelength (nm)')
    axes[2].set_ylabel('Absorptivity Coefficient')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(ncol=2)
    
    plt.tight_layout()
    plots_path = 'notebooks/honey_resolved_profiles.png'
    plt.savefig(plots_path, dpi=200)
    plt.close()
    print(f"Saved resolved profiles to {plots_path}")
    
    # Plot 2: PCA Score Clusters (Row-Normalized)
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    unique_cids = np.unique(class_ids)
    for cid in unique_cids:
        indices = (class_ids == cid)
        cname = class_names[cid]
        plt.scatter(
            scores_norm_pca[indices, 0], 
            scores_norm_pca[indices, 1], 
            label=cname,
            c=colors[(cid-1) % len(colors)],
            edgecolors='k',
            alpha=0.8,
            s=80
        )
        
    plt.title(f'PCA of Row-Normalized Resolved Honey Scores (1-NN Accuracy: {classification_acc_norm*100:.1f}%)', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Botanical Origin', fontsize=10)
    
    pca_path = 'notebooks/honey_pca_separation.png'
    plt.tight_layout()
    plt.savefig(pca_path, dpi=200)
    plt.close()
    print(f"Saved PCA separation plot to {pca_path}")

if __name__ == '__main__':
    train_honey_dataset()
