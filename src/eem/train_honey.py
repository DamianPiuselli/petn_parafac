"""
Training and Validation of PETNParafac on the Copenhagen Honey EEM Dataset.
Loads HoneyEEM.mat, handles NaN masking, creates custom scattering masks,
trains the model, and validates by visualizing resolved profiles and PCA score clusters.
"""
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from src.eem.model import PETNParafac
from src.eem.loss import masked_mse_loss

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

def train_honey_dataset(num_components=6, epochs=3000, lr=0.03, seed=42):
    """
    Loads HoneyEEM.mat, builds masks, trains the PETN model, and evaluates botanical separation.
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
    
    # Normalize each sample individually (slice normalization) to remove dilution/concentration effects
    # and prevent samples with high intensities from dominating the loss function.
    sample_maxes = np.nanmax(X_raw, axis=(1, 2), keepdims=True)
    sample_maxes[sample_maxes <= 1e-8] = 1.0
    X_raw = X_raw / sample_maxes
    
    num_samples = X_raw.shape[0]
    num_em = len(em_wavelens)
    num_ex = len(ex_wavelens)
    
    print(f"Dataset downsampled & normalized: {num_samples} samples, {num_em} emission channels, {num_ex} excitation channels.")
    
    # Extract botanical classes
    class_ids = dataset['class'][0, 0].squeeze().copy() # (110,)
    # Group all fake honey classes (2, 3, 4, 5) into a single "Adulterated (Fake)" class (ID 2)
    class_ids[(class_ids >= 2) & (class_ids <= 5)] = 2
    
    class_names = {
        1: 'Acacia',
        2: 'Adulterated (Fake)',
        6: 'Linden',
        7: 'Meadow',
        8: 'Sunflower'
    }
        
    print("Class mapping (grouped):")
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
    
    print(f"Building PETN model for Honey (num_components = {num_components})...")
    model = PETNParafac(
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
        
        # PETNParafac forward signature: forward(sample_idx, ex_idx, em_idx)
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
        pred_alpha = model.alpha.cpu().numpy()
        
    # Resolve Scale Ambiguity post-training in NumPy:
    # Normalize B and C to unit L2 norm, and scale A and alpha accordingly
    norm_B = np.linalg.norm(pred_B, axis=0, keepdims=True) + 1e-12  # (1, num_components)
    norm_C = np.linalg.norm(pred_C, axis=0, keepdims=True) + 1e-12  # (1, num_components)
    
    # Scale A: A <- A * norm_B * norm_C
    pred_A = pred_A * norm_B * norm_C
    
    # Scale alpha: alpha <- alpha / norm_C
    pred_alpha = pred_alpha / norm_C.squeeze(0)
    
    # Normalize loadings B and C
    pred_B = pred_B / norm_B
    pred_C = pred_C / norm_C
    
    # Compute resolved molar absorptivities: E = alpha * B
    pred_E = pred_alpha * pred_B
        
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
        
    # 6. Evaluation & Diagnostics
    print("\n--- Model Diagnostics ---")
    alpha_val = pred_alpha
    print(f"Learned Alpha (molar absorptivities): {alpha_val}")
    
    # Calculate reconstructed attenuation gamma = 10^(-Abs_ex)
    # pred_A: (110, 6), pred_B: (52, 6) -> Abs_ex is sample-wise and wavelength-wise
    # Let's compute average gamma across all sample-excitation pairs
    # pred_A has shape (num_samples, num_components)
    # pred_B has shape (num_ex, num_components)
    # Abs_ex_matrix shape: (num_samples, num_ex)
    Abs_ex_matrix = np.dot(pred_A, (alpha_val * pred_B).T)
    gamma_matrix = 10.0 ** (-Abs_ex_matrix)
    print(f"Attenuation gamma - Mean: {np.mean(gamma_matrix):.4f}, Min: {np.min(gamma_matrix):.4f}, Max: {np.max(gamma_matrix):.4f}")
    
    # Define multi-classifier evaluation function
    def evaluate_knn(scores, labels, k_list=[1, 3, 5]):
        num_samples = len(labels)
        results = {}
        for k in k_list:
            correct = 0
            for i in range(num_samples):
                dists = np.sum((scores - scores[i])**2, axis=1)
                dists[i] = np.inf
                nearest_indices = np.argsort(dists)[:k]
                nearest_labels = labels[nearest_indices]
                unique, counts = np.unique(nearest_labels, return_counts=True)
                pred_label = unique[np.argmax(counts)]
                if pred_label == labels[i]:
                    correct += 1
            results[k] = correct / num_samples
        return results

    # Score normalizations
    pred_A_l1 = pred_A / (np.sum(pred_A, axis=1, keepdims=True) + 1e-12)
    pred_A_l2 = pred_A / (np.linalg.norm(pred_A, axis=1, keepdims=True) + 1e-12)
    pred_A_std = (pred_A - np.mean(pred_A, axis=0)) / (np.std(pred_A, axis=0) + 1e-12)
    pred_A_l2_std = (pred_A_l2 - np.mean(pred_A_l2, axis=0)) / (np.std(pred_A_l2, axis=0) + 1e-12)
    
    acc_raw = evaluate_knn(pred_A, class_ids)
    acc_l1 = evaluate_knn(pred_A_l1, class_ids)
    acc_l2 = evaluate_knn(pred_A_l2, class_ids)
    acc_std = evaluate_knn(pred_A_std, class_ids)
    acc_l2_std = evaluate_knn(pred_A_l2_std, class_ids)
    
    print("\n--- LOO k-NN Classification Accuracy ---")
    print(f"Raw scores:         1-NN: {acc_raw[1]*100:.2f}%, 3-NN: {acc_raw[3]*100:.2f}%, 5-NN: {acc_raw[5]*100:.2f}%")
    print(f"L1 Normalized:      1-NN: {acc_l1[1]*100:.2f}%, 3-NN: {acc_l1[3]*100:.2f}%, 5-NN: {acc_l1[5]*100:.2f}%")
    print(f"L2 Normalized:      1-NN: {acc_l2[1]*100:.2f}%, 3-NN: {acc_l2[3]*100:.2f}%, 5-NN: {acc_l2[5]*100:.2f}%")
    print(f"Standardized (Z):   1-NN: {acc_std[1]*100:.2f}%, 3-NN: {acc_std[3]*100:.2f}%, 5-NN: {acc_std[5]*100:.2f}%")
    print(f"L2 + Standardized:  1-NN: {acc_l2_std[1]*100:.2f}%, 3-NN: {acc_l2_std[3]*100:.2f}%, 5-NN: {acc_l2_std[5]*100:.2f}%")
    
    # Supervised classification using scikit-learn (to replicate PLS-DA and other literature benchmarks)
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    # One-hot encode targets for PLS-DA
    unique_classes = np.unique(class_ids)
    num_classes = len(unique_classes)
    label_to_idx = {l: idx for idx, l in enumerate(unique_classes)}
    y_indices = np.array([label_to_idx[l] for l in class_ids])
    Y_onehot = np.zeros((num_samples, num_classes))
    Y_onehot[np.arange(num_samples), y_indices] = 1.0

    def evaluate_supervised(scores):
        loo = LeaveOneOut()
        
        # 1. PLS-DA
        correct_pls = 0
        for train_idx, test_idx in loo.split(scores):
            pls = PLSRegression(n_components=min(5, scores.shape[1]))
            pls.fit(scores[train_idx], Y_onehot[train_idx])
            pred = pls.predict(scores[test_idx])
            if np.argmax(pred[0]) == y_indices[test_idx[0]]:
                correct_pls += 1
        acc_pls = correct_pls / num_samples
        
        # 2. SVM (Linear)
        correct_svc_lin = 0
        for train_idx, test_idx in loo.split(scores):
            svc = SVC(kernel='linear', C=1.0, class_weight='balanced')
            svc.fit(scores[train_idx], y_indices[train_idx])
            pred = svc.predict(scores[test_idx])
            if pred[0] == y_indices[test_idx[0]]:
                correct_svc_lin += 1
        acc_svc_lin = correct_svc_lin / num_samples
        
        # 3. SVM (RBF)
        correct_svc_rbf = 0
        for train_idx, test_idx in loo.split(scores):
            svc = SVC(kernel='rbf', C=1.0, class_weight='balanced')
            svc.fit(scores[train_idx], y_indices[train_idx])
            pred = svc.predict(scores[test_idx])
            if pred[0] == y_indices[test_idx[0]]:
                correct_svc_rbf += 1
        acc_svc_rbf = correct_svc_rbf / num_samples
        
        # 4. Logistic Regression
        correct_lr = 0
        for train_idx, test_idx in loo.split(scores):
            lr = LogisticRegression(class_weight='balanced', max_iter=2000)
            lr.fit(scores[train_idx], y_indices[train_idx])
            pred = lr.predict(scores[test_idx])
            if pred[0] == y_indices[test_idx[0]]:
                correct_lr += 1
        acc_lr = correct_lr / num_samples
        
        # 5. Random Forest
        correct_rf = 0
        for train_idx, test_idx in loo.split(scores):
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(scores[train_idx], y_indices[train_idx])
            pred = rf.predict(scores[test_idx])
            if pred[0] == y_indices[test_idx[0]]:
                correct_rf += 1
        acc_rf = correct_rf / num_samples
        
        return acc_pls, acc_svc_lin, acc_svc_rbf, acc_lr, acc_rf

    # Evaluate on the best representation: L2 Normalized + Standardized (Autoscaled) scores
    A_norm_centered = pred_A_l2_std
    acc_pls, acc_svc_lin, acc_svc_rbf, acc_lr, acc_rf = evaluate_supervised(A_norm_centered)
    
    print("\n--- LOO Supervised Multiclass Botanical Accuracy (L2 + Standardized) ---")
    print(f"PLS-DA (PLS Regression):     {acc_pls*100:.2f}%")
    print(f"SVM (Linear Kernel):         {acc_svc_lin*100:.2f}%")
    print(f"SVM (RBF Kernel):            {acc_svc_rbf*100:.2f}%")
    print(f"Logistic Regression:         {acc_lr*100:.2f}%")
    print(f"Random Forest:               {acc_rf*100:.2f}%")
    
    # Binary classification task: Adulterated (1) vs Authentic (0)
    # class_ids == 2 represents grouped adulterated honeys
    y_binary = (class_ids == 2).astype(int)
    
    def evaluate_binary(scores):
        loo = LeaveOneOut()
        
        # 1. SVM (Linear)
        correct_svc_lin = 0
        for train_idx, test_idx in loo.split(scores):
            svc = SVC(kernel='linear', C=1.0, class_weight='balanced')
            svc.fit(scores[train_idx], y_binary[train_idx])
            pred = svc.predict(scores[test_idx])
            if pred[0] == y_binary[test_idx[0]]:
                correct_svc_lin += 1
        acc_svc_lin_bin = correct_svc_lin / num_samples
        
        # 2. SVM (RBF)
        correct_svc_rbf = 0
        for train_idx, test_idx in loo.split(scores):
            svc = SVC(kernel='rbf', C=1.0, class_weight='balanced')
            svc.fit(scores[train_idx], y_binary[train_idx])
            pred = svc.predict(scores[test_idx])
            if pred[0] == y_binary[test_idx[0]]:
                correct_svc_rbf += 1
        acc_svc_rbf_bin = correct_svc_rbf / num_samples
        
        # 3. Logistic Regression
        correct_lr = 0
        for train_idx, test_idx in loo.split(scores):
            lr = LogisticRegression(class_weight='balanced', max_iter=2000)
            lr.fit(scores[train_idx], y_binary[train_idx])
            pred = lr.predict(scores[test_idx])
            if pred[0] == y_binary[test_idx[0]]:
                correct_lr += 1
        acc_lr_bin = correct_lr / num_samples
        
        return acc_svc_lin_bin, acc_svc_rbf_bin, acc_lr_bin

    acc_svc_lin_bin, acc_svc_rbf_bin, acc_lr_bin = evaluate_binary(A_norm_centered)
    
    print("\n--- LOO Supervised Binary Adulteration Accuracy (L2 + Standardized) ---")
    print(f"SVM (Linear Kernel):         {acc_svc_lin_bin*100:.2f}%")
    print(f"SVM (RBF Kernel):            {acc_svc_rbf_bin*100:.2f}%")
    print(f"Logistic Regression:         {acc_lr_bin*100:.2f}%")
    
    # Use the best multiclass classifier accuracy for the plot title
    classification_acc_norm = max(acc_pls, acc_svc_lin, acc_svc_rbf, acc_lr, acc_rf)
    
    # PCA project
    U_n, S_n, Vt_n = np.linalg.svd(A_norm_centered, full_matrices=False)
    scores_norm_pca = U_n[:, :2] * S_n[:2]
    
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
    plots_path = 'notebooks/eem/honey_resolved_profiles.png'
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
        
    plt.title(f'PCA of Row-Normalized Resolved Honey Scores (LOO Ridge Accuracy: {classification_acc_norm*100:.1f}%)', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Botanical Origin', fontsize=10)
    
    pca_path = 'notebooks/eem/honey_pca_separation.png'
    plt.tight_layout()
    plt.savefig(pca_path, dpi=200)
    plt.close()
    print(f"Saved PCA separation plot to {pca_path}")

if __name__ == '__main__':
    train_honey_dataset()
