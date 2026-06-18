"""
Benchmark script for UCPH Simulated GC-MS dataset.
Saves model outputs and prints evaluation metrics.
"""

import os
import sys
import numpy as np
import scipy.io
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.chroma.model import ChromaPETN
from src.chroma.baselines import COWPARAFAC, MCRALS

def calculate_cosine_similarity(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    return np.max([np.dot(v1_norm, v2_norm), np.dot(v1_norm, -v2_norm)])

def match_and_align_profiles(A_pred, C_pred, A_true, C_true):
    A_pred = A_pred.astype(float).copy()
    C_pred = C_pred.astype(float).copy()
    A_true = A_true.astype(float).copy()
    C_true = C_true.astype(float).copy()
    
    R_pred = A_pred.shape[1]
    R_true = A_true.shape[1]
    
    sim_matrix = np.zeros((R_pred, R_true))
    for r_pred in range(R_pred):
        for r_true in range(R_true):
            sim_matrix[r_pred, r_true] = calculate_cosine_similarity(C_pred[:, r_pred], C_true[:, r_true])
            
    perm = []
    used = set()
    for r_pred in range(R_pred):
        best_sim = -1.0
        best_idx = 0
        for r_true in range(R_true):
            if r_true in used:
                continue
            sim = sim_matrix[r_pred, r_true]
            if sim > best_sim:
                best_sim = sim
                best_idx = r_true
        perm.append(best_idx)
        used.add(best_idx)
        
    A_pred_ordered = np.zeros((A_pred.shape[0], R_true))
    C_pred_ordered = np.zeros((C_pred.shape[0], R_true))
    
    for r_pred in range(R_pred):
        if r_pred < len(perm):
            true_idx = perm[r_pred]
            if true_idx < R_true:
                A_pred_ordered[:, true_idx] = A_pred[:, r_pred]
                C_pred_ordered[:, true_idx] = C_pred[:, r_pred]
                
    for r in range(R_true):
        norm_c = np.linalg.norm(C_pred_ordered[:, r])
        if norm_c > 0:
            C_pred_ordered[:, r] /= norm_c
            A_pred_ordered[:, r] *= norm_c
        norm_c_true = np.linalg.norm(C_true[:, r])
        if norm_c_true > 0:
            C_true[:, r] /= norm_c_true
            A_true[:, r] *= norm_c_true
            
    a_sims = [calculate_cosine_similarity(A_pred_ordered[:, r], A_true[:, r]) for r in range(R_true)]
    c_sims = [calculate_cosine_similarity(C_pred_ordered[:, r], C_true[:, r]) for r in range(R_true)]
    
    return {
        'a_ordered': A_pred_ordered,
        'c_ordered': C_pred_ordered,
        'a_similarities': a_sims,
        'c_similarities': c_sims,
        'mean_a_sim': np.mean(a_sims),
        'mean_c_sim': np.mean(c_sims)
    }

def train_chroma_petn_fast(X, num_components, epochs=800, lr=0.015, warp_reg_coef=0.001, warp_type='linear', num_segments=4):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    I, J, K = X_tensor.shape
    
    model = ChromaPETN(num_samples=I, num_time=J, num_spec=K, num_components=num_components, warp_type=warp_type, num_segments=num_segments)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    coords_i, coords_j, coords_k = torch.meshgrid(
        torch.arange(I), torch.arange(J), torch.arange(K), indexing='ij'
    )
    coords_i = coords_i.flatten()
    coords_j = coords_j.flatten()
    coords_k = coords_k.flatten()
    y_target = X_tensor[coords_i, coords_j, coords_k]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(coords_i, coords_j, coords_k)
        loss_mse = torch.nn.functional.mse_loss(y_pred, y_target)
        
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
        
        if (epoch + 1) % 200 == 0:
            print(f"    Epoch {epoch+1:4d}/{epochs} | MSE Loss: {loss_mse.item():.6f} | Reg: {loss_warp_reg.item():.6f}")
            
    return model

def main():
    print("==================================================")
    print(" 1. BENCHMARKING UCPH SIMULATED GC-MS DATASET")
    print("==================================================")
    
    mat = scipy.io.loadmat('data/chroma/simulated/Data/SixMass.mat')
    A00 = mat['A00'][:150, :]
    
    np.random.seed(42)
    comp_indices = [0, 2, 4]
    C_true_spec = A00[:, comp_indices]
    C_true_spec = C_true_spec / np.sqrt(np.sum(C_true_spec**2, axis=0))
    
    num_time = 70
    num_samples = 50
    t = np.linspace(0, 1, num_time)
    B_pure = np.zeros((num_time, 3))
    B_pure[:, 0] = 0.5 * np.ones(num_time)
    B_pure[:, 1] = np.exp(-((t - 0.4)**2)/(2 * 0.08**2))
    B_pure[:, 2] = np.exp(-((t - 0.6)**2)/(2 * 0.06**2))
    B_pure[:, 1] /= np.max(B_pure[:, 1])
    B_pure[:, 2] /= np.max(B_pure[:, 2])
    
    rtshift = np.random.randint(-7, 8, size=(num_samples, 3))
    rtshift[:, 0] = 0
    
    B_samples = np.zeros((num_time, 3, num_samples))
    for i in range(num_samples):
        for f in range(3):
            shift = rtshift[i, f]
            if shift == 0:
                B_samples[:, f, i] = B_pure[:, f]
            elif shift < 0:
                B_samples[:-shift, f, i] = B_pure[shift:, f]
            else:
                B_samples[shift:, f, i] = B_pure[:-shift, f]
                
    A_true_scores = np.random.uniform(0.5, 2.5, size=(num_samples, 3))
    A_true_scores = A_true_scores / np.sqrt(np.sum(A_true_scores**2, axis=0))
    
    X_clean = np.zeros((num_samples, num_time, 150))
    for i in range(num_samples):
        for f in range(3):
            X_clean[i, :, :] += A_true_scores[i, f] * np.outer(B_samples[:, f, i], C_true_spec[:, f])
            
    noise = np.random.normal(0, 1, X_clean.shape)
    X_noisy = X_clean + noise * (np.std(X_clean) / np.std(noise) * 0.05)
    X_noisy_norm = X_noisy / X_noisy.max()
    
    # Run models
    print("  Fitting MCR-ALS...")
    mcr = MCRALS(num_components=3, max_iter=100)
    mcr.fit(X_noisy_norm)
    metrics_mcr = match_and_align_profiles(mcr.A_, mcr.C_, A_true_scores, C_true_spec)
    
    print("  Fitting COW-PARAFAC...")
    cow = COWPARAFAC(num_components=3, N_seg=8, slack=3)
    cow.fit(X_noisy_norm)
    metrics_cow = match_and_align_profiles(cow.A_, cow.C_, A_true_scores, C_true_spec)
    
    print("  Fitting Chroma-PETN (Linear)...")
    torch.manual_seed(42)
    petn_lin = train_chroma_petn_fast(X_noisy_norm, num_components=3, warp_type='linear')
    A_petn_lin = petn_lin.sample_embeddings.weight.detach().cpu().numpy()
    B_petn_lin = petn_lin.time_embeddings.weight.detach().cpu().numpy()
    C_petn_lin = petn_lin.spec_embeddings.weight.detach().cpu().numpy()
    metrics_petn_lin = match_and_align_profiles(A_petn_lin, C_petn_lin, A_true_scores, C_true_spec)
    
    print("  Fitting Chroma-PETN (Spline)...")
    torch.manual_seed(42)
    petn_spl = train_chroma_petn_fast(X_noisy_norm, num_components=3, warp_type='spline', num_segments=4)
    A_petn_spl = petn_spl.sample_embeddings.weight.detach().cpu().numpy()
    B_petn_spl = petn_spl.time_embeddings.weight.detach().cpu().numpy()
    C_petn_spl = petn_spl.spec_embeddings.weight.detach().cpu().numpy()
    metrics_petn_spl = match_and_align_profiles(A_petn_spl, C_petn_spl, A_true_scores, C_true_spec)
    
    print("\nSimulated GC-MS Results Summary:")
    print(f"  MCR-ALS     | Scores R^2: {metrics_mcr['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_mcr['mean_c_sim']**2:.4f}")
    print(f"  COW-PARAFAC | Scores R^2: {metrics_cow['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_cow['mean_c_sim']**2:.4f}")
    print(f"  PETN (Lin)  | Scores R^2: {metrics_petn_lin['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_petn_lin['mean_c_sim']**2:.4f}")
    print(f"  PETN (Spl)  | Scores R^2: {metrics_petn_spl['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_petn_spl['mean_c_sim']**2:.4f}")
    
    # Save results to file
    out_dir = 'notebooks/chroma/results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'simulated_results.npz')
    np.savez(
        out_path,
        A_true=A_true_scores, C_true=C_true_spec, B_true=B_pure,
        mcr_A=metrics_mcr['a_ordered'], mcr_C=metrics_mcr['c_ordered'], mcr_B=mcr.B_,
        cow_A=metrics_cow['a_ordered'], cow_C=metrics_cow['c_ordered'], cow_B=cow.B_,
        petn_lin_A=metrics_petn_lin['a_ordered'], petn_lin_C=metrics_petn_lin['c_ordered'], petn_lin_B=B_petn_lin,
        petn_spl_A=metrics_petn_spl['a_ordered'], petn_spl_C=metrics_petn_spl['c_ordered'], petn_spl_B=B_petn_spl
    )
    print(f"Saved model outputs to: {out_path}")

if __name__ == '__main__':
    main()
