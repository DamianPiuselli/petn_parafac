"""
Benchmark script for Solidago Root Extracts (HPLC-DAD) dataset.
Saves model outputs and prints evaluation metrics.
"""

import os
import sys
import numpy as np
import torch
import rdata

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.chroma.model import ChromaPETN
from src.chroma.baselines import COWPARAFAC, MCRALS

def calculate_cosine_similarity(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    return np.max([np.dot(v1_norm, v2_norm), np.dot(v1_norm, -v2_norm)])

from src.chroma.train import train_chroma_petn

def train_chroma_petn_fast(X, num_components, epochs=800, lr=0.015, warp_reg_coef=0.001, warp_type='linear', num_segments=4, tol=1e-6, patience=50, batch_size=None, compile_model=True):

    return train_chroma_petn(
        dataset=X,
        epochs=epochs,
        lr=lr,
        warp_reg_coef=warp_reg_coef,
        warp_type=warp_type,
        num_segments=num_segments,
        tol=tol,
        patience=patience,
        num_components=num_components,
        batch_size=batch_size,
        compile_model=compile_model
    )


def main():
    print("==================================================")
    print(" 2. BENCHMARKING SOLIDAGO ROOT EXTRACTS DATASET")
    print("==================================================")
    
    # Load processed data
    parsed_pr = rdata.parser.parse_file('data/chroma/solidago/Sa_pr.RData')
    converted_pr = rdata.conversion.convert(parsed_pr)
    obj_pr = converted_pr['Sa_pr']
    
    # Load reference aligned data
    parsed_warp = rdata.parser.parse_file('data/chroma/solidago/Sa_warp.RData')
    converted_warp = rdata.conversion.convert(parsed_warp)
    obj_warp = converted_warp['Sa_warp']
    
    vials = ['119', '121', '122', '458']
    
    # Construct raw data tensor X (4, 434, 60)
    X_raw = np.array([obj_pr[v].values for v in vials])
    X_ref_aligned = np.array([obj_warp[v].values for v in vials])
    X_ref_aligned = np.nan_to_num(X_ref_aligned, nan=0.0)
    
    # Normalize inputs
    X_raw = np.clip(X_raw, 0.0, None)
    X_raw_norm = X_raw / X_raw.max()
    
    # Fit models (4 components)
    print("  Fitting MCR-ALS...")
    mcr = MCRALS(num_components=4, max_iter=100)
    mcr.fit(X_raw_norm)
    X_mcr_pred = np.einsum('ir,jr,kr->ijk', mcr.A_, mcr.B_, mcr.C_)
    r2_mcr = 1.0 - (np.sum((X_raw_norm - X_mcr_pred)**2) / np.sum(X_raw_norm**2))
    
    print("  Fitting COW-PARAFAC...")
    cow = COWPARAFAC(num_components=4, N_seg=15, slack=5)
    cow.fit(X_raw_norm)
    X_cow_pred = np.einsum('ir,jr,kr->ijk', cow.A_, cow.B_, cow.C_)
    cow_aligned_tic = np.sum(cow.X_aligned_, axis=2)
    ref_aligned_tic = np.sum(X_ref_aligned, axis=2)
    cow_align_sim = np.mean([calculate_cosine_similarity(cow_aligned_tic[i], ref_aligned_tic[i]) for i in range(4)])
    r2_cow = 1.0 - (np.sum((cow.X_aligned_/cow.X_aligned_.max() - X_cow_pred)**2) / np.sum((cow.X_aligned_/cow.X_aligned_.max())**2))
    
    print("  Fitting Chroma-PETN (Linear)...")
    torch.manual_seed(42)
    petn_lin = train_chroma_petn_fast(X_raw_norm, num_components=4, warp_type='linear')
    A_petn_lin = petn_lin.sample_embeddings.weight.detach().cpu().numpy()
    B_petn_lin = petn_lin.time_embeddings.weight.detach().cpu().numpy()
    C_petn_lin = petn_lin.spec_embeddings.weight.detach().cpu().numpy()
    with torch.no_grad():
        device_lin = next(petn_lin.parameters()).device
        coords_i, coords_j, coords_k = torch.meshgrid(
            torch.arange(4, device=device_lin),
            torch.arange(434, device=device_lin),
            torch.arange(60, device=device_lin),
            indexing='ij'
        )
        y_pred_lin = petn_lin(coords_i.flatten(), coords_j.flatten(), coords_k.flatten()).cpu().numpy().reshape(4, 434, 60)
    r2_petn_lin = 1.0 - (np.sum((X_raw_norm - y_pred_lin)**2) / np.sum(X_raw_norm**2))
    
    # Extract aligned profiles for linear PETN model
    t_observed = np.linspace(0.0, 1.0, 434)
    petn_lin_aligned_tics = []
    for i in range(4):
        stretch = petn_lin.warp_stretch[i].item()
        shift = petn_lin.warp_shift[i].item()
        t_warped = t_observed - (stretch * t_observed + shift)
        x_warped = np.clip(t_warped * 433, 0, 432.99)
        x0 = np.floor(x_warped).astype(int)
        x1 = x0 + 1
        w = x_warped - x0
        val0 = B_petn_lin[x0]
        val1 = B_petn_lin[x1]
        b_warped = (1.0 - w[:, None]) * val0 + w[:, None] * val1
        petn_lin_aligned_tics.append(np.sum(b_warped @ C_petn_lin.T, axis=1) * np.sum(A_petn_lin[i]))
    petn_lin_align_sim = np.mean([calculate_cosine_similarity(petn_lin_aligned_tics[i], ref_aligned_tic[i]) for i in range(4)])
    
    print("  Fitting Chroma-PETN (Spline)...")
    torch.manual_seed(42)
    petn_spl = train_chroma_petn_fast(X_raw_norm, num_components=4, warp_type='spline', num_segments=5)
    A_petn_spl = petn_spl.sample_embeddings.weight.detach().cpu().numpy()
    B_petn_spl = petn_spl.time_embeddings.weight.detach().cpu().numpy()
    C_petn_spl = petn_spl.spec_embeddings.weight.detach().cpu().numpy()
    with torch.no_grad():
        device_spl = next(petn_spl.parameters()).device
        coords_i_spl, coords_j_spl, coords_k_spl = torch.meshgrid(
            torch.arange(4, device=device_spl),
            torch.arange(434, device=device_spl),
            torch.arange(60, device=device_spl),
            indexing='ij'
        )
        y_pred_spl = petn_spl(coords_i_spl.flatten(), coords_j_spl.flatten(), coords_k_spl.flatten()).cpu().numpy().reshape(4, 434, 60)
    r2_petn_spl = 1.0 - (np.sum((X_raw_norm - y_pred_spl)**2) / np.sum(X_raw_norm**2))

    
    # Extract aligned profiles from spline PETN model
    petn_spl_aligned_tics = []
    for i in range(4):
        shift = petn_spl.warp_shift[i].item()
        log_inc = petn_spl.warp_log_increments[i].detach().cpu().numpy()
        inc = (1.0 / 5) * np.exp(log_inc)
        w_knots = shift + np.cumsum(np.concatenate([[0.0], inc]))
        val = t_observed * 5
        k = np.clip(np.floor(val).astype(int), 0, 4)
        u = val - k
        t_warped = (1.0 - u) * w_knots[k] + u * w_knots[k+1]
        x_warped = np.clip(t_warped * 433, 0, 432.99)
        x0 = np.floor(x_warped).astype(int)
        x1 = x0 + 1
        w = x_warped - x0
        val0 = B_petn_spl[x0]
        val1 = B_petn_spl[x1]
        b_warped = (1.0 - w[:, None]) * val0 + w[:, None] * val1
        petn_spl_aligned_tics.append(np.sum(b_warped @ C_petn_spl.T, axis=1) * np.sum(A_petn_spl[i]))
    petn_spl_align_sim = np.mean([calculate_cosine_similarity(petn_spl_aligned_tics[i], ref_aligned_tic[i]) for i in range(4)])
    
    print("\nSolidago Root Extracts Results Summary:")
    print(f"  MCR-ALS     | Recon R^2: {r2_mcr:.4f}")
    print(f"  COW-PARAFAC | Recon R^2: {r2_cow:.4f} | Alignment Sim: {cow_align_sim:.4f}")
    print(f"  PETN (Lin)  | Recon R^2: {r2_petn_lin:.4f} | Alignment Sim: {petn_lin_align_sim:.4f}")
    print(f"  PETN (Spl)  | Recon R^2: {r2_petn_spl:.4f} | Alignment Sim: {petn_spl_align_sim:.4f}")
    
    # Save results to file
    out_dir = 'notebooks/chroma/results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'solidago_results.npz')
    np.savez(
        out_path,
        X_raw=X_raw_norm, X_ref_aligned=X_ref_aligned,
        mcr_A=mcr.A_, mcr_B=mcr.B_, mcr_C=mcr.C_,
        cow_A=cow.A_, cow_B=cow.B_, cow_C=cow.C_, cow_X_aligned=cow.X_aligned_,
        petn_lin_A=A_petn_lin, petn_lin_B=B_petn_lin, petn_lin_C=C_petn_lin,
        petn_spl_A=A_petn_spl, petn_spl_B=B_petn_spl, petn_spl_C=C_petn_spl
    )
    print(f"Saved model outputs to: {out_path}")

if __name__ == '__main__':
    main()
