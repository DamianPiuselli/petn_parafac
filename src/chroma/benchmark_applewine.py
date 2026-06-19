"""
Benchmark script for Copenhagen Apple Wine GC-MS dataset.
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

def match_and_align_profiles(A_pred, B_pred, C_pred, A_true, B_true, C_true):
    A_pred = A_pred.astype(float).copy()
    B_pred = B_pred.astype(float).copy() if B_pred is not None else None
    C_pred = C_pred.astype(float).copy()
    A_true = A_true.astype(float).copy()
    B_true = B_true.astype(float).copy() if B_true is not None else None
    C_true = C_true.astype(float).copy()
    
    R_pred = A_pred.shape[1]
    R_true = A_true.shape[1]
    
    sim_matrix = np.zeros((R_pred, R_true))
    for r_pred in range(R_pred):
        for r_true in range(R_true):
            sim_matrix[r_pred, r_true] = calculate_cosine_similarity(C_pred[:, r_pred], C_true[:, r_true])
            
    from scipy.optimize import linear_sum_assignment
    cost_matrix = 1.0 - sim_matrix
    pred_ind, true_ind = linear_sum_assignment(cost_matrix)
    perm = [0] * R_pred
    for p_idx, t_idx in zip(pred_ind, true_ind):
        perm[p_idx] = t_idx
        
    A_pred_ordered = np.zeros((A_pred.shape[0], R_true))
    B_pred_ordered = np.zeros((B_pred.shape[0], R_true)) if B_pred is not None else None
    C_pred_ordered = np.zeros((C_pred.shape[0], R_true))
    
    for r_pred in range(R_pred):
        if r_pred < len(perm):
            true_idx = perm[r_pred]
            if true_idx < R_true:
                A_pred_ordered[:, true_idx] = A_pred[:, r_pred]
                if B_pred is not None:
                    B_pred_ordered[:, true_idx] = B_pred[:, r_pred]
                C_pred_ordered[:, true_idx] = C_pred[:, r_pred]
                
    for r in range(R_true):
        # Normalize B
        if B_pred_ordered is not None:
            norm_b = np.linalg.norm(B_pred_ordered[:, r])
            if norm_b > 0:
                B_pred_ordered[:, r] /= norm_b
                A_pred_ordered[:, r] *= norm_b
            
        if B_true is not None:
            norm_b_true = np.linalg.norm(B_true[:, r])
            if norm_b_true > 0:
                B_true[:, r] /= norm_b_true
                A_true[:, r] *= norm_b_true

        # Normalize C
        norm_c = np.linalg.norm(C_pred_ordered[:, r])
        if norm_c > 0:
            C_pred_ordered[:, r] /= norm_c
            A_pred_ordered[:, r] *= norm_c
            
        norm_c_true = np.linalg.norm(C_true[:, r])
        if norm_c_true > 0:
            C_true[:, r] /= norm_c_true
            A_true[:, r] *= norm_c_true
            
    a_sims = [calculate_cosine_similarity(A_pred_ordered[:, r], A_true[:, r]) for r in range(R_true)]
    b_sims = [calculate_cosine_similarity(B_pred_ordered[:, r], B_true[:, r]) for r in range(R_true)] if B_pred_ordered is not None and B_true is not None else [0.0]*R_true
    c_sims = [calculate_cosine_similarity(C_pred_ordered[:, r], C_true[:, r]) for r in range(R_true)]
    
    res = {
        'a_ordered': A_pred_ordered,
        'c_ordered': C_pred_ordered,
        'a_similarities': a_sims,
        'c_similarities': c_sims,
        'mean_a_sim': np.mean(a_sims),
        'mean_c_sim': np.mean(c_sims)
    }
    if B_pred_ordered is not None:
        res['b_ordered'] = B_pred_ordered
    return res

from src.chroma.train import train_chroma_petn

def train_chroma_petn_fast(X, num_components, epochs=800, lr=0.015, warp_reg_coef=0.001, warp_type='linear', num_segments=4, derivative_order=0, sg_window_size=11, batch_size=None, tol=1e-6, patience=50, compile_model=True):

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
        derivative_order=derivative_order,
        sg_window_size=sg_window_size,
        batch_size=batch_size,
        compile_model=compile_model
    )


def main():
    print("==================================================")
    print(" 4. BENCHMARKING COPENHAGEN APPLE WINE DATASET")
    print("==================================================")
    
    # Load preprocessed data Interval 1
    mat_int = scipy.io.loadmat('data/chroma/applewine/CDFS/Intervals.mat')
    X_int1 = mat_int['Int1'].astype(float) # Shape (155, 94, 286)
    
    # Load reference literature model
    mat_mod = scipy.io.loadmat('data/chroma/applewine/CDFS/Models.mat')
    m1_ref = mat_mod['m1'][0, 0]
    
    # Extract literature scores and spectra for 4 components (index 3)
    A_lit_spec = m1_ref['A'][0, 3].astype(float) # Shape (286, 4) - mass spectra
    C_lit_scores = m1_ref['C'][0, 3].astype(float) # Shape (155, 4) - scores
    
    A_true_scores = C_lit_scores
    C_true_spec = A_lit_spec
    
    # Normalize inputs
    X_max = X_int1.max()
    X_int1_norm = X_int1 / X_max
    
    # Fit Models (4 components!)
    print("  Fitting MCR-ALS...")
    mcr = MCRALS(num_components=4, max_iter=100)
    mcr.fit(X_int1_norm)
    metrics_mcr = match_and_align_profiles(mcr.A_, mcr.B_, mcr.C_, A_true_scores, None, C_true_spec)
    
    print("  Fitting COW-PARAFAC...")
    cow = COWPARAFAC(num_components=4, N_seg=8, slack=3)
    cow.fit(X_int1_norm)
    metrics_cow = match_and_align_profiles(cow.A_, cow.B_, cow.C_, A_true_scores, None, C_true_spec)
    
    print("  Fitting Chroma-PETN (Linear)...")
    torch.manual_seed(42)
    petn_lin = train_chroma_petn_fast(X_int1_norm, num_components=4, warp_type='linear', epochs=800)
    A_petn_lin = petn_lin.sample_embeddings.weight.detach().cpu().numpy()
    B_petn_lin = petn_lin.time_embeddings.weight.detach().cpu().numpy()
    C_petn_lin = petn_lin.spec_embeddings.weight.detach().cpu().numpy()
    metrics_petn_lin = match_and_align_profiles(A_petn_lin, B_petn_lin, C_petn_lin, A_true_scores, None, C_true_spec)
    
    print("  Fitting Chroma-PETN (Spline)...")
    torch.manual_seed(42)
    petn_spl = train_chroma_petn_fast(X_int1_norm, num_components=4, warp_type='spline', num_segments=4, epochs=800)
    A_petn_spl = petn_spl.sample_embeddings.weight.detach().cpu().numpy()
    B_petn_spl = petn_spl.time_embeddings.weight.detach().cpu().numpy()
    C_petn_spl = petn_spl.spec_embeddings.weight.detach().cpu().numpy()
    metrics_petn_spl = match_and_align_profiles(A_petn_spl, B_petn_spl, C_petn_spl, A_true_scores, None, C_true_spec)
    
    print("\nApple Wine GC-MS Results Summary:")
    print(f"  MCR-ALS     | Scores R^2: {metrics_mcr['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_mcr['mean_c_sim']**2:.4f}")
    print(f"  COW-PARAFAC | Scores R^2: {metrics_cow['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_cow['mean_c_sim']**2:.4f}")
    print(f"  PETN (Lin)  | Scores R^2: {metrics_petn_lin['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_petn_lin['mean_c_sim']**2:.4f}")
    print(f"  PETN (Spl)  | Scores R^2: {metrics_petn_spl['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_petn_spl['mean_c_sim']**2:.4f}")
    
    # Save results to file
    out_dir = 'notebooks/chroma/results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'applewine_results.npz')
    np.savez(
        out_path,
        A_true=A_true_scores, C_true=C_true_spec,
        mcr_A=metrics_mcr['a_ordered'], mcr_C=metrics_mcr['c_ordered'], mcr_B=metrics_mcr.get('b_ordered', None),
        cow_A=metrics_cow['a_ordered'], cow_C=metrics_cow['c_ordered'], cow_B=metrics_cow.get('b_ordered', None),
        petn_lin_A=metrics_petn_lin['a_ordered'], petn_lin_C=metrics_petn_lin['c_ordered'], petn_lin_B=metrics_petn_lin['b_ordered'],
        petn_spl_A=metrics_petn_spl['a_ordered'], petn_spl_C=metrics_petn_spl['c_ordered'], petn_spl_B=metrics_petn_spl['b_ordered']
    )
    print(f"Saved model outputs to: {out_path}")

    # Generate and save comparison plots
    print("Generating Apple Wine GC-MS performance plots...")
    from src.common.utils import (
        plot_chroma_resolved_vs_true_profiles,
        plot_chroma_alignment_comparison,
        plot_scores_parity
    )
    plot_dir = 'notebooks/chroma'
    os.makedirs(plot_dir, exist_ok=True)
    
    time_grid = np.linspace(0.0, 1.0, X_int1.shape[1])
    spec_grid = mat_mod['mz'].flatten()
    
    applewine_names = [
        "Component 1 (Secondary Aliphatic Ester/Ketone)",
        "Component 2 (Ambient Air/Nitrogen)",
        "Component 3 (Acetaldehyde/CO2)",
        "Component 4 (Isoamyl Acetate)"
    ]

    # 1. Resolved profiles comparison (Linear model vs Literature)
    plot_chroma_resolved_vs_true_profiles(
        None, C_true_spec, metrics_petn_lin['b_ordered'], metrics_petn_lin['c_ordered'],
        time_grid, spec_grid, component_names=applewine_names,
        save_path=os.path.join(plot_dir, 'applewine_resolved_profiles.png')
    )
    
    # 2. Alignment comparison (Observed vs Aligned Chromatograms)
    X_aligned = np.einsum('ir,jr,kr->ijk', metrics_petn_lin['a_ordered'], metrics_petn_lin['b_ordered'], metrics_petn_lin['c_ordered'])
    plot_chroma_alignment_comparison(
        time_grid, X_int1_norm, X_aligned,
        save_path=os.path.join(plot_dir, 'applewine_alignment_comparison.png')
    )
    
    # 3. Scores (concentrations) parity comparison plot against Literature Model (with Cal. Standards)
    plot_scores_parity(
        A_true_scores, metrics_petn_lin['a_ordered'], num_calib=5, components_to_plot=[3],
        component_names=applewine_names,
        save_path=os.path.join(plot_dir, 'applewine_scores_quantification.png')
    )

if __name__ == '__main__':
    main()
