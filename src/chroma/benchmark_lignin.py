"""
Benchmark script for Lignin Phenols (HPLC-DAD) dataset.
Saves model outputs and prints evaluation metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.chroma.model import ChromaPETN
from src.chroma.baselines import COWPARAFAC, MCRALS

def calculate_cosine_similarity(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    return np.max([np.dot(v1_norm, v2_norm), np.dot(v1_norm, -v2_norm)])

def match_and_align_profiles(A_pred, C_pred, A_true, C_true, B_pred=None):
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
    B_pred_ordered = np.zeros((B_pred.shape[0], R_true)) if B_pred is not None else None
    
    for r_pred in range(R_pred):
        if r_pred < len(perm):
            true_idx = perm[r_pred]
            if true_idx < R_true:
                A_pred_ordered[:, true_idx] = A_pred[:, r_pred]
                C_pred_ordered[:, true_idx] = C_pred[:, r_pred]
                if B_pred is not None:
                    B_pred_ordered[:, true_idx] = B_pred[:, r_pred]
                
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

def train_chroma_petn_fast(X, num_components, epochs=800, lr=0.015, warp_reg_coef=0.001, warp_type='linear', num_segments=4, derivative_order=0, sg_window_size=11, batch_size=50000, tol=1e-6, patience=50):
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
        batch_size=batch_size
    )

def main():
    print("==================================================")
    print(" 3. BENCHMARKING LIGNIN PHENOLS CALIBRATION DATASET")
    print("==================================================")
    
    # Load StdSpectra
    mat_std = scipy.io.loadmat('data/chroma/lignin/StdSpectra.mat')
    std_spectra = mat_std['StdSpectra'][0, 0]['A'] # Shape (13, 100)
    
    # Load concentration Excel and reconstruct 18x13 concentration matrix
    df_conc = pd.read_excel('data/chroma/lignin/Varied standard concentrations.xlsx')
    headers = [float(x) for x in df_conc.columns]
    data_rows = df_conc.values.astype(float)
    conc_18x13 = np.vstack([[headers], data_rows])
    
    # Load and parse 18 raw files
    data_dir = 'data/chroma/lignin/ASCII files'
    vs_files = sorted([f for f in os.listdir(data_dir) if f.startswith('VS')])
    
    def parse_lignin_file(path):
        with open(path) as f:
            lines = f.readlines()
        pda_idx = [i for i, l in enumerate(lines) if l.startswith('[PDA 3D]')][0]
        pda_lines = lines[pda_idx+1:]
        data_start_idx = 0
        for idx, line in enumerate(pda_lines):
            if line.strip().startswith(','):
                data_start_idx = idx + 1
                break
        times, intensities = [], []
        for line in pda_lines[data_start_idx:]:
            sline = line.strip()
            if not sline: continue
            parts = sline.split(',')
            try:
                times.append(float(parts[0]))
                intensities.append([float(x) for x in parts[1:]])
            except ValueError: break
        return np.array(times), np.array(intensities)
        
    print("  Loading 18 ASCII chromatography files...")
    X_lignin = []
    for f in vs_files:
        times, intensities = parse_lignin_file(os.path.join(data_dir, f))
        # Slice to first 100 wavelengths
        intensities = intensities[:, :100]
        # Crop to elution window [7.0, 18.0] min
        time_mask = (times >= 7.0) & (times <= 18.0)
        intensities = intensities[time_mask, :]
        # Downsample time axis by 2 to speed up training
        intensities = intensities[::2, :]
        X_lignin.append(intensities)
        
    X_lignin = np.array(X_lignin) # Shape (18, 516, 100)
    X_lignin = np.clip(X_lignin, 0.0, None)
    X_max = X_lignin.max()
    X_lignin_norm = X_lignin / X_max
    
    # Fit Models (13 components!)
    print("  Fitting MCR-ALS...")
    mcr = MCRALS(num_components=13, max_iter=100)
    mcr.fit(X_lignin_norm)
    metrics_mcr = match_and_align_profiles(mcr.A_, mcr.C_, conc_18x13, std_spectra.T, B_pred=mcr.B_)
    
    print("  Fitting COW-PARAFAC...")
    cow = COWPARAFAC(num_components=13, N_seg=12, slack=3)
    cow.fit(X_lignin_norm)
    metrics_cow = match_and_align_profiles(cow.A_, cow.C_, conc_18x13, std_spectra.T, B_pred=cow.B_)
    
    print("  Fitting Chroma-PETN (Linear)...")
    torch.manual_seed(42)
    petn_lin = train_chroma_petn_fast(X_lignin_norm, num_components=13, warp_type='linear', epochs=800, derivative_order=2, sg_window_size=15)
    A_petn_lin = petn_lin.sample_embeddings.weight.detach().cpu().numpy()
    B_petn_lin = petn_lin.time_embeddings.weight.detach().cpu().numpy()
    C_petn_lin = petn_lin.spec_embeddings.weight.detach().cpu().numpy()
    metrics_petn_lin = match_and_align_profiles(A_petn_lin, C_petn_lin, conc_18x13, std_spectra.T, B_pred=B_petn_lin)
    
    print("  Fitting Chroma-PETN (Spline)...")
    torch.manual_seed(42)
    petn_spl = train_chroma_petn_fast(X_lignin_norm, num_components=13, warp_type='spline', num_segments=4, epochs=800, derivative_order=2, sg_window_size=15)
    A_petn_spl = petn_spl.sample_embeddings.weight.detach().cpu().numpy()
    B_petn_spl = petn_spl.time_embeddings.weight.detach().cpu().numpy()
    C_petn_spl = petn_spl.spec_embeddings.weight.detach().cpu().numpy()
    metrics_petn_spl = match_and_align_profiles(A_petn_spl, C_petn_spl, conc_18x13, std_spectra.T, B_pred=B_petn_spl)
    
    print("\nLignin Phenols Results Summary:")
    print(f"  MCR-ALS     | Scores R^2: {metrics_mcr['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_mcr['mean_c_sim']**2:.4f}")
    print(f"  COW-PARAFAC | Scores R^2: {metrics_cow['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_cow['mean_c_sim']**2:.4f}")
    print(f"  PETN (Lin)  | Scores R^2: {metrics_petn_lin['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_petn_lin['mean_c_sim']**2:.4f}")
    print(f"  PETN (Spl)  | Scores R^2: {metrics_petn_spl['mean_a_sim']**2:.4f} | Spectra R^2: {metrics_petn_spl['mean_c_sim']**2:.4f}")
    
    # Save results to file
    out_dir = 'notebooks/chroma/results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'lignin_results.npz')
    np.savez(
        out_path,
        A_true=conc_18x13, C_true=std_spectra.T,
        mcr_A=metrics_mcr['a_ordered'], mcr_C=metrics_mcr['c_ordered'], mcr_B=metrics_mcr['b_ordered'],
        cow_A=metrics_cow['a_ordered'], cow_C=metrics_cow['c_ordered'], cow_B=metrics_cow['b_ordered'],
        petn_lin_A=metrics_petn_lin['a_ordered'], petn_lin_C=metrics_petn_lin['c_ordered'], petn_lin_B=metrics_petn_lin['b_ordered'],
        petn_spl_A=metrics_petn_spl['a_ordered'], petn_spl_C=metrics_petn_spl['c_ordered'], petn_spl_B=metrics_petn_spl['b_ordered']
    )
    print(f"Saved model outputs to: {out_path}")

if __name__ == '__main__':
    main()
