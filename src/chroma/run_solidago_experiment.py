"""
Solidago altissima HPLC-DAD Real-World Experiment Runner.
Loads the preprocessed Solidago root extract chromatograms, trains HPLC_PETN
with baseline removal (2nd-derivative Savitzky-Golay filters) and linear warping,
exports resolved loadings, and generates diagnostic visualizations.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rdata

from src.chroma import HPLC_PETN, extract_loadings, plot_alignment_verification
from src.chroma.train import train_chroma_petn

def load_solidago_preprocessed(data_dir):
    """
    Loads preprocessed Solidago altissima HPLC-DAD chromatograms and matches them with metadata.
    """
    # 1. Load Metadata
    metadata_path = os.path.join(data_dir, "Sa_metadata.csv")
    metadata = pd.read_csv(metadata_path)
    metadata['vial'] = metadata['vial'].astype(str)
    
    # 2. Parse RData
    rdata_path = os.path.join(data_dir, "Sa_pr.RData")
    parsed = rdata.parser.parse_file(rdata_path)
    converted = rdata.conversion.convert(parsed)
    sa_pr = converted['Sa_pr']
    
    # Samples are stored as keys matching the vial numbers
    sample_keys = [str(v) for v in metadata['vial']]
    
    # 3. Read Coordinates from the first sample
    first_sample = sa_pr[sample_keys[0]]
    time_coords = first_sample.coords['dim_0'].values.astype(float)
    wavelength_coords = first_sample.coords['dim_1'].values.astype(float)
    
    # 4. Construct the 3D Tensor: (Samples, Time, Wavelengths)
    I, J, K = len(sample_keys), len(time_coords), len(wavelength_coords)
    X = np.zeros((I, J, K))
    for idx, sample_id in enumerate(sample_keys):
        X[idx] = sa_pr[sample_id].values
        
    return X, metadata, time_coords, wavelength_coords

def plot_resolved_profiles(A, B, C, metadata, time_coords, wavelength_coords, save_path):
    """
    Plots the resolved UV spectra, canonical chromatograms, and sample scores.
    """
    num_components = A.shape[1]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Plot Pure UV Spectra (C)
    for r in range(num_components):
        axes[0].plot(wavelength_coords, C[:, r], label=f"Component {r+1}", color=colors[r], linewidth=2.0)
    axes[0].set_xlabel("Wavelength (nm)", fontsize=11)
    axes[0].set_ylabel("Normalized Absorbance", fontsize=11)
    axes[0].set_title("Resolved UV-Vis Spectra (C Loading)", fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc="upper right")
    
    # 2. Plot Chromatography Profiles (B)
    for r in range(num_components):
        axes[1].plot(time_coords, B[:, r], label=f"Component {r+1}", color=colors[r], linewidth=2.0)
    axes[1].set_xlabel("Retention Time (min)", fontsize=11)
    axes[1].set_ylabel("Normalized Intensity", fontsize=11)
    axes[1].set_title("Canonical Chromatograms (B Loading)", fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # 3. Plot Scores (A)
    x_indices = np.arange(len(metadata))
    vials = metadata['vial'].tolist()
    trts = metadata['trt'].tolist()
    
    bar_width = 0.22
    for r in range(num_components):
        offset = (r - (num_components - 1) / 2) * bar_width
        axes[2].bar(x_indices + offset, A[:, r], bar_width, label=f"Component {r+1}", color=colors[r], alpha=0.85)
        
    axes[2].set_xticks(x_indices)
    x_labels = [f"Vial {v}\n(trt: {t})" for v, t in zip(vials, trts)]
    axes[2].set_xticklabels(x_labels, fontsize=10)
    axes[2].set_ylabel("Score Intensity (mAU * min)", fontsize=11)
    axes[2].set_title("Sample Scores (A Loading) by Treatment", fontsize=12, fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.4, axis='y')
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved resolved profiles plot to: {save_path}")
    plt.close()

def run_solidago_experiment():
    print("==============================================================")
    print("RUNNING CHROMA-PETN EXPERIMENT ON REAL SOLIDAGO ROOT EXTRACTS")
    print("==============================================================")
    
    # 1. Load Data
    data_dir = "data/chroma/solidago"
    print(f"Loading preprocessed Solidago data from: {data_dir}...")
    X, metadata, time_coords, wavelength_coords = load_solidago_preprocessed(data_dir)
    print(f"Loaded dataset: X shape = {X.shape} | Samples = {X.shape[0]}, Time steps = {X.shape[1]}, Wavelengths = {X.shape[2]}")
    print(f"Sample Metadata:\n{metadata}\n")
    
    save_dir = "notebooks/chroma/experiments/solidago"
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Train Model
    # Fit 3 components using linear warping and a second-derivative Savitzky-Golay filter
    # full-grid mode batch_size=None, compile_model=False for platform compatibility
    print("Training HPLC-PETN model...")
    model = train_chroma_petn(
        dataset=X,
        epochs=1200,
        lr=0.015,
        warp_reg_coef=0.001,
        warp_type='linear',
        num_components=3,
        derivative_order=2,
        sg_window_size=11,
        sg_polyorder=2,
        batch_size=None,
        compile_model=False,
        patience=100,
        tol=1e-6,
        lambda_raw=0.0,
        lambda_smooth_B=0.01,  # smooth peaks penalty
        model_type='hplc',
        init_svd=True
    )
    
    # 3. Extract and Clean Loadings (Resolve scaling ambiguities)
    print("\nExtracting and normalizing resolved loadings...")
    loadings = extract_loadings(model)
    A, B, C = loadings['A'], loadings['B'], loadings['C']
    R = A.shape[1]
    
    # Normalize profiles so that chromatography profiles B and spectra C have unit norm,
    # and all amplitude is absorbed into sample scores A.
    for r in range(R):
        norm_b = np.linalg.norm(B[:, r]) + 1e-10
        norm_c = np.linalg.norm(C[:, r]) + 1e-10
        B[:, r] /= norm_b
        C[:, r] /= norm_c
        A[:, r] *= (norm_b * norm_c)
        
    # Save CSVs
    comp_names = [f"Component_{r+1}" for r in range(R)]
    df_A = pd.DataFrame(A, index=metadata['vial'].tolist(), columns=comp_names)
    df_B = pd.DataFrame(B, index=time_coords, columns=comp_names)
    df_C = pd.DataFrame(C, index=wavelength_coords, columns=comp_names)
    
    df_A.to_csv(os.path.join(save_dir, "resolved_scores.csv"))
    df_B.to_csv(os.path.join(save_dir, "resolved_chromatograms.csv"))
    df_C.to_csv(os.path.join(save_dir, "resolved_spectra.csv"))
    print(f"CSVs exported to: {save_dir}/")
    
    # 4. Generate Diagnostic Plots
    print("\nGenerating diagnostic plots...")
    # Alignment verification
    plot_path_align = os.path.join(save_dir, 'solidago_alignment_verification.png')
    plot_alignment_verification(model, X, save_path=plot_path_align)
    
    # Resolved profiles
    plot_path_resolved = os.path.join(save_dir, 'solidago_resolved_profiles.png')
    plot_resolved_profiles(A, B, C, metadata, time_coords, wavelength_coords, save_path=plot_path_resolved)
    
    # 5. Print Learned Warping Parameters
    print("\nLearned Warping Parameters per Sample (Mean-Centered):")
    raw_model = getattr(model, '_orig_mod', model)
    alpha_learned = raw_model.alpha.detach().cpu().numpy()
    beta_learned = raw_model.beta.detach().cpu().numpy()
    
    warp_df = pd.DataFrame({
        'vial': metadata['vial'],
        'trt': metadata['trt'],
        'alpha_C1': alpha_learned[:, 0],
        'alpha_C2': alpha_learned[:, 1],
        'alpha_C3': alpha_learned[:, 2],
        'beta_C1': beta_learned[:, 0],
        'beta_C2': beta_learned[:, 1],
        'beta_C3': beta_learned[:, 2],
    })
    print(warp_df.to_string(index=False))
    
    print("\n==============================================================")
    print("SOLIDAGO CHROMATOGRAPHY ALIGNMENT EXPERIMENT COMPLETED")
    print("==============================================================")

if __name__ == '__main__':
    run_solidago_experiment()
