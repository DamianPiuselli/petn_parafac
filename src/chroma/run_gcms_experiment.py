"""
GC-MS Submodel Experiment Runner.
Simulates realistic GC-MS data with EMG tailing peaks and sparse spectra.
Trains GCMS_PETN in sparse coordinate mode and evaluates alignment.
"""
import os
import torch
import numpy as np

from src.chroma import GCMS_PETN, GCMSDataGenerator, plot_alignment_verification
from src.chroma.train import train_chroma_petn, evaluate_chroma_alignment

def run_gcms_experiment():
    print("==============================================================")
    print("RUNNING GC-MS SPECIFIC ALIGNMENT EXPERIMENT")
    print("==============================================================")
    
    # 1. Generate Realistic GC-MS Dataset
    # 15 samples, 100 time scans, 80 m/z channels, 3 resolved components
    print("Generating simulated GC-MS data (EMG profiles & sparse spectra)...")
    generator = GCMSDataGenerator(num_samples=15, num_time=100, num_spec=80, num_components=3, seed=42)
    dataset = generator.generate_dataset(noise_std=0.01, max_shift=0.06, max_stretch=0.08, warp_type='linear')
    
    # Verify sparsity
    sparsity = np.mean(dataset['X'] == 0.0) * 100.0
    print(f"Data Matrix shape: {dataset['X'].shape} | Sparsity (exact zeros): {sparsity:.2f}%")
    
    # 2. Train GCMS_PETN Model
    # We train in coordinate-based mode with a batch size and a threshold to evaluate the sparse COO optimization
    print("\nTraining GCMS_PETN in sparse coordinate-based mode (threshold = 0.005)...")
    model = train_chroma_petn(
        dataset=dataset,
        epochs=1200,
        lr=0.015,
        warp_reg_coef=0.001,
        warp_type='linear',
        num_components=3,
        batch_size=None,
        threshold=None,
        compile_model=True,
        patience=150,
        lambda_res=20.0
    )
    
    # 3. Evaluate & Detach Loadings
    print("\nEvaluating alignment and component profile recovery...")
    metrics = evaluate_chroma_alignment(model, dataset, save_dir='notebooks/chroma/experiments/gcms')
    
    # 4. Generate Diagnostic Verification Plot
    plot_path = 'notebooks/chroma/experiments/gcms/gcms_alignment_verification.png'
    plot_alignment_verification(model, dataset['X'], save_path=plot_path)
    print("Experiment completed successfully.")

if __name__ == '__main__':
    run_gcms_experiment()
