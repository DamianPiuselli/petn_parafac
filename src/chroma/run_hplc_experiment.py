"""
HPLC-DAD Submodel Experiment Runner.
Simulates realistic HPLC-DAD data with symmetric peaks, continuous spectra, and solvent drift.
Trains HPLC_PETN (with baseline offset parameters & Savitzky-Golay filters) and evaluates alignment.
"""
import os
import torch
import numpy as np

from src.chroma import HPLC_PETN, HPLCDADDataGenerator, plot_alignment_verification
from src.chroma.train import train_chroma_petn, evaluate_chroma_alignment

def run_hplc_experiment():
    print("==============================================================")
    print("RUNNING HPLC-DAD ALIGNMENT EXPERIMENTS ON MULTIPLE DERIVATIVE ORDERS")
    print("==============================================================")
    
    # 1. Generate Realistic HPLC-DAD Dataset
    # 15 samples, 100 time scans, 80 wavelength channels, 3 components
    print("Generating simulated HPLC-DAD data (continuous bands & solvent drift)...")
    generator = HPLCDADDataGenerator(num_samples=15, num_time=100, num_spec=80, num_components=3, seed=42)
    dataset = generator.generate_dataset(noise_std=0.015, max_shift=0.05, max_stretch=0.08, warp_type='linear')
    
    print(f"Data Matrix shape: {dataset['X'].shape} | Simulated solvent baseline background added.\n")
    
    derivative_orders = [0, 1, 2]
    results_summary = {}
    
    for deriv_order in derivative_orders:
        print(f"\n--------------------------------------------------------------")
        print(f"TRAINING HPLC_PETN WITH DERIVATIVE ORDER {deriv_order}")
        print(f"--------------------------------------------------------------")
        
        save_dir = f'notebooks/chroma/experiments/hplc/deriv_{deriv_order}'
        os.makedirs(save_dir, exist_ok=True)
        
        # Train HPLC_PETN Model
        # We train in dense grid-based mode since HPLC data is continuous.
        model = train_chroma_petn(
            dataset=dataset,
            epochs=1500,
            lr=0.012,
            warp_reg_coef=0.001,
            warp_type='linear',
            num_components=3,
            derivative_order=deriv_order,
            sg_window_size=11,
            sg_polyorder=2,
            batch_size=None,  # Full-grid training
            compile_model=True,
            patience=150,
            lambda_raw=1.0,
            lambda_smooth_B=0.01,
            model_type='hplc'
        )
        
        # Evaluate & Detach Loadings
        print(f"\nEvaluating alignment and component profile recovery for order {deriv_order}...")
        metrics = evaluate_chroma_alignment(model, dataset, save_dir=save_dir)
        results_summary[deriv_order] = metrics
        
        # Generate Diagnostic Verification Plot
        plot_path = os.path.join(save_dir, 'hplc_alignment_verification.png')
        plot_alignment_verification(model, dataset['X'], save_path=plot_path)
        
    # Print comparison summary table
    print("\n" + "="*80)
    print("                      DERIVATIVE ORDER COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Order':<6} | {'Mean B Sim':<11} | {'Mean C Sim':<11} | {'Mean A Sim':<11} | {'Shift Corr':<10} | {'Stretch Corr':<12}")
    print("-"*80)
    for order in derivative_orders:
        m = results_summary[order]
        mean_b = np.mean(m['b_similarities'])
        mean_c = np.mean(m['c_similarities'])
        mean_a = np.mean(m['a_similarities'])
        shift_c = m['shift_correlation']
        stretch_c = m['stretch_correlation']
        print(f"{order:<6d} | {mean_b:<11.4f} | {mean_c:<11.4f} | {mean_a:<11.4f} | {shift_c:<10.4f} | {stretch_c:<12.4f}")
    print("="*80)
    print("All experiments completed successfully.")

if __name__ == '__main__':
    run_hplc_experiment()
