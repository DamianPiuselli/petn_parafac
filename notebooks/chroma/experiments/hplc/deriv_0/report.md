# HPLC-PETN Model Calibration & Recovery Report

## 1. Summary of Recovered Component Loadings
Below are the cosine similarities (correlation coefficients) between the ground truth and PETN-resolved profiles for each of the 3 chemical components.

| Component | Component Label | Concentration Score (A) | Chromatography Profile (B) | Spectral Profile (C) |
|---|---|---|---|---|
| **Component 1** | Component 1 (Baseline Interference) | 0.995379 | 0.793173 | 0.991801 |
| **Component 2** | Component 2 (Peak 1) | 0.999389 | 0.985498 | 0.967799 |
| **Component 3** | Component 3 (Peak 2) | 0.999289 | 0.990700 | 0.997270 |

### Key Averages:
- **Mean Score Similarity:** 0.998019
- **Mean Elution Profile Similarity:** 0.923124
- **Mean Spectral Profile Similarity:** 0.985623

## 2. Retention Time Warping & Alignment Performance
The warping head aligns sample-specific shifting and stretching to the canonical time grid.

- **Mean Coordinate Alignment Error (MAE):** 2.388689e-03 (normalized time units)
- **Shift Parameter (beta) Correlation:** 0.993021
- **Stretch Parameter (alpha) Correlation:** 0.986261
- **Mean Absolute Shift Parameter Error:** 2.752259e-03

## 3. Visualization Artifacts
The following plots have been generated and saved to the experiment folder:
1. **[Resolved Profiles](chroma_resolved_profiles.png)**: Overlays true vs. recovered elution profiles (B) and absorption/mass spectra (C).
2. **[Alignment Comparison](chroma_alignment_comparison.png)**: Shows total elution profiles (TIC) across all samples before and after alignment.
3. **[Scores Comparison](scores_comparison.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
4. **[Warp Parameter Recovery](chroma_warp_parameters.png)**: True vs. predicted shift and stretch parameters for each sample.
