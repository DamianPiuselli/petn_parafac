# GCMS-PETN Model Calibration & Recovery Report

## 1. Summary of Recovered Component Loadings
Below are the cosine similarities (correlation coefficients) between the ground truth and PETN-resolved profiles for each of the 3 chemical components.

| Component | Component Label | Concentration Score (A) | Chromatography Profile (B) | Spectral Profile (C) |
|---|---|---|---|---|
| **Component 1** | Analyte Component 1 | 0.999372 | 0.996719 | 0.997852 |
| **Component 2** | Analyte Component 2 | 0.999360 | 0.998240 | 0.999278 |
| **Component 3** | Analyte Component 3 | 0.999319 | 0.998937 | 0.999269 |

### Key Averages:
- **Mean Score Similarity:** 0.999350
- **Mean Elution Profile Similarity:** 0.997965
- **Mean Spectral Profile Similarity:** 0.998800

## 2. Retention Time Warping & Alignment Performance
The warping head aligns sample-specific shifting and stretching to the canonical time grid.

- **Mean Coordinate Alignment Error (MAE):** 1.851584e-03 (normalized time units)
- **Shift Parameter (beta) Correlation:** 0.992246
- **Stretch Parameter (alpha) Correlation:** 0.982999
- **Mean Absolute Shift Parameter Error:** 3.426767e-03

## 3. Visualization Artifacts
The following plots have been generated and saved to the experiment folder:
1. **[Resolved Profiles](chroma_resolved_profiles.png)**: Overlays true vs. recovered elution profiles (B) and absorption/mass spectra (C).
2. **[Alignment Comparison](chroma_alignment_comparison.png)**: Shows total elution profiles (TIC) across all samples before and after alignment.
3. **[Scores Comparison](scores_comparison.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
4. **[Warp Parameter Recovery](chroma_warp_parameters.png)**: True vs. predicted shift and stretch parameters for each sample.
