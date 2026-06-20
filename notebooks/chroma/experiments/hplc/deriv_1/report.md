# HPLC-PETN Model Calibration & Recovery Report

## 1. Summary of Recovered Component Loadings
Below are the cosine similarities (correlation coefficients) between the ground truth and PETN-resolved profiles for each of the 3 chemical components.

| Component | Component Label | Concentration Score (A) | Chromatography Profile (B) | Spectral Profile (C) |
|---|---|---|---|---|
| **Component 1** | Component 1 (Baseline Interference) | 0.999405 | 0.990434 | 0.999964 |
| **Component 2** | Component 2 (Peak 1) | 0.999486 | 0.990628 | 0.999964 |
| **Component 3** | Component 3 (Peak 2) | 0.999336 | 0.990570 | 0.999954 |

### Key Averages:
- **Mean Score Similarity:** 0.999409
- **Mean Elution Profile Similarity:** 0.990544
- **Mean Spectral Profile Similarity:** 0.999960

## 2. Retention Time Warping & Alignment Performance
The warping head aligns sample-specific shifting and stretching to the canonical time grid.

- **Mean Coordinate Alignment Error (MAE):** 2.899749e-03 (normalized time units)
- **Shift Parameter (beta) Correlation:** 0.976483
- **Stretch Parameter (alpha) Correlation:** 0.984581
- **Mean Absolute Shift Parameter Error:** 4.942166e-03

## 3. Visualization Artifacts
The following plots have been generated and saved to the experiment folder:
1. **[Resolved Profiles](chroma_resolved_profiles.png)**: Overlays true vs. recovered elution profiles (B) and absorption/mass spectra (C).
2. **[Alignment Comparison](chroma_alignment_comparison.png)**: Shows total elution profiles (TIC) across all samples before and after alignment.
3. **[Scores Comparison](scores_comparison.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
4. **[Warp Parameter Recovery](chroma_warp_parameters.png)**: True vs. predicted shift and stretch parameters for each sample.
