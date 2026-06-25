# GCMS-PETN Model Calibration & Recovery Report

## 1. Summary of Recovered Component Loadings
Below are the cosine similarities (correlation coefficients) between the ground truth and PETN-resolved profiles for each of the 3 chemical components.

| Component | Component Label | Concentration Score (A) | Chromatography Profile (B) | Spectral Profile (C) |
|---|---|---|---|---|
| **Component 1** | Analyte Component 1 | 0.995657 | 0.901352 | 0.993412 |
| **Component 2** | Analyte Component 2 | 0.908605 | 0.920070 | 0.998260 |
| **Component 3** | Analyte Component 3 | 0.462118 | 0.949088 | 0.998573 |

### Key Averages:
- **Mean Score Similarity:** 0.788793
- **Mean Elution Profile Similarity:** 0.923503
- **Mean Spectral Profile Similarity:** 0.996748

## 2. Retention Time Warping & Alignment Performance
The warping head aligns sample-specific shifting and stretching to the canonical time grid.

- **Mean Coordinate Alignment Error (MAE):** 5.716013e-03 (normalized time units)
- **Shift Parameter (beta) Correlation:** 0.514826
- **Stretch Parameter (alpha) Correlation:** 0.771743
- **Mean Absolute Shift Parameter Error:** 3.774609e-02

## 3. Visualization Artifacts
The following plots have been generated and saved to the experiment folder:
1. **[Resolved Profiles](chroma_resolved_profiles.png)**: Overlays true vs. recovered elution profiles (B) and absorption/mass spectra (C).
2. **[Alignment Comparison](chroma_alignment_comparison.png)**: Shows total elution profiles (TIC) across all samples before and after alignment.
3. **[Scores Comparison](scores_comparison.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
4. **[Warp Parameter Recovery](chroma_warp_parameters.png)**: True vs. predicted shift and stretch parameters for each sample.
