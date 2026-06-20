# GCMS-PETN Model Calibration & Recovery Report

## 1. Summary of Recovered Component Loadings
Below are the cosine similarities (correlation coefficients) between the ground truth and PETN-resolved profiles for each of the 3 chemical components.

| Component | Component Label | Concentration Score (A) | Chromatography Profile (B) | Spectral Profile (C) |
|---|---|---|---|---|
| **Component 1** | Analyte Component 1 | 0.999295 | 0.997303 | 0.998297 |
| **Component 2** | Analyte Component 2 | 0.999229 | 0.998411 | 0.998886 |
| **Component 3** | Analyte Component 3 | 0.999322 | 0.999043 | 0.999166 |

### Key Averages:
- **Mean Score Similarity:** 0.999282
- **Mean Elution Profile Similarity:** 0.998252
- **Mean Spectral Profile Similarity:** 0.998783

## 2. Retention Time Warping & Alignment Performance
The warping head aligns sample-specific shifting and stretching to the canonical time grid.

- **Mean Coordinate Alignment Error (MAE):** 1.611729e-03 (normalized time units)
- **Shift Parameter (beta) Correlation:** 0.994263
- **Stretch Parameter (alpha) Correlation:** 0.987559
- **Mean Absolute Shift Parameter Error:** 2.990644e-03

## 3. Visualization Artifacts
The following plots have been generated and saved to the experiment folder:
1. **[Resolved Profiles](chroma_resolved_profiles.png)**: Overlays true vs. recovered elution profiles (B) and absorption/mass spectra (C).
2. **[Alignment Comparison](chroma_alignment_comparison.png)**: Shows total elution profiles (TIC) across all samples before and after alignment.
3. **[Scores Comparison](scores_comparison.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
4. **[Warp Parameter Recovery](chroma_warp_parameters.png)**: True vs. predicted shift and stretch parameters for each sample.
