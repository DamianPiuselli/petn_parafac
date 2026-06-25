# EEM-PETN Model Calibration & Recovery Report

## 1. Summary of Recovered Component Loadings & Absorptivities
Below are the recovery metrics (R² scores and Cosine Similarities) between the ground truth and EEM-PETN resolved profiles for each of the 3 chemical components.

| Component | Fluorophore Label | Score (A) R² | Score (A) CosSim | Excitation (B) R² | Excitation (B) CosSim | Emission (C) R² | Emission (C) CosSim | Excitation Abs (E) R² | Emission Abs (M) R² |
|---|---|---|---|---|---|---|---|---|---|
| **Component 1** | Fluorophore Component 1 | 0.960071 | 0.992668 | 0.860130 | 0.968806 | 0.972330 | 0.988968 | 0.505576 | 0.000000 |
| **Component 2** | Fluorophore Component 2 | 0.989059 | 0.995377 | 0.978672 | 0.992477 | 0.914898 | 0.963286 | 0.944215 | 0.000000 |
| **Component 3** | Fluorophore Component 3 | 0.998279 | 0.999395 | 0.986652 | 0.993462 | 0.948757 | 0.979783 | 0.886305 | 0.000000 |

### Key Averages:
- **Average Concentration Score R² (A):** 0.982470
- **Average Excitation Loading R² (B):** 0.941818
- **Average Emission Loading R² (C):** 0.945328
- **Average Excitation Absorptivity R² (E):** 0.778699
- **Average Emission Absorptivity R² (M):** 0.000000

## 2. Visualization Artifacts
The following plots have been generated and saved to the EEM output folder:
1. **[Resolved Profiles](phase3_resolved_profiles.png)**: Overlays true vs. recovered excitation (B) and emission (C) profiles.
2. **[Resolved Absorptivities](phase3_resolved_absorptivities.png)**: Overlays true vs. recovered excitation (E) and emission (M) molar absorptivity curves.
3. **[EEM Heatmaps](phase3_eem_heatmaps.png)**: Visualizes the true EEM, corrupted EEM, scatter mask, and reconstructed EEM.
4. **[Scores Comparison](scores_comparison.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
