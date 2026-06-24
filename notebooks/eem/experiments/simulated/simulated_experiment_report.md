# EEM-PETN Model Calibration & Recovery Report

## 1. Summary of Recovered Component Loadings & Absorptivities
Below are the recovery metrics (R² scores and Cosine Similarities) between the ground truth and EEM-PETN resolved profiles for each of the 3 chemical components.

| Component | Fluorophore Label | Score (A) R² | Score (A) CosSim | Excitation (B) R² | Excitation (B) CosSim | Emission (C) R² | Emission (C) CosSim | Excitation Abs (E) R² | Emission Abs (M) R² |
|---|---|---|---|---|---|---|---|---|---|
| **Component 1** | Fluorophore Component 1 | 0.995979 | 0.998205 | 0.943014 | 0.984729 | 0.974618 | 0.990251 | 0.896057 | 0.000000 |
| **Component 2** | Fluorophore Component 2 | 0.995083 | 0.997627 | 0.984705 | 0.994455 | 0.903652 | 0.958187 | 0.980507 | 0.000000 |
| **Component 3** | Fluorophore Component 3 | 0.995961 | 0.998812 | 0.993992 | 0.997250 | 0.946034 | 0.978919 | 0.989633 | 0.000000 |

### Key Averages:
- **Average Concentration Score R² (A):** 0.995675
- **Average Excitation Loading R² (B):** 0.973903
- **Average Emission Loading R² (C):** 0.941435
- **Average Excitation Absorptivity R² (E):** 0.955399
- **Average Emission Absorptivity R² (M):** 0.000000

## 2. Visualization Artifacts
The following plots have been generated and saved to the EEM output folder:
1. **[Resolved Profiles](phase3_resolved_profiles.png)**: Overlays true vs. recovered excitation (B) and emission (C) profiles.
2. **[Resolved Absorptivities](phase3_resolved_absorptivities.png)**: Overlays true vs. recovered excitation (E) and emission (M) molar absorptivity curves.
3. **[EEM Heatmaps](phase3_eem_heatmaps.png)**: Visualizes the true EEM, corrupted EEM, scatter mask, and reconstructed EEM.
4. **[Scores Comparison](scores_comparison.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
