# EEM-PETN Model Calibration & Recovery Report
 
 ## 1. Summary of Recovered Component Loadings & Absorptivities
 Below are the recovery metrics (R² scores and Cosine Similarities) between the ground truth and EEM-PETN resolved profiles for each of the 3 chemical components.
 
 | Component | Fluorophore Label | Score (A) R² | Score (A) CosSim | Excitation (B) R² | Excitation (B) CosSim | Emission (C) R² | Emission (C) CosSim | Excitation Abs (E) R² | Emission Abs (M) R² |
 |---|---|---|---|---|---|---|---|---|---|
| **Component 1** | Fluorophore Component 1 | 0.986404 | 0.996572 | 0.920324 | 0.983620 | 0.977450 | 0.993409 | 0.929644 | 0.000000 |
| **Component 2** | Fluorophore Component 2 | 0.988449 | 0.995760 | 0.979635 | 0.994194 | 0.918563 | 0.967565 | 0.679359 | 0.000000 |
| **Component 3** | Fluorophore Component 3 | 0.997019 | 0.998888 | 0.987157 | 0.993566 | 0.951897 | 0.981976 | 0.693346 | 0.000000 |

 ### Key Averages:
 - **Average Concentration Score R² (A):** 0.990624
 - **Average Excitation Loading R² (B):** 0.962372
 - **Average Emission Loading R² (C):** 0.949304
 - **Average Excitation Absorptivity R² (E):** 0.767450
 - **Average Emission Absorptivity R² (M):** 0.000000
 
 ## 2. Visualization Artifacts
 The following plots have been generated and saved to the EEM output folder:
 1. **[Resolved Profiles](simulated_resolved_profiles.png)**: Overlays true vs. recovered excitation (B) and emission (C) profiles.
 2. **[Resolved Absorptivities](simulated_resolved_absorptivities.png)**: Overlays true vs. recovered excitation (E) and emission (M) molar absorptivity curves.
 3. **[EEM Heatmaps](simulated_eem_heatmaps.png)**: Visualizes the true EEM, corrupted EEM, scatter mask, and reconstructed EEM.
 4. **[Scores Comparison](simulated_scores.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
 