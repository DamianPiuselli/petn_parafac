# EEM-PETN Model Calibration & Recovery Report
 
 ## 1. Summary of Recovered Component Loadings & Absorptivities
 Below are the recovery metrics (R² scores and Cosine Similarities) between the ground truth and EEM-PETN resolved profiles for each of the 3 chemical components.
 
 | Component | Fluorophore Label | Score (A) R² | Score (A) CosSim | Excitation (B) R² | Excitation (B) CosSim | Emission (C) R² | Emission (C) CosSim | Excitation Abs (E) R² | Emission Abs (M) R² |
 |---|---|---|---|---|---|---|---|---|---|
| **Component 1** | Fluorophore Component 1 | 0.995983 | 0.998204 | 0.943051 | 0.984736 | 0.974621 | 0.990254 | 0.895748 | 0.000000 |
| **Component 2** | Fluorophore Component 2 | 0.995087 | 0.997628 | 0.984711 | 0.994455 | 0.903609 | 0.958170 | 0.980399 | 0.000000 |
| **Component 3** | Fluorophore Component 3 | 0.995960 | 0.998811 | 0.993995 | 0.997252 | 0.946021 | 0.978914 | 0.989628 | 0.000000 |

 ### Key Averages:
 - **Average Concentration Score R² (A):** 0.995677
 - **Average Excitation Loading R² (B):** 0.973919
 - **Average Emission Loading R² (C):** 0.941417
 - **Average Excitation Absorptivity R² (E):** 0.955258
 - **Average Emission Absorptivity R² (M):** 0.000000
 
 ## 2. Visualization Artifacts
 The following plots have been generated and saved to the EEM output folder:
 1. **[Resolved Profiles](simulated_resolved_profiles.png)**: Overlays true vs. recovered excitation (B) and emission (C) profiles.
 2. **[Resolved Absorptivities](simulated_resolved_absorptivities.png)**: Overlays true vs. recovered excitation (E) and emission (M) molar absorptivity curves.
 3. **[EEM Heatmaps](simulated_eem_heatmaps.png)**: Visualizes the true EEM, corrupted EEM, scatter mask, and reconstructed EEM.
 4. **[Scores Comparison](simulated_scores.png)**: Parity plots of true vs. predicted concentrations (scores) with a 1:1 diagonal reference.
 