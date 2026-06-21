# Chroma-PETN Solidago Root Extracts HPLC-DAD Experiment Report

## 1. Executive Summary
This report provides a formal evaluation of the Gray-Box Physics-Embedded Tensor Network (Chroma-PETN) applied to real-world chromatographic data: *Solidago altissima* root extracts (HPLC-DAD). The network successfully aligns retention-time shifted peaks and decomposes overlapping bands within the localized time window of **10.00 to 13.00 minutes** while adjusting for solvent baseline drift in an end-to-end differentiable pipeline.

## 2. Model Configuration & Training Convergence
| Parameter | Value |
|---|---|
| **Model Type** | `HPLC_PETN` (HPLC-DAD optimization) |
| **Sliced Time Window** | **10.00 to 13.00 minutes** |
| **Resolved Components (R)** | 2 |
| **Warping Mode** | `linear` |
| **Savitzky-Golay Filter** | Order: 2 (derivative), Window size: 11 |
| **Spectral Similarity Penalty ($\lambda_{\text{sim}}$)** | 100.0 |
| **Baseline L2 Penalty ($\lambda_{\text{base}}$)** | 0.5 |
| **Convergence Epoch** | 1200 |
| **Final Model Loss (Derivative MSE)** | 2.82453e+01 |
| **Reconstructed Fit R² (Variance Explained)** | **74.71%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.30 min | 318.0 nm | 6124.3 | 13105.5 | 0.47x |
| **Component 2** | 12.36 min | 202.0 nm | 4623.4 | 2472.7 | 1.87x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **1.87x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1       | Component_2       |
| -----|-------------------|------------------ |
| 119  | 2153.857666015625 | 6504.75390625     |
| 122  | 10094.818359375   | 2741.958251953125 |
| 121  | 7374.1083984375   | 1449.069091796875 |
| 458  | 18836.830078125   | 3496.29052734375  |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1             | beta_C1                | alpha_C2             | beta_C2              |
| -----|-----|----------------------|------------------------|----------------------|--------------------- |
| 119  | +   | -0.04332786053419113 | 0.05517686903476715    | 0.20000000298023224  | -0.08424852788448334 |
| 122  | +   | 0.03953426331281662  | -0.06591300666332245   | 0.011620118282735348 | -0.06468351185321808 |
| 121  | -   | 0.000189097598195076 | 0.011980623006820679   | -0.09251948446035385 | 0.06913483142852783  |
| 458  | -   | 0.003604498226195574 | -0.0012444807216525078 | -0.11962655186653137 | 0.07979719340801239  |

## 5. Visualizations
Below are the diagnostic figures illustrating the model alignment and resolved components:

### A. Resolved Loadings separated by Component
Shows resolved chromatography profiles (B) and absorbance spectra (C) on a component-by-component basis.

![Resolved Profiles](solidago_resolved_profiles.png)

### B. Dedicated Scores Comparison
Shows resolved concentration levels (scores) color-coded by sample vial and herbivore exclusion treatment.

![Sample Scores](solidago_scores.png)

### C. Alignment Comparison (Unaligned vs. Aligned TICs)
Left panel displays unaligned Total Ion Chromatograms (observed), and the right shows aligned chromatograms with warp adjustments applied.

![Unaligned vs. Aligned](solidago_alignment_comparison.png)

### D. Reconstruction & Fitting Overlay
Top panel displays observed vs reconstructed intensities at the maximum absorbance channel. Bottom panel displays observed vs reconstructed Total Ion Chromatograms (TICs).

![Original vs Reconstructed](solidago_alignment_verification.png)
