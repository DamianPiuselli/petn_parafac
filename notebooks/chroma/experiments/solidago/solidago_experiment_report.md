# Chroma-PETN Solidago Root Extracts HPLC-DAD Experiment Report

## 1. Executive Summary
This report provides a formal evaluation of the Gray-Box Physics-Embedded Tensor Network (Chroma-PETN) applied to real-world chromatographic data: *Solidago altissima* root extracts (HPLC-DAD). The network successfully aligns retention-time shifted peaks and decomposes overlapping bands within the localized time window of **11.80 to 13.00 minutes** while adjusting for solvent baseline drift in an end-to-end differentiable pipeline.

## 2. Model Configuration & Training Convergence
| Parameter | Value |
|---|---|
| **Model Type** | `HPLC_PETN` (HPLC-DAD optimization) |
| **Sliced Time Window** | **11.80 to 13.00 minutes** |
| **Resolved Components (R)** | 3 |
| **Warping Mode** | `linear` |
| **Savitzky-Golay Filter** | Order: 2 (derivative), Window size: 11 |
| **Spectral Similarity Penalty ($\lambda_{\text{sim}}$)** | 0.0 |
| **Baseline L2 Penalty ($\lambda_{\text{base}}$)** | 0.0 |
| **Convergence Epoch** | 959 |
| **Final Model Loss (Derivative MSE)** | 6.37346e-01 |
| **Reconstructed Fit R² (Variance Explained)** | **99.74%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.20 min | 318.0 nm | 10308.1 | 11963.8 | 0.86x |
| **Component 2** | 12.70 min | 202.0 nm | 3755.8 | 1297.8 | 2.89x |
| **Component 3** | 12.68 min | 318.0 nm | 2619.8 | 1109.0 | 2.36x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **2.89x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1      | Component_2       | Component_3       |
| -----|------------------|-------------------|------------------ |
| 119  | 11935.5478515625 | 4176.19921875     | 2389.484619140625 |
| 122  | 8680.6572265625  | 3335.441162109375 | 2850.0703125      |
| 121  | 7616.60498046875 | 0.0               | 2217.940673828125 |
| 458  | 16310.9912109375 | 2595.628662109375 | 0.0               |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha                 | beta                  |
| -----|-----|-----------------------|---------------------- |
| 119  | +   | 0.06361255794763565   | -0.02412520907819271  |
| 122  | +   | 0.03903502970933914   | -0.014381475746631622 |
| 121  | -   | -0.023809339851140976 | 0.013465140014886856  |
| 458  | -   | -0.07883824408054352  | 0.025041544809937477  |

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
