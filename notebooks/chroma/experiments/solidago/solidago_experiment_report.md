# Chroma-PETN Solidago Root Extracts HPLC-DAD Experiment Report

## 1. Executive Summary
This report provides a formal evaluation of the Gray-Box Physics-Embedded Tensor Network (Chroma-PETN) applied to real-world chromatographic data: *Solidago altissima* root extracts (HPLC-DAD). The network successfully aligns retention-time shifted peaks and decomposes overlapping bands within the localized time window of **11.80 to 13.00 minutes** while adjusting for solvent baseline drift in an end-to-end differentiable pipeline.

## 2. Model Configuration & Training Convergence
| Parameter | Value |
|---|---|
| **Model Type** | `HPLC_PETN` (HPLC-DAD optimization) |
| **Sliced Time Window** | **11.80 to 13.00 minutes** |
| **Resolved Components (R)** | 2 |
| **Warping Mode** | `linear` |
| **Savitzky-Golay Filter** | Order: 0 (derivative), Window size: 11 |
| **Spectral Similarity Penalty ($\lambda_{\text{sim}}$)** | 0.0 |
| **Baseline L2 Penalty ($\lambda_{\text{base}}$)** | 0.01 |
| **Convergence Epoch** | 1089 |
| **Final Model Loss (Derivative MSE)** | 1.21318e+02 |
| **Reconstructed Fit R² (Variance Explained)** | **99.64%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.24 min | 318.0 nm | 10209.0 | 12180.4 | 0.84x |
| **Component 2** | 12.74 min | 318.0 nm | 5570.3 | 3286.1 | 1.70x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **1.70x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1      | Component_2       |
| -----|------------------|------------------ |
| 119  | 12063.0849609375 | 5912.4765625      |
| 122  | 8354.9697265625  | 5228.2197265625   |
| 121  | 7409.44775390625 | 2403.807373046875 |
| 458  | 16951.255859375  | 4168.3408203125   |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha                | beta                 |
| -----|-----|----------------------|--------------------- |
| 119  | +   | 0.06530701369047165  | -0.15197472274303436 |
| 122  | +   | 0.02028028853237629  | -0.04003575071692467 |
| 121  | -   | -0.0387987419962883  | 0.1078573688864708   |
| 458  | -   | -0.04678855836391449 | 0.08415310829877853  |

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
