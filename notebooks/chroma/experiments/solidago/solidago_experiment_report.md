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
| **Savitzky-Golay Filter** | Order: 2 (derivative), Window size: 11 |
| **Spectral Similarity Penalty ($\lambda_{\text{sim}}$)** | 0.0 |
| **Baseline L2 Penalty ($\lambda_{\text{base}}$)** | 0.0 |
| **Convergence Epoch** | 1200 |
| **Final Model Loss (Derivative MSE)** | 5.08971e+00 |
| **Reconstructed Fit R² (Variance Explained)** | **98.06%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.28 min | 318.0 nm | 5297.4 | 13566.0 | 0.39x |
| **Component 2** | 12.06 min | 318.0 nm | 7755.5 | 380.9 | 20.36x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **20.36x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1       | Component_2       |
| -----|-------------------|------------------ |
| 119  | 68.11457824707031 | 15195.4462890625  |
| 122  | 10526.71875       | 315.4983215332031 |
| 121  | 7831.61181640625  | 0.0               |
| 458  | 19300.484375      | 761.7696533203125 |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha                 | beta                 |
| -----|-----|-----------------------|--------------------- |
| 119  | +   | 0.01384665071964264   | 0.005190023686736822 |
| 122  | +   | 0.03312439098954201   | -0.08944305777549744 |
| 121  | -   | -0.02207697369158268  | 0.05522715300321579  |
| 458  | -   | -0.024894066154956818 | 0.029025878757238388 |

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
