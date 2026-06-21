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
| **Convergence Epoch** | 869 |
| **Final Model Loss (Derivative MSE)** | 3.46237e-01 |
| **Reconstructed Fit R² (Variance Explained)** | **99.84%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.20 min | 318.0 nm | 9956.0 | 12434.0 | 0.80x |
| **Component 2** | 12.74 min | 202.0 nm | 1773.7 | 1606.0 | 1.10x |
| **Component 3** | 12.70 min | 318.0 nm | 2776.7 | 1862.8 | 1.49x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 3** is upregulated by **1.49x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1     | Component_2       | Component_3       |
| -----|-----------------|-------------------|------------------ |
| 119  | 11346.69921875  | 2210.55322265625  | 2499.327392578125 |
| 122  | 8565.25         | 1336.84521484375  | 3054.122802734375 |
| 121  | 7694.8369140625 | 305.5028076171875 | 2210.772216796875 |
| 458  | 17173.216796875 | 2906.438232421875 | 1514.780029296875 |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1              | beta_C1               | alpha_C2              | beta_C2              | alpha_C3             | beta_C3              |
| -----|-----|-----------------------|-----------------------|-----------------------|----------------------|----------------------|--------------------- |
| 119  | +   | 0.02872462011873722   | -0.01216893084347248  | -0.008007034659385681 | 0.01816350221633911  | 0.011437300592660904 | 0.02709953486919403  |
| 122  | +   | 0.010043861344456673  | -0.004107074812054634 | -0.025274941697716713 | 0.006324641406536102 | 0.010713556781411171 | 0.013206328265368938 |
| 121  | -   | 0.0008009125012904406 | 0.0038333036936819553 | -0.1662968546152115   | -0.0722469687461853  | -0.05709632858633995 | 0.027415134012699127 |
| 458  | -   | -0.039569392800331116 | 0.012442702427506447  | 0.19957883656024933   | 0.04775882139801979  | 0.034945469349622726 | -0.06772099435329437 |

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
