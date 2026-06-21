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
| **Convergence Epoch** | 1200 |
| **Final Model Loss (Derivative MSE)** | 2.75430e+00 |
| **Reconstructed Fit R² (Variance Explained)** | **97.49%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.32 min | 318.0 nm | 6386.1 | 14751.3 | 0.43x |
| **Component 2** | 12.08 min | 318.0 nm | 7169.2 | 2436.7 | 2.94x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **2.94x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1        | Component_2        |
| -----|--------------------|------------------- |
| 119  | 1581.7493896484375 | 13272.203125       |
| 122  | 11190.4072265625   | 1066.2406005859375 |
| 121  | 8259.8427734375    | 838.5316772460938  |
| 458  | 21242.818359375    | 4034.90234375      |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1               | beta_C1               | alpha_C2             | beta_C2               |
| -----|-----|------------------------|-----------------------|----------------------|---------------------- |
| 119  | +   | -0.0029816380701959133 | 0.035731419920921326  | -0.04514811560511589 | 0.03011495992541313   |
| 122  | +   | 0.02573510631918907    | -0.0601646825671196   | -0.05145810544490814 | 0.004332950338721275  |
| 121  | -   | -0.013333314098417759  | 0.01934073306620121   | 0.06462518125772476  | -0.025199223309755325 |
| 458  | -   | -0.009420153684914112  | 0.0050925337709486485 | 0.03198104351758957  | -0.009248687885701656 |

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
