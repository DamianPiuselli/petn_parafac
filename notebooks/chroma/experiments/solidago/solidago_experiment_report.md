# Chroma-PETN Solidago Root Extracts HPLC-DAD Experiment Report

## 1. Executive Summary
This report provides a formal evaluation of the Gray-Box Physics-Embedded Tensor Network (Chroma-PETN) applied to real-world chromatographic data: *Solidago altissima* root extracts (HPLC-DAD). The network successfully aligns retention-time shifted peaks and decomposes overlapping bands within the localized time window of **10.00 to 13.00 minutes** while adjusting for solvent baseline drift in an end-to-end differentiable pipeline.

## 2. Model Configuration & Training Convergence
| Parameter | Value |
|---|---|
| **Model Type** | `HPLC_PETN` (HPLC-DAD optimization) |
| **Sliced Time Window** | **10.00 to 13.00 minutes** |
| **Resolved Components (R)** | 4 |
| **Warping Mode** | `linear` |
| **Savitzky-Golay Filter** | Order: 2 (derivative), Window size: 11 |
| **Spectral Similarity Penalty ($\lambda_{\text{sim}}$)** | 0.0 |
| **Baseline L2 Penalty ($\lambda_{\text{base}}$)** | 0.0 |
| **Convergence Epoch** | 1200 |
| **Final Model Loss (Derivative MSE)** | 3.27710e+00 |
| **Reconstructed Fit R² (Variance Explained)** | **92.13%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.30 min | 318.0 nm | 3331.2 | 11377.7 | 0.29x |
| **Component 2** | 12.08 min | 202.0 nm | 3004.2 | 581.1 | 5.17x |
| **Component 3** | 12.22 min | 318.0 nm | 6966.9 | 1664.6 | 4.19x |
| **Component 4** | 12.38 min | 318.0 nm | 2900.4 | 12797.5 | 0.23x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **5.17x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1      | Component_2       | Component_3       | Component_4       |
| -----|------------------|-------------------|-------------------|------------------ |
| 119  | 3530.91943359375 | 4805.12451171875  | 5582.02587890625  | 5541.93896484375  |
| 122  | 3131.50390625    | 1203.25537109375  | 8351.6806640625   | 258.84033203125   |
| 121  | 4301.951171875   | 898.2981567382812 | 2794.25927734375  | 23445.72265625    |
| 458  | 18453.537109375  | 263.8547058105469 | 534.9168701171875 | 2149.249755859375 |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1             | beta_C1               | alpha_C2              | beta_C2              | alpha_C3             | beta_C3               | alpha_C4             | beta_C4             |
| -----|-----|----------------------|-----------------------|-----------------------|----------------------|----------------------|-----------------------|----------------------|-------------------- |
| 119  | +   | -0.0392216257750988  | -0.058972880244255066 | -0.021135631948709488 | -0.08022560179233551 | 0.03970443457365036  | -0.06570105999708176  | 0.036096956580877304 | -0.1334909349679947 |
| 122  | +   | -0.0094557860866189  | -0.010259542614221573 | 0.1300470381975174    | 0.06692735850811005  | 0.004347092472016811 | -0.006781463511288166 | -0.10206829011440277 | -0.0979170873761177 |
| 121  | -   | 0.09489516913890839  | 0.034501079469919205  | 0.09640860557556152   | 0.07809153199195862  | -0.08657971769571304 | 0.10341668874025345   | 0.10984420776367188  | 0.1494663506746292  |
| 458  | -   | -0.04621776193380356 | 0.034731339663267136  | -0.20000000298023224  | -0.06479328870773315 | 0.0425281897187233   | -0.03093416802585125  | -0.04387287050485611 | 0.08194166421890259 |

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
