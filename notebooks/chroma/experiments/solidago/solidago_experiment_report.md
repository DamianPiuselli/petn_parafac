# Chroma-PETN Solidago Root Extracts HPLC-DAD Experiment Report

## 1. Executive Summary
This report provides a formal evaluation of the Gray-Box Physics-Embedded Tensor Network (Chroma-PETN) applied to real-world chromatographic data: *Solidago altissima* root extracts (HPLC-DAD). The network successfully aligns retention-time shifted peaks and decomposes overlapping bands within the localized time window of **10.00 to 13.00 minutes** while adjusting for solvent baseline drift in an end-to-end differentiable pipeline.

## 2. Model Configuration & Training Convergence
| Parameter | Value |
|---|---|
| **Model Type** | `HPLC_PETN` (HPLC-DAD optimization) |
| **Sliced Time Window** | **10.00 to 13.00 minutes** |
| **Resolved Components (R)** | 3 |
| **Warping Mode** | `linear` |
| **Savitzky-Golay Filter** | Order: 2 (derivative), Window size: 11 |
| **Spectral Similarity Penalty ($\lambda_{\text{sim}}$)** | 100.0 |
| **Baseline L2 Penalty ($\lambda_{\text{base}}$)** | 0.5 |
| **Convergence Epoch** | 1200 |
| **Final Model Loss (Derivative MSE)** | 2.38666e+01 |
| **Reconstructed Fit R² (Variance Explained)** | **77.49%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.30 min | 318.0 nm | 5858.7 | 12566.1 | 0.47x |
| **Component 2** | 12.62 min | 202.0 nm | 7141.9 | 2886.6 | 2.47x |
| **Component 3** | 12.22 min | 280.0 nm | 3410.2 | 2863.3 | 1.19x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **2.47x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1        | Component_2        | Component_3        |
| -----|--------------------|--------------------|------------------- |
| 119  | 2027.1739501953125 | 11970.587890625    | 4166.87744140625   |
| 122  | 9690.19921875      | 2313.130615234375  | 2653.469970703125  |
| 121  | 7105.1142578125    | 1634.4920654296875 | 1735.0670166015625 |
| 458  | 18027.146484375    | 4138.75146484375   | 3991.600830078125  |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1             | beta_C1              | alpha_C2              | beta_C2              | alpha_C3              | beta_C3               |
| -----|-----|----------------------|----------------------|-----------------------|----------------------|-----------------------|---------------------- |
| 119  | +   | -0.05895795673131943 | 0.07046646624803543  | 0.05607489496469498   | 0.12159509211778641  | 0.06233297660946846   | -0.0979016050696373   |
| 122  | +   | 0.04052160307765007  | -0.06780527532100677 | -0.025273513048887253 | 0.052288543432950974 | 0.026801928877830505  | -0.033404912799596786 |
| 121  | -   | 0.006357681937515736 | 0.006243126932531595 | -0.009699216112494469 | -0.08856470137834549 | -0.042214419692754745 | 0.06987497955560684   |
| 458  | -   | 0.012078669853508472 | -0.00890431459993124 | -0.021102169528603554 | -0.08531893044710159 | -0.04692048951983452  | 0.06143154203891754   |

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
