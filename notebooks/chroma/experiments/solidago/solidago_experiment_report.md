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
| **Convergence Epoch** | 414 |
| **Final Model Loss (Derivative MSE)** | 5.59266e+00 |
| **Reconstructed Fit R² (Variance Explained)** | **97.56%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.32 min | 318.0 nm | 1842.6 | 13121.1 | 0.14x |
| **Component 2** | 12.06 min | 318.0 nm | 6643.4 | 497.5 | 13.35x |
| **Component 3** | 12.20 min | 318.0 nm | 4667.7 | 503.7 | 9.27x |
| **Component 4** | 12.38 min | 202.0 nm | 10.5 | 327.4 | 0.03x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **13.35x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1       | Component_2        | Component_3        | Component_4        |
| -----|-------------------|--------------------|--------------------|------------------- |
| 119  | 615.3922119140625 | 12088.2353515625   | 1430.9666748046875 | 18.627599716186523 |
| 122  | 3069.852783203125 | 1198.4752197265625 | 7904.44140625      | 2.4612178802490234 |
| 121  | 7571.75830078125  | 228.27447509765625 | 94.5107421875      | 435.3424377441406  |
| 458  | 18670.48046875    | 766.6322631835938  | 912.8704223632812  | 219.44827270507812 |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1              | beta_C1              | alpha_C2              | beta_C2              | alpha_C3              | beta_C3               | alpha_C4             | beta_C4              |
| -----|-----|-----------------------|----------------------|-----------------------|----------------------|-----------------------|-----------------------|----------------------|--------------------- |
| 119  | +   | 0.029933800920844078  | 0.013054651208221912 | -0.018611524254083633 | 0.011248299852013588 | 0.08006244152784348   | 0.06724712252616882   | -0.1761806309223175  | -0.14785003662109375 |
| 122  | +   | 0.0019983453676104546 | -0.04654946178197861 | 0.029844747856259346  | -0.09139850735664368 | -0.007433662191033363 | 0.001339346170425415  | 0.20000000298023224  | 0.13120928406715393  |
| 121  | -   | -0.014634557068347931 | 0.020343665033578873 | -0.06841541826725006  | 0.04073287919163704  | 0.015623587183654308  | -0.06120699644088745  | -0.07261103391647339 | -0.06560561060905457 |
| 458  | -   | -0.017297588288784027 | 0.013151144608855247 | 0.057182200253009796  | 0.0394173227250576   | -0.08825236558914185  | -0.007379472255706787 | 0.04843167960643768  | 0.08224637806415558  |

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
