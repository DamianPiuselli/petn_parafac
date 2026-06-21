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
| **Spectral Similarity Penalty ($\lambda_{\text{sim}}$)** | 0.0 |
| **Baseline L2 Penalty ($\lambda_{\text{base}}$)** | 0.0 |
| **Convergence Epoch** | 1200 |
| **Final Model Loss (Derivative MSE)** | 7.32611e-01 |
| **Reconstructed Fit R² (Variance Explained)** | **99.47%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.32 min | 318.0 nm | 1072.2 | 13098.1 | 0.08x |
| **Component 2** | 11.82 min | 318.0 nm | 4769.9 | 314.1 | 15.19x |
| **Component 3** | 12.20 min | 318.0 nm | 9896.8 | 1744.9 | 5.67x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **15.19x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1     | Component_2        | Component_3      |
| -----|-----------------|--------------------|----------------- |
| 119  | 0.0             | 7331.294921875     | 9588.33203125    |
| 122  | 2144.3662109375 | 2208.45751953125   | 10205.3271484375 |
| 121  | 7707.572265625  | 158.0887908935547  | 2048.220703125   |
| 458  | 18488.65234375  | 470.07391357421875 | 1441.66748046875 |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1              | beta_C1               | alpha_C2             | beta_C2              | alpha_C3              | beta_C3                |
| -----|-----|-----------------------|-----------------------|----------------------|----------------------|-----------------------|----------------------- |
| 119  | +   | 0.039826203137636185  | -0.022263584658503532 | 0.0287935808300972   | 0.0726301521062851   | 0.025341201573610306  | -0.06531955301761627   |
| 122  | +   | -0.015976719558238983 | -0.008739819750189781 | 0.01376598421484232  | -0.10244421660900116 | 0.001783179584890604  | -0.0075487555004656315 |
| 121  | -   | -0.013863115571439266 | 0.02367176115512848   | -0.05263539031147957 | -0.08834846317768097 | 0.011071893386542797  | 0.016745295375585556   |
| 458  | -   | -0.00998637080192566  | 0.007331644184887409  | 0.01007582526654005  | 0.11816252768039703  | -0.038196273148059845 | 0.05612301081418991    |

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
