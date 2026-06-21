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
| **Convergence Epoch** | 150 |
| **Final Model Loss (Derivative MSE)** | 3.34106e+01 |
| **Reconstructed Fit R² (Variance Explained)** | **68.65%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.30 min | 316.0 nm | 3337.9 | 13502.1 | 0.25x |
| **Component 2** | 12.08 min | 202.0 nm | 3582.0 | 351.2 | 10.20x |
| **Component 3** | 12.20 min | 270.0 nm | 2401.4 | 418.1 | 5.74x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **10.20x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1       | Component_2        | Component_3        |
| -----|-------------------|--------------------|------------------- |
| 119  | 2086.004638671875 | 5757.25634765625   | 1306.6669921875    |
| 122  | 4589.74609375     | 1406.7127685546875 | 3496.044189453125  |
| 121  | 7552.29541015625  | 64.1406478881836   | 246.78616333007812 |
| 458  | 19451.951171875   | 638.2239990234375  | 589.4068603515625  |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1               | beta_C1              | alpha_C2              | beta_C2               | alpha_C3              | beta_C3               |
| -----|-----|------------------------|----------------------|-----------------------|-----------------------|-----------------------|---------------------- |
| 119  | +   | 0.007153583690524101   | 0.016370989382267    | -0.004036350175738335 | 0.0016243597492575645 | 0.05521106347441673   | -0.016928184777498245 |
| 122  | +   | 0.00445720087736845    | -0.03963332250714302 | 0.001709542702883482  | -0.04564140364527702  | 0.018816713243722916  | -0.021031705662608147 |
| 121  | -   | -0.0025381557643413544 | 0.014315897598862648 | 0.03567791357636452   | -0.051591165363788605 | -0.062496479600667953 | -0.003464963287115097 |
| 458  | -   | -0.009072628803551197  | 0.008946433663368225 | -0.0333511084318161   | 0.09560821205377579   | -0.011531294323503971 | 0.04142485186457634   |

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
