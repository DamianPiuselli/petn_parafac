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
| **Convergence Epoch** | 194 |
| **Final Model Loss (Derivative MSE)** | 9.80895e+00 |
| **Reconstructed Fit R² (Variance Explained)** | **77.55%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.32 min | 318.0 nm | 1922.0 | 11871.9 | 0.16x |
| **Component 2** | 12.06 min | 318.0 nm | 5553.8 | 597.4 | 9.30x |
| **Component 3** | 12.20 min | 318.0 nm | 4316.9 | 581.5 | 7.42x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **9.30x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1       | Component_2       | Component_3        |
| -----|-------------------|-------------------|------------------- |
| 119  | 962.7311401367188 | 9240.849609375    | 1712.0562744140625 |
| 122  | 2881.349365234375 | 1866.79248046875  | 6921.787109375     |
| 121  | 6413.43896484375  | 432.52783203125   | 487.6727294921875  |
| 458  | 17330.26171875    | 762.2970581054688 | 675.23193359375    |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1              | beta_C1              | alpha_C2               | beta_C2               | alpha_C3              | beta_C3               |
| -----|-----|-----------------------|----------------------|------------------------|-----------------------|-----------------------|---------------------- |
| 119  | +   | 0.030982624739408493  | 0.014964177273213863 | -0.0033654291182756424 | 0.004926004912704229  | 0.12807875871658325   | 0.0908837616443634    |
| 122  | +   | 0.003038837807253003  | -0.0370107963681221  | -0.023709384724497795  | -0.05776834115386009  | -0.005498044658452272 | 0.0011367707047611475 |
| 121  | -   | -0.01432684063911438  | 0.013637354597449303 | -0.013842221349477768  | -0.006618246901780367 | -0.05827710032463074  | -0.0628545880317688   |
| 458  | -   | -0.019694620743393898 | 0.008409267291426659 | 0.040917035192251205   | 0.059460584074258804  | -0.06430360674858093  | -0.02916594035923481  |

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
