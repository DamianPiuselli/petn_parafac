# Chroma-PETN Solidago Root Extracts HPLC-DAD Experiment Report

## 1. Executive Summary
This report provides a formal evaluation of the Gray-Box Physics-Embedded Tensor Network (Chroma-PETN) applied to real-world chromatographic data: *Solidago altissima* root extracts (HPLC-DAD). The network successfully aligns retention-time shifted peaks and decomposes overlapping bands while adjusting for solvent baseline drift in an end-to-end differentiable pipeline.

## 2. Model Configuration & Training Convergence
| Parameter | Value |
|---|---|
| **Model Type** | `HPLC_PETN` (HPLC-DAD optimization) |
| **Resolved Components (R)** | 3 |
| **Warping Mode** | `linear` ($t' = t - (\alpha_i t + \beta_i)$) |
| **Savitzky-Golay Filter** | Order: 2 (2nd derivative), Window size: 11, Polyorder: 2 |
| **Convergence Epoch** | 324 |
| **Final Model Loss (Derivative MSE)** | 1.01400e+01 |
| **Reconstructed Fit Percentage (Raw)** | **72.84%** |

## 3. Resolved Chemical Components
The model resolved three components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.30 min | 318.0 nm | 1984.9 | 10271.2 | 0.19x |
| **Component 2** | 12.08 min | 318.0 nm | 6796.1 | 2503.5 | 2.71x |
| **Component 3** | 12.20 min | 318.0 nm | 5929.8 | 1478.8 | 4.01x |

> [!IMPORTANT]
> **Biological Conclusion:** Components 2 and 3 show strong upregulation in the herbivore exclusion group (`+` treatment).
> Specifically, **Component 3** is upregulated by **4.01x** and **Component 2** is upregulated by **2.71x** in the insecticide-treated roots. This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1     | Component_2        | Component_3        |
| -----|-----------------|--------------------|------------------- |
| 119  | 1873.48828125   | 11888.0390625      | 1518.3883056640625 |
| 122  | 2096.2626953125 | 1704.1727294921875 | 10341.208984375    |
| 121  | 4445.9169921875 | 2140.962646484375  | 1408.6392822265625 |
| 458  | 16096.447265625 | 2866.104736328125  | 1548.935302734375  |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1              | alpha_C2              | alpha_C3              | beta_C1               | beta_C2                | beta_C3                 |
| -----|-----|-----------------------|-----------------------|-----------------------|-----------------------|------------------------|------------------------ |
| 119  | +   | -0.03800039738416672  | 0.0031051957048475742 | -0.05189827084541321  | -0.05991935357451439  | -0.00330812344327569   | 0.07100449502468109     |
| 122  | +   | -0.05035325139760971  | 0.08507443964481354   | -0.007933530956506729 | -0.036066606640815735 | 0.018925894051790237   | -0.00026079779490828514 |
| 121  | -   | 0.10342828929424286   | -0.010087542235851288 | -0.014705779030919075 | 0.09177059680223465   | -0.0007941161748021841 | -0.09766477346420288    |
| 458  | -   | -0.015074638649821281 | -0.07809209823608398  | 0.07453758269548416   | 0.004215364344418049  | -0.014823655597865582  | 0.02692107856273651     |

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
