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
| **Spectral Similarity Penalty ($\lambda_{\text{sim}}$)** | 100.0 |
| **Baseline L2 Penalty ($\lambda_{\text{base}}$)** | 0.5 |
| **Convergence Epoch** | 1191 |
| **Final Model Loss (Derivative MSE)** | 2.76750e+01 |
| **Reconstructed Fit R² (Variance Explained)** | **73.95%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.30 min | 318.0 nm | 6889.2 | 13605.6 | 0.51x |
| **Component 2** | 12.06 min | 202.0 nm | 2906.5 | 1617.2 | 1.80x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 2** is upregulated by **1.80x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1     | Component_2        |
| -----|-----------------|------------------- |
| 119  | 3248.53515625   | 4149.5859375       |
| 122  | 10529.9375      | 1663.4923095703125 |
| 121  | 7742.1328125    | 1094.3253173828125 |
| 458  | 19469.017578125 | 2140.153564453125  |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1             | beta_C1               | alpha_C2             | beta_C2              |
| -----|-----|----------------------|-----------------------|----------------------|--------------------- |
| 119  | +   | 0.035576917231082916 | -0.01883494108915329  | -0.03428228199481964 | 0.029285266995429993 |
| 122  | +   | 0.01552771870046854  | -0.042663391679525375 | 0.11165042966604233  | -0.15000000596046448 |
| 121  | -   | -0.02819964662194252 | 0.03825261816382408   | -0.07871253043413162 | 0.15000000596046448  |
| 458  | -   | -0.02290498837828636 | 0.023245714604854584  | 0.001344385789707303 | -0.03179718554019928 |

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
