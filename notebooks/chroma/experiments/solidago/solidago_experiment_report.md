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
| **Convergence Epoch** | 1200 |
| **Final Model Loss (Derivative MSE)** | 1.36650e+00 |
| **Reconstructed Fit R² (Variance Explained)** | **99.47%** |

## 3. Resolved Chemical Components
The model resolved the localized components. Below are their characteristic physical properties:

| Component | RT apex ($t_{\max}$) | Spectral Maxima ($\lambda_{\max}$) | Mean Score ($+$) | Mean Score ($-$) | Ratio ($+/$-) |
|---|---|---|---|---|---|
| **Component 1** | 12.32 min | 318.0 nm | 1969.7 | 13662.7 | 0.14x |
| **Component 2** | 12.06 min | 318.0 nm | 8040.4 | 851.6 | 9.44x |
| **Component 3** | 12.20 min | 318.0 nm | 4485.4 | 235.1 | 19.08x |

> [!IMPORTANT]
> **Biological Conclusion:** In the localized peak window, the resolved components display distinct profiles. Specifically, **Component 3** is upregulated by **19.08x** in the insecticide-treated roots (`+` treatment). This aligns with ecological studies indicating that herbivore exclusion selects for goldenrod genotypes with elevated allelopathic polyacetylenes (e.g. CDME, which absorbs strongly in the UV range).

## 4. Detailed Tables

### Sample Scores (A Loading)
| vial | Component_1        | Component_2        | Component_3        |
| -----|--------------------|--------------------|------------------- |
| 119  | 443.73236083984375 | 14286.7998046875   | 1077.4547119140625 |
| 122  | 3495.680419921875  | 1793.9967041015625 | 7893.4140625       |
| 121  | 7796.4853515625    | 311.488037109375   | 0.0                |
| 458  | 19528.896484375    | 1391.676513671875  | 470.20855712890625 |

### Learned Warping Parameters (Mean-Centered)
| vial | trt | alpha_C1             | beta_C1              | alpha_C2              | beta_C2               | alpha_C3              | beta_C3              |
| -----|-----|----------------------|----------------------|-----------------------|-----------------------|-----------------------|--------------------- |
| 119  | +   | 0.15648648142814636  | 0.04842383414506912  | -0.022146085277199745 | 0.0019159754738211632 | -0.011012629605829716 | 0.15000000596046448  |
| 122  | +   | -0.03142032399773598 | -0.12162788212299347 | 0.12028533220291138   | 0.11485983431339264   | -0.02397918701171875  | 0.01613948866724968  |
| 121  | -   | -0.06089770793914795 | 0.050675809383392334 | -0.06659674644470215  | -0.15000000596046448  | 0.1007709950208664    | -0.1382492482662201  |
| 458  | -   | -0.06416844576597214 | 0.022528236731886864 | -0.031542494893074036 | 0.047745052725076675  | -0.06577916443347931  | -0.02801409550011158 |

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
