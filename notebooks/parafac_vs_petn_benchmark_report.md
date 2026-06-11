# PETN-PARAFAC vs. Classical PARAFAC Comparative Benchmark Report
**Statistical evaluation averaged over N=10 independent random dataset seeds.**

This report compares the performance of **Classical Non-Negative PARAFAC (TensorLy)**—both under raw conditions and with standard **2D Scattering Interpolation** preprocessing—against our **Physics-Embedded Tensor Network (PETN-PARAFAC)** across four EEM simulation scenarios.

---

## 🔬 1. Methodology & Tool Justification

In analytical chemometrics, fitting raw Excitation-Emission Matrices (EEMs) containing Rayleigh and Raman scattering diagonals is known to corrupt resolved loading profiles. Standard practice requires preprocessing the data to isolate and remove these artifacts before applying the trilinear PARAFAC decomposition. To ensure a scientifically rigorous and fair comparison, this benchmark compares our PETN-PARAFAC model against two classical baselines:

### A. Classical PARAFAC (Raw)
Acts as the negative control, fitting a standard non-negative PARAFAC model directly to the raw corrupted tensor. This demonstrates the extent of spectral warping when interferences are left untreated.

### B. Classical PARAFAC (Interpolated)
Represents the state-of-the-art preprocessing pipeline used in industry-standard software toolboxes like **`drEEM`** (MATLAB) and **`staRdom`/`eemR`** (R):
*   **Scattering Excision**: The 1st and 2nd order Rayleigh and water Raman scattering diagonals are identified and excised (set to `NaN` or masked out).
*   **2D Spatial Interpolation**: The missing pixels are filled in using surrounding valid data points. In MATLAB's `drEEM` (via the `eemscat` routine), this utilizes MATLAB's native `scatteredInterpolant` class, which constructs a **2D Delaunay Triangulation** of the valid points and performs linear or natural-neighbor interpolation.
*   **Python Emulation**: In our benchmark script, we implement this identical protocol using `scipy.interpolate.griddata(..., method='linear')` with a nearest-neighbor extrapolation fallback for boundaries. This guarantees that the preprocessed baseline fed to TensorLy's `non_negative_parafac` is mathematically equivalent to the output of standard chemometrics preprocessing pipelines.

### C. Physics-Embedded Tensor Network (PETN-PARAFAC)
Our gray-box model operates directly on the raw corrupted EEMs without any preprocessing interpolation. Instead, physical constraints are embedded inside the model's graph:
*   **Masked Loss vs. Weighted PARAFAC (W-PARAFAC)**: In traditional chemometrics, one can theoretically down-weight scattering regions using W-PARAFAC. However, solving W-PARAFAC with Alternating Least Squares (ALS) requires heavy iterative updates that are computationally slow and highly prone to local minima. PETN achieves this naturally by element-wise multiplying the loss gradients by a binary mask ($W$) during backpropagation, blinding the optimizer to the diagonals.
*   **Cuvette Attenuation Head**: To resolve the non-linear Cuvette Inner Filter Effect (IFE), the PETN embeds a physical Beer-Lambert layer inside the forward graph routing: $\hat{I}_{obs} = I_{true} \times 10^{-Abs}$. This physically binds the model's hypothesis space, separating non-linear attenuation from the pure trilinear chemical loadings.

### D. Robust Seed Evaluation
To verify that performance advantages are not a result of favorable random seed selection, the benchmark generates **N=10 independent datasets** from different seeds (seeds 42 to 51). The table below reports the **Mean ± Standard Deviation** of the $R^2$ recovery metrics across all runs.

---

## 📊 2. Comparative Metrics Table

| Scenario | Method | Scores $R^2$ (Concentration) | Excitation $R^2$ ($B$) | Emission $R^2$ ($C$) | Execution Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ideal System (Noise=0.005, No Scatter, No IFE)** | Classical PARAFAC (Raw) | 0.9999±0.0001 | 0.9997±0.0004 | 0.9997±0.0004 | 0.16s |
| | Classical PARAFAC (Interpolated) | 0.9999±0.0001 | 0.9997±0.0004 | 0.9997±0.0004 | 0.16s |
| | **PETN-PARAFAC** | **1.0000±0.0000** | **1.0000±0.0000** | **1.0000±0.0000** | 27.33s |
| --- | --- | --- | --- | --- | --- |
| **Scattered System (Noise=0.005, Scatter=True, No IFE)** | Classical PARAFAC (Raw) | 0.0798±0.0317 | 0.0000±0.0000 | 0.0000±0.0000 | 0.07s |
| | Classical PARAFAC (Interpolated) | 0.9829±0.0070 | 0.9218±0.0395 | 0.9222±0.0166 | 0.09s |
| | **PETN-PARAFAC** | **0.9991±0.0002** | **0.9878±0.0041** | **0.9860±0.0020** | 27.39s |
| --- | --- | --- | --- | --- | --- |
| **IFE Attenuated System (Noise=0.005, No Scatter, IFE=True)** | Classical PARAFAC (Raw) | 0.9527±0.0075 | 0.9803±0.0035 | 0.9982±0.0005 | 0.17s |
| | Classical PARAFAC (Interpolated) | 0.9527±0.0075 | 0.9803±0.0035 | 0.9982±0.0005 | 0.17s |
| | **PETN-PARAFAC** | **0.9984±0.0020** | **0.9991±0.0011** | **1.0000±0.0000** | 31.53s |
| --- | --- | --- | --- | --- | --- |
| **Fully Corrupted System (Noise=0.005, Scatter=True, IFE=True)** | Classical PARAFAC (Raw) | 0.0340±0.0140 | 0.0000±0.0000 | 0.0000±0.0000 | 0.08s |
| | Classical PARAFAC (Interpolated) | 0.6392±0.1668 | 0.2589±0.3174 | 0.3856±0.2459 | 0.13s |
| | **PETN-PARAFAC** | **0.9866±0.0068** | **0.9470±0.0162** | **0.9473±0.0102** | 29.59s |
| --- | --- | --- | --- | --- | --- |

---

## 🔍 3. Key Insights & Analysis

### A. Ideal Conditions (Scenario 1)
*   **Observation**: All methods converge to high score and loading recovery ($R^2 \ge 0.9999$).
*   **Takeaway**: In the absence of physical interferences, standard ALS (PARAFAC) and Gradient Descent (PETN) yield identical mathematical and physical factorizations, confirming that PETN acts as a mathematically valid PARAFAC replica in linear settings.

### B. Handling Scattering Interferences (Scenario 2)
*   **Observation**: Raw Classical PARAFAC fails completely ($R^2 = 0.0798\pm0.0317$ for scores). Pre-interpolating EEMs allows PARAFAC to recover components very well ($R^2 = 0.9829\pm0.0070$ scores).
*   **Observation**: PETN-PARAFAC, operating directly on the raw corrupted EEMs without preprocessing, achieves even higher recovery ($R^2 = 0.9991\pm0.0002$ scores, $R^2 \ge 0.9860\pm0.0041$ loadings).
*   **Takeaway**: Traditional interpolation introduces small spatial smoothing errors near the boundaries of the scattering lines. PETN avoids this because it does not interpolate; its **Masked Loss** simply ignores those pixels, and the rigid trilinear outer product mathematically interpolates the underlying signal during the factorization run.

### C. Resolving Cuvette Inner Filter Effects (Scenario 3)
*   **Observation**: Under non-linear IFE attenuation, both Raw and Interpolated Classical PARAFAC show degraded recovery (scores $R^2 = 0.9527\pm0.0075$). PETN-PARAFAC resolves the scores and loadings at **$R^2 \ge 0.9984\pm0.0020$** across all seeds.
*   **Takeaway**: Scattering interpolation cannot help with the Inner Filter Effect because IFE is a concentration-dependent, volume-wide absorption non-linearity rather than a spatial diagonal artifact. Classical PARAFAC's linear structure cannot accommodate this, leading to warped loading vectors. PETN's **Cuvette Attenuation Head** successfully deconvolves the non-linear attenuation.

### D. Combined Multi-Artifact Systems (Scenario 4)
*   **Observation**: Under combined interferences, even with Scattering Interpolation, Classical PARAFAC breaks down completely (**$R^2 = 0.6392\pm0.1668$** for scores, **$R^2 = 0.2589\pm0.3174$** for excitation loadings). PETN-PARAFAC maintains outstanding performance, achieving **$R^2 = 0.9866\pm0.0068$** for scores and **$R^2 \ge 0.9470\pm0.0162$** for loadings across all 10 independent datasets.
*   **Takeaway**: This is the most crucial result. When a system is affected by multiple overlapping interferences (scatter + IFE), traditional linear methods get stuck in local minima or completely fail to resolve components. PETN handles both interferences natively in a unified optimization run, proving to be a highly robust grey-box method for real-world spectroscopy calibration.

---

## ⚡ 4. Computational Cost
*   Preprocessing interpolation adds a small overhead (~0.05s) to the Classical PARAFAC pipeline, which remains extremely fast (~0.13s total).
*   PETN-PARAFAC is slower (~29.6s on CPU) as it optimizes weights via iteration loops in PyTorch. However, this represents a highly acceptable trade-off given the massive gains in chemical resolution and concentration prediction accuracy.
