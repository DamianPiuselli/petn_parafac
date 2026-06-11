# PETN-PARAFAC vs. Classical PARAFAC Comparative Benchmark Report

This report compares the performance of **Classical Non-Negative PARAFAC (TensorLy)**—both under raw conditions and with standard **2D Scattering Interpolation** preprocessing—against our **Physics-Embedded Tensor Network (PETN-PARAFAC)** across four EEM simulation scenarios.

---

## 📊 Comparative Metrics Table

| Scenario | Method | Scores $R^2$ (Concentration) | Excitation $R^2$ ($B$) | Emission $R^2$ ($C$) | Execution Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ideal System (Noise=0.005, No Scatter, No IFE)** | Classical PARAFAC (Raw) | 1.0000 | 1.0000 | 1.0000 | 0.18s |
| | Classical PARAFAC (Interpolated) | 1.0000 | 1.0000 | 1.0000 | 0.18s |
| | **PETN-PARAFAC** | **1.0000** | **1.0000** | **1.0000** | 40.73s |
| --- | --- | --- | --- | --- | --- |
| **Scattered System (Noise=0.005, Scatter=True, No IFE)** | Classical PARAFAC (Raw) | 0.0642 | 0.0000 | 0.0000 | 0.04s |
| | Classical PARAFAC (Interpolated) | 0.9865 | 0.9231 | 0.9371 | 0.10s |
| | **PETN-PARAFAC** | **0.9993** | **0.9876** | **0.9871** | 39.37s |
| --- | --- | --- | --- | --- | --- |
| **IFE Attenuated System (Noise=0.005, No Scatter, IFE=True)** | Classical PARAFAC (Raw) | 0.9578 | 0.9783 | 0.9980 | 0.23s |
| | Classical PARAFAC (Interpolated) | 0.9578 | 0.9783 | 0.9980 | 0.23s |
| | **PETN-PARAFAC** | **0.9989** | **0.9989** | **1.0000** | 40.00s |
| --- | --- | --- | --- | --- | --- |
| **Fully Corrupted System (Noise=0.005, Scatter=True, IFE=True)** | Classical PARAFAC (Raw) | 0.0316 | 0.0000 | 0.0000 | 0.05s |
| | Classical PARAFAC (Interpolated) | 0.3535 | 0.0000 | 0.1344 | 0.14s |
| | **PETN-PARAFAC** | **0.9896** | **0.9487** | **0.9546** | 39.97s |
| --- | --- | --- | --- | --- | --- |

---

## 🔍 Key Insights & Analysis

### 1. Ideal Conditions (Scenario 1)
*   **Observation**: All methods converge to perfect score and loading recovery ($R^2 = 1.0000$).
*   **Takeaway**: In the absence of physical interferences, standard ALS (PARAFAC) and Gradient Descent (PETN) yield identical mathematical and physical factorizations.

### 2. Handling Scattering Interferences (Scenario 2)
*   **Observation**: Raw Classical PARAFAC fails completely ($R^2 = 0.0642$ scores, $0.0000$ loadings). However, once preprocessing **Scattering Interpolation** is applied, PARAFAC performance recovers significantly, yielding **$R^2 = 0.9865$** for scores.
*   **Observation**: PETN-PARAFAC, operating directly on the raw corrupted EEMs without any prior interpolation preprocessing, outperforms the preprocessed PARAFAC, achieving **$R^2 = 0.9993$** for scores and **$R^2 \ge 0.987$** for loading shapes.
*   **Takeaway**: Classical PARAFAC is highly sensitive to raw scatter and requires a separate, complex interpolation preprocessing pipeline. Furthermore, interpolation introduces small spatial errors near the boundary of the scattering lines. PETN avoids this by using a **Masked Loss** to blind the model to the corrupted diagonals during backpropagation, optimizing weights directly and solely on the true valid pixels.

### 3. Resolving Cuvette Inner Filter Effects (Scenario 3)
*   **Observation**: Under non-linear IFE attenuation, both Raw and Interpolated Classical PARAFAC show degraded recovery (scores $R^2 = 0.9578$). PETN-PARAFAC resolves the scores and loadings at **$R^2 \ge 0.998$**.
*   **Takeaway**: Standard scattering interpolation cannot help with the Inner Filter Effect because IFE is a concentration-dependent, volume-wide absorption non-linearity rather than a spatial artifact. Classical PARAFAC's linear structure cannot accommodate this, leading to warped loading vectors. PETN's **Cuvette Attenuation Head** models the physical Beer-Lambert equations inside the computational graph, successfully deconvolving the non-linear attenuation.

### 4. Fully Corrupted Systems (Scenario 4)
*   **Observation**: Under combined interferences, even with Scattering Interpolation, Classical PARAFAC breaks down completely (**$R^2 = 0.3535$** for scores, **$0.0000$** for excitation loadings). PETN-PARAFAC maintains outstanding performance, achieving **$R^2 = 0.9896$** for scores and **$R^2 \ge 0.948$** for loadings.
*   **Takeaway**: This is the most crucial result. When a system is affected by multiple overlapping artifacts (scatter + IFE), traditional linear methods get stuck in local minima or completely fail to resolve components. PETN handles both interferences natively in a unified optimization run, yielding superior resolution and concentration predictability under complex, realistic laboratory conditions.

---

## ⚡ Computational Cost
*   Preprocessing interpolation adds a small overhead (~0.05s) to the Classical PARAFAC pipeline, which remains extremely fast (~0.14s total).
*   PETN-PARAFAC is slower (~40s on CPU) as it optimizes weights via iteration loops in PyTorch. However, this represents a highly acceptable trade-off given the massive gains in chemical resolution and concentration prediction accuracy.
