# PETN-PARAFAC vs. Classical PARAFAC Comparative Benchmark Report

This report compares the performance of **Classical Non-Negative PARAFAC (TensorLy)** and our **Physics-Embedded Tensor Network (PETN-PARAFAC)** across four laboratory EEM simulation scenarios containing common spectroscopy interferences.

---

## 📊 Comparative Metrics Table

| Scenario | Method | Scores $R^2$ (Concentration) | Excitation $R^2$ ($B$) | Emission $R^2$ ($C$) | Execution Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ideal System (Noise=0.005, No Scatter, No IFE)** | Classical PARAFAC | 0.9999 | 0.9997 | 0.9997 | 0.23s |
| | **PETN-PARAFAC** | **1.0000** | **1.0000** | **1.0000** | 40.97s |
| --- | --- | --- | --- | --- | --- |
| **Scattered System (Noise=0.005, Scatter=True, No IFE)** | Classical PARAFAC | 0.0642 | 0.0000 | 0.0000 | 0.03s |
| | **PETN-PARAFAC** | **0.9993** | **0.9876** | **0.9871** | 38.97s |
| --- | --- | --- | --- | --- | --- |
| **IFE Attenuated System (Noise=0.005, No Scatter, IFE=True)** | Classical PARAFAC | 0.9578 | 0.9783 | 0.9980 | 0.23s |
| | **PETN-PARAFAC** | **0.9989** | **0.9989** | **1.0000** | 39.68s |
| --- | --- | --- | --- | --- | --- |
| **Fully Corrupted System (Noise=0.005, Scatter=True, IFE=True)** | Classical PARAFAC | 0.0316 | 0.0000 | 0.0000 | 0.04s |
| | **PETN-PARAFAC** | **0.9896** | **0.9487** | **0.9546** | 39.99s |
| --- | --- | --- | --- | --- | --- |

---

## 🔍 Key Insights & Analysis

### 1. Ideal Conditions (Scenario 1)
*   **Observation**: Both Classical PARAFAC and PETN recover scores and loadings with near-perfect accuracy ($R^2 > 0.999$).
*   **Takeaway**: In the absence of physical interferences, the optimization routines of both ALS (PARAFAC) and Gradient Descent (PETN) converge to the same global mathematical minimum.

### 2. Scattering Interferences (Scenario 2)
*   **Observation**: Classical PARAFAC breaks down completely, with scores recovery dropping to **$R^2 = 0.0642$** and spectral profiles dropping to **$R^2 = 0.0000$**. Meanwhile, PETN retains near-perfect recovery (**$R^2 \ge 0.987$** across all matrices).
*   **Takeaway**: Because Classical PARAFAC has no concept of masking, it attempts to fit the high-intensity scattering diagonal as an actual chemical loading. Since the diagonal shape is completely orthogonal to the true Gaussian fluorophore spectra, the ALS algorithm outputs unphysical spikes resulting in $0.00$ correlation. PETN's custom **Masked Loss** blinds the network to these diagonals, allowing its rigid outer-product core to smoothly interpolate the true chemical peaks underneath.

### 3. Cuvette Inner Filter Effect (Scenario 3)
*   **Observation**: Classical PARAFAC shows degraded concentrations recovery ($R^2 = 0.9578$) and distorted excitation loadings ($R^2 = 0.9783$). PETN resolves scores and loadings at **$R^2 \ge 0.998$**.
*   **Takeaway**: The cuvette Inner Filter Effect (IFE) violates the linear trilinear model assumption by non-linearly suppressing emission intensity at higher concentrations. Classical PARAFAC, operating on a strictly linear model, cannot model this suppression and skews its loadings and scores to minimize fit error. PETN's **Cuvette Attenuation Head** models the non-linear Beer-Lambert absorption in the forward pass, successfully separating attenuation from the pure spectra.

### 4. Fully Corrupted System (Scenario 4)
*   **Observation**: Classical PARAFAC breaks down completely under combined artifacts ($R^2 = 0.0316$ for scores, $0.0000$ for loadings). PETN maintains outstanding performance, resolving concentrations at **$R^2 = 0.9896$** and loading profiles at **$R^2 \ge 0.948$**.
*   **Takeaway**: Under combined scattering and IFE, Classical PARAFAC results are highly distorted and chemically uninterpretable. PETN successfully isolates and corrects both interferences simultaneously, proving to be a highly robust grey-box method for real-world spectroscopy calibration.

---

## ⚡ Computational Cost
*   Classical PARAFAC using Alternating Least Squares (ALS) is extremely fast (~0.03s to 0.23s) as it operates directly on NumPy arrays using closed-form linear updates.
*   PETN training uses Gradient Descent (Adam, 1500 epochs), which is more computationally intensive (~39s on CPU). However, this represents a highly acceptable trade-off given the massive gains in chemical resolution and concentration prediction accuracy.
