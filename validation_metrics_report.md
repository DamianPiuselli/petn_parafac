# Validation Metrics for PARAFAC, PARAFAC2, and Chroma-PETN: A Literature Review and Implementation Guide

This report provides a comprehensive review of validation diagnostics for PARAFAC and PARAFAC2 models in the absence of ground truth data, and derives a novel method to adapt these diagnostics for validating our gray-box **Chroma-PETN** (Physics-Embedded Tensor Network) model.

---

## 1. The Core Consistency Diagnostic (CORCONDIA)

The Core Consistency Diagnostic (CORCONDIA), proposed by Bro and Kiers (2003), is the standard heuristic for validating Canonical Polyadic / PARAFAC models and determining the optimal number of components ($R$).

### A. Mathematical Definition for PARAFAC
A standard three-way PARAFAC model decomposes a tensor $\mathcal{X} \in \mathbb{R}^{I \times J \times K}$ into three loading matrices: $\mathbf{A} \in \mathbb{R}^{I \times R}$ (scores), $\mathbf{B} \in \mathbb{R}^{J \times R}$, and $\mathbf{C} \in \mathbb{R}^{K \times R}$. This can be represented as:

$$\mathcal{X} \approx \mathcal{S} \times_1 \mathbf{A} \times_2 \mathbf{B} \times_3 \mathbf{C}$$

where $\mathcal{S} \in \mathbb{R}^{R \times R \times R}$ is the **super-diagonal target core array** with elements $s_{pqr} = 1$ if $p = q = r$ and $s_{pqr} = 0$ otherwise.

To evaluate model consistency, CORCONDIA estimates an unconstrained **Tucker3 core tensor** $\mathcal{G} \in \mathbb{R}^{R \times R \times R}$ by keeping the calculated loading matrices $\mathbf{A}, \mathbf{B}, \mathbf{C}$ fixed and solving the linear least-squares regression problem:

$$\min_{\mathcal{G}} \|\mathcal{X} - \mathcal{G} \times_1 \mathbf{A} \times_2 \mathbf{B} \times_3 \mathbf{C}\|_F^2$$

Vectorizing the tensors converts this into a standard multiple linear regression system:

$$\text{vec}(\mathcal{G}) = (\mathbf{C} \otimes \mathbf{B} \otimes \mathbf{A})^\dagger \text{vec}(\mathcal{X})$$

where $\otimes$ denotes the Kronecker product and $\dagger$ represents the Moore-Penrose pseudoinverse. Using the properties of Kronecker products, the estimated core can be computed via multi-mode projection:

$$\mathcal{G} = \mathcal{X} \times_1 \mathbf{A}^\dagger \times_2 \mathbf{B}^\dagger \times_3 \mathbf{C}^\dagger$$

where $\mathbf{A}^\dagger = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$.

Once $\mathcal{G}$ is calculated, the diagnostic compares it to the ideal super-diagonal core $\mathcal{S}$:

$$\text{CORCONDIA} = 100 \times \left( 1 - \frac{\sum_{p=1}^R \sum_{q=1}^R \sum_{r=1}^R (g_{pqr} - s_{pqr})^2}{\text{divisor}} \right)$$

#### Divisor Options:
1. **`\"nfac\"` (Default):** Normalizes by the sum of squares of the ideal core $\mathcal{S}$, which equals the number of components $R$:
   $$\text{divisor} = \sum_{p,q,r} s_{pqr}^2 = R$$
2. **`\"core\"`:** Normalizes by the sum of squares of the estimated core $\mathcal{G}$:
   $$\text{divisor} = \sum_{p,q,r} g_{pqr}^2$$

#### Interpretation:
- **$\approx 100\%$**: The data conforms perfectly to the trilinear PARAFAC structure.
- **$\geq 90\%$**: High model stability and validity.
- **$\approx 50\%$ to $80\%$**: Marginal; indicates some degree of over-factoring or structural deviation.
- **$< 50\%$ or Negative**: Strong evidence of overfitting or model mis-specification (the model is capturing noise, or the components are highly collinear, which causes the unconstrained core $\mathcal{G}$ to exhibit large off-diagonal interactions).

---

### B. Adaptation for PARAFAC2
The PARAFAC2 model relaxes the strict trilinear assumption to handle shifts or peak deformations by allowing the profile matrix in the second mode to vary across samples, such that $\mathbf{B}_i \in \mathbb{R}^{J \times R}$ for each sample $i \in \{1, \dots, I\}$. It imposes a constant cross-product constraint:

$$\mathbf{B}_i^T \mathbf{B}_i = \mathbf{B}^T \mathbf{B} \quad \forall i$$

which is implemented by parameterizing $\mathbf{B}_i = \mathbf{P}_i \mathbf{B}$, where $\mathbf{P}_i \in \mathbb{R}^{J \times R}$ has orthonormal columns ($\mathbf{P}_i^T \mathbf{P}_i = \mathbf{I}_R$) and $\mathbf{B} \in \mathbb{R}^{R \times R}$ is a common coordinate-representing matrix.

Because the profile matrix is sample-specific, a standard Tucker3 core cannot be directly estimated. However, Kamstrup-Nielsen et al. (2013) adapted CORCONDIA to PARAFAC2 using the **projection trick**:

1. Project each observed data slice $\mathbf{X}_i \in \mathbb{R}^{J \times K}$ onto its corresponding orthogonal basis $\mathbf{P}_i$:
   $$\tilde{\mathbf{X}}_i = \mathbf{P}_i^T \mathbf{X}_i \quad \in \mathbb{R}^{R \times K}$$
2. Stack these projected slices to form a compressed tensor $\tilde{\mathcal{X}} \in \mathbb{R}^{I \times R \times K}$.
3. Under the PARAFAC2 model, the compressed tensor follows a standard PARAFAC1 trilinear structure:
   $$\tilde{\mathbf{X}}_i \approx \mathbf{B} \mathbf{D}_i \mathbf{C}^T$$
   where $\mathbf{D}_i = \text{diag}(\mathbf{A}_{i, :})$.
4. Calculate the standard PARAFAC1 CORCONDIA on the compressed tensor $\tilde{\mathcal{X}}$ using the score matrix $\mathbf{A}$, the common matrix $\mathbf{B}$, and the spectral matrix $\mathbf{C}$:
   $$\mathcal{G} = \tilde{\mathcal{X}} \times_1 \mathbf{A}^\dagger \times_2 \mathbf{B}^\dagger \times_3 \mathbf{C}^\dagger$$
   $$\text{CORCONDIA}_{\text{PARAFAC2}} = 100 \times \left( 1 - \frac{\sum (g_{pqr} - s_{pqr})^2}{R} \right)$$

---

## 2. Other Standard Diagnostic Metrics

### A. Split-Half Analysis
* **Concept**: Split-half analysis is the gold standard for testing model stability. Because PARAFAC solutions are unique, fitting the model to two independent subsets of the data should yield identical loading vectors (up to scaling and permutation).
* **Procedure**:
  1. Split the data along the sample mode (first mode) into independent halves (e.g., Splitting 1: odd vs. even samples; Splitting 2: random halves).
  2. Fit the model to each half independently.
  3. Align the components of the two models (resolving permutation and scaling ambiguities).
  4. Compare the loading profiles of the non-split modes (chromatography and spectral modes) between the two models.
* **Interpretation**: If the components represent true physical/chemical phenomena, the resolved profiles will match across splits. If the model is over-factored, the splits will capture noise in different ways, resulting in mismatched profiles.

### B. Tucker Congruence Coefficient (TCC)
* **Concept**: TCC (also known as cosine similarity) measures the similarity between two loading vectors without mean centering, preserving absolute magnitudes and shapes.
* **Formula**:
  $$\text{TCC}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^T \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}$$
* **Thresholds**:
  - $\text{TCC} \geq 0.95$: Virtual identity.
  - $0.90 \leq \text{TCC} < 0.95$: High similarity.
  - $0.85 \leq \text{TCC} < 0.90$: Fair similarity.
  - $\text{TCC} < 0.85$: Failure to replicate (indicates instability).

### C. Percent Variance Explained (Fit Percentage)
* **Concept**: Quantifies the percentage of the data's total sum of squares captured by the model.
* **Formula**:
  $$\text{Fit \%} = \left( 1 - \frac{\sum_{i,j,k} (x_{ijk} - \hat{x}_{ijk})^2}{\sum_{i,j,k} x_{ijk}^2} \right) \times 100$$
* **Interpretation**: While high fit percentage is desirable, it must be balanced against CORCONDIA. A model with 99% fit but 10% core consistency is overfitted; a model with 96% fit and 98% core consistency is far more robust.

### D. Visual Inspection of Residuals
* **Concept**: Analyzing what the model failed to capture.
* **Chromatographic Visualizations**:
  - **Residual Heatmaps**: For each sample $i$, plot a 2D heatmap of residuals (Time vs. Spectra). 
  - **S-Curves**: Any misaligned peaks will display a characteristic \"s-curve\" shape in the residuals along the time axis.
  - **Unmodeled Peaks**: Systematic peaks in the residual heatmap indicate that a chemical component was missed (under-factored).
  - **Random Noise**: A successful alignment and fit will leave only random, unstructured noise.

---

## 3. Adapting and Implementing for Chroma-PETN

### A. Mathematical Adaptations
In our **Chroma-PETN** model, the chromatography profiles are warped using sample-specific continuous functions (linear, quadratic, or spline) rather than orthonormal projection matrices $\mathbf{P}_i$. The model forward pass computes the warped profiles:

$$(\mathbf{B}_i^{\text{warped}})_{j, r} = B^{\text{warped}}_{i, j, r}$$

To calculate the Core Consistency Diagnostic for Chroma-PETN, we define a **generalized Tucker3 core regression** that directly reflects our model structure:

$$x_{ijk} \approx \sum_{r=1}^R \sum_{s=1}^R \sum_{t=1}^R g_{rst} A_{ir} B^{\text{warped}}_{i, j, s} C_{kt}$$

Since $\mathbf{A}$, $\mathbf{B}_i^{\text{warped}}$, and $\mathbf{C}$ are fixed parameters after model training, this is a linear system of the form $\mathbf{y} \approx \mathbf{H} \mathbf{g}$. We construct the design matrix $\mathbf{H} \in \mathbb{R}^{(IJK) \times R^3}$ where the column corresponding to the index $(r, s, t)$ is:

$$H_{(i, j, k), (r, s, t)} = A_{ir} B^{\text{warped}}_{i, j, s} C_{kt}$$

The unconstrained core $\mathbf{g}$ is computed via least-squares:

$$\mathbf{g} = \mathbf{H}^\dagger \mathbf{y}$$

This formulation has significant advantages:
1. It is mathematically exact and directly mirrors the model's warping physics.
2. It avoids the need to numerically invert the warping function or interpolate raw data.
3. It scales efficiently ($R^3 = 27$ columns for a 3-component model).

---

## 4. Proposed Validation Workflow

For validating Chroma-PETN in a laboratory setting where ground truth spectra/concentrations are unavailable, we recommend the following step-by-step diagnostic workflow:

### A. Workflow Steps
1. **Fit and Core Consistency Comparison:** Fit models across a range of ranks (e.g., $R=1, 2, 3, 4$). Inspect the trade-off between Fit % and CORCONDIA. The optimal rank is typically the largest $R$ that maintains a **CORCONDIA > 90%**. A sharp drop in CORCONDIA (e.g., from 95% at $R=3$ to -10% at $R=4$) is a clear boundary indicating that the $R=4$ model is over-factored.
2. **Split-Half Validation:** Perform a split-half stability test on the selected $R$:
   * Split the dataset into odd and even samples.
   * Fit models to both halves independently.
   * Resolve component permutation and compute TCC between the two sets of resolved profiles (common chromatography profiles $\mathbf{B}_{canonical}$ and spectral profiles $\mathbf{C}$).
   * Verify that TCC is $\geq 0.95$ for all components across splits.
3. **Residual Inspection:** Generate residual heatmaps (Time vs. Spectra) for all samples:
   * Ensure that the heatmaps show only uniform random noise.
   * Look out for \"S-shaped\" residual bands which suggest that the warping head (e.g., linear vs. quadratic) did not have enough degrees of freedom to capture non-linear retention time shifting. If seen, upgrade the warp type to `spline` or `quadratic` in `ChromaPETN`.

---

## 5. Python Validation Module
Below is a complete, production-ready Python script implementing the validation metrics for the Chroma-PETN model.

```python
import torch
import numpy as np

@torch.no_grad()
def extract_chroma_petn_factors(model):
    \"\"\"
    Extracts the score, spectral loading matrices, and computes the 
    warped chromatographic profiles for all samples and time points.
    
    Returns:
        A: NumPy array of shape (num_samples, num_components)
        B_warped: NumPy array of shape (num_samples, num_time, num_components)
        C: NumPy array of shape (num_spec, num_components)
    \"\"\"
    model.eval()
    device = next(model.parameters()).device
    I = model.num_samples
    J = model.num_time
    K = model.num_spec
    R = model.num_components
    
    # 1. Extract standard factor matrices
    A = model.sample_embeddings.weight.detach().cpu().numpy()
    C = model.spec_embeddings.weight.detach().cpu().numpy()
    
    # 2. Compute B_warped by running the warping and interpolation logic
    coords_i, coords_j = torch.meshgrid(
        torch.arange(I, device=device),
        torch.arange(J, device=device),
        indexing='ij'
    )
    coords_i = coords_i.flatten()
    coords_j = coords_j.flatten()
    
    t = coords_j.float() / (J - 1)
    
    if model.warp_type == 'linear':
        stretch_i = model.warp_stretch[coords_i]
        shift_i = model.warp_shift[coords_i]
        t_warped = t - (stretch_i * t + shift_i)
    elif model.warp_type == 'quadratic':
        alpha_i = model.warp_alpha[coords_i]
        beta_i = model.warp_beta[coords_i]
        gamma_i = model.warp_gamma[coords_i]
        t_warped = t - (alpha_i * (t ** 2) + beta_i * t + gamma_i)
    elif model.warp_type == 'spline':
        shift_i = model.warp_shift[coords_i]
        log_inc_i = model.warp_log_increments[coords_i]
        
        inc_i = (1.0 / model.num_segments) * torch.exp(log_inc_i)
        zeros = torch.zeros((t.shape[0], 1), device=t.device, dtype=t.dtype)
        cum_inc = torch.cumsum(torch.cat([zeros, inc_i], dim=1), dim=1)
        w = shift_i.unsqueeze(-1) + cum_inc
        
        val = t * model.num_segments
        k = torch.clamp(torch.floor(val).long(), 0, model.num_segments - 1)
        u = val - k.float()
        
        batch_indices = torch.arange(t.shape[0], device=t.device)
        w_k = w[batch_indices, k]
        w_kp1 = w[batch_indices, k + 1]
        t_warped = (1.0 - u) * w_k + u * w_kp1
        
    x_warped = t_warped * (J - 1)
    x_clamped = torch.clamp(x_warped, 0.0, J - 1.0 - 1e-3)
    x_0 = torch.floor(x_clamped).long()
    x_1 = x_0 + 1
    
    w_interp = (x_clamped - x_0.float()).unsqueeze(-1)
    
    B_weights = model.time_embeddings.weight
    val_0 = B_weights[x_0]
    val_1 = B_weights[x_1]
    
    b_warped = (1.0 - w_interp) * val_0 + w_interp * val_1
    B_warped = b_warped.view(I, J, R).cpu().numpy()
    
    return A, B_warped, C


def compute_chroma_corcondia(X, A, B_warped, C, divisor='nfac'):
    \"\"\"
    Computes the Core Consistency Diagnostic (CORCONDIA) for Chroma-PETN 
    using the unconstrained Tucker3 core design matrix formulation.
    
    Args:
        X: Observed data tensor, shape (I, J, K)
        A: Score matrix, shape (I, R)
        B_warped: Warped chromatography profiles, shape (I, J, R)
        C: Spectral loading matrix, shape (K, R)
        divisor: Normalization method ('nfac' or 'core')
        
    Returns:
        core_consistency: Float value representing the core consistency percentage
        G: The estimated unconstrained Tucker core, shape (R, R, R)
    \"\"\"
    I, J, K = X.shape
    R = A.shape[1]
    
    # 1. Build design matrix H using broadcasting
    # H[i, j, k, r, s, t] = A[i, r] * B_warped[i, j, s] * C[k, t]
    A_exp = A[:, None, None, :, None, None]            # Shape: (I, 1, 1, R, 1, 1)
    B_exp = B_warped[:, :, None, None, :, None]        # Shape: (I, J, 1, 1, R, 1)
    C_exp = C[None, None, :, None, None, :]            # Shape: (1, 1, K, 1, 1, R)
    
    H = A_exp * B_exp * C_exp                          # Shape: (I, J, K, R, R, R)
    H = H.reshape(I * J * K, R * R * R)
    
    # 2. Flatten observed tensor
    y = X.flatten()
    
    # 3. Perform linear least squares to find the unconstrained core G
    g_flat, _, _, _ = np.linalg.lstsq(H, y, rcond=None)
    G = g_flat.reshape(R, R, R)
    
    # 4. Construct the ideal super-diagonal core array S
    S = np.zeros((R, R, R))
    for r in range(R):
        S[r, r, r] = 1.0
        
    # 5. Compute metric
    sum_sq_diff = np.sum((G - S) ** 2)
    
    if divisor == 'nfac':
        div_val = float(R)
    elif divisor == 'core':
        div_val = np.sum(G ** 2)
    else:
        raise ValueError(\"divisor must be either 'nfac' or 'core'\")
        
    core_consistency = 100.0 * (1.0 - (sum_sq_diff / div_val))
    return core_consistency, G


def compute_tcc(v1, v2):
    \"\"\"Computes the Tucker Congruence Coefficient (TCC) between two vectors.\"\"\"
    num = np.dot(v1, v2)
    denom = np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    return num / denom if denom != 0 else 0.0


def compute_percent_variance_explained(X, X_pred):
    \"\"\"Computes the uncentered percent variance explained (Fit %).\"\"\"
    residual_ss = np.sum((X - X_pred) ** 2)
    total_ss = np.sum(X ** 2)
    return 100.0 * (1.0 - (residual_ss / total_ss)) if total_ss != 0 else 0.0


def compute_residuals(X, X_pred):
    \"\"\"Computes the residuals tensor.\"\"\"
    return X - X_pred
```
