# Physics-Embedded Tensor Network (PETN) - Model Implementation Plan

This implementation plan is structured as an agile development backlog. It tracks the progress of our hybrid, "Gray-Box" deep learning library across two distinct tracks: EEM Spectroscopy and Chromatography Alignment. 

---

## 🔬 Track A: EEM Spectroscopy (Inner Filter Effect & Scattering)

### Phase A1: The Minimum Viable Product (MVP) | ✅ COMPLETED
* **Goal:** Replicate standard linear PARAFAC using coordinate embeddings.
* **Tasks:**
  * Developed a synthetic trilinear tensor generator (3 components, Gaussians, noise).
  * Built the PyTorch `PETNParafac` trilinear core using coordinate indexing.
  * Verified $R^2 > 0.99$ recovery of clean uncorrupted loadings.

### Phase A2: Artifact Masking (Optical Scattering) | ✅ COMPLETED
* **Goal:** Blinds the optimizer to high-intensity Rayleigh and Raman scattering lines.
* **Tasks:**
  * Added 1st/2nd order Rayleigh and water Raman scatter ridges to the simulator.
  * Coded `masked_mse_loss` that accepts a binary mask $W$ representing scattering regions.
  * Verified that the trilinear core successfully interpolates clean profiles directly underneath the scattering lines.

### Phase A3: Non-Linear Physical Corrections (IFE) | ✅ COMPLETED
* **Goal:** Resolve cuvette Inner Filter Effects (absorbance attenuation).
* **Tasks:**
  * Updated the simulator to inject concentration-dependent IFE attenuation based on the Lakowicz correction.
  * Designed the **Cuvette Attenuation Head** inside the forward pass, using a learnable component-specific molar absorptivity scaling factor ($\alpha_r$) and registered background solvent profiles ($\text{Abs}_{\text{bg}}$) to model physical attenuation $\gamma \in [0, 1]$.
  * Verified separation of linear chemical signals from non-linear absorbance.

### Phase A4: Real-World Benchmark Validation | 🔄 IN PROGRESS
* **Goal:** Transition from synthetic data to real instrument runs.
* **Tasks:**
  * [x] Ingest and preprocess the Copenhagen Honey dataset and the Amino Acids mixture dataset.
  * [x] Evaluate resolved score clusters: achieved **100% binary adulteration accuracy** and **72.7% multiclass origin accuracy** on Honey.
  * [ ] Validate EEM-PETN against real-world environmental mixtures with variable scattering matrices.

### Phase A5: Extensions & Production Features | 📋 BACKLOG
* **Step A5.1: Semi-Supervised Standard Constraints**
  * Constrain score embeddings $A$ of calibration standards to their known physical values using a supervised auxiliary loss term to resolve absolute concentration scales.
* **Step A5.2: Frozen Inference Projection**
  * Create a utility to freeze trained loading embeddings ($B, C, \alpha$) and solve a non-linear optimizer to estimate concentrations for new, unknown samples.
* **Step A5.3: MLP-Based Geometry Calibration**
  * Integrate a small MLP to learn the instrument-specific effective pathlength distribution, correcting for cuvette differences across spectrofluorometers.

---

## 🧪 Track B: Chromatography Alignment (GC-MS / HPLC-DAD Time Shifting)

### Phase B1: MVP Linear Warping Alignment | ✅ COMPLETED
* **Goal:** Resolve components under run-to-run linear shifting and stretching.
* **Tasks:**
  * [x] Developed [generator.py](./src/chroma/generator.py) to simulate overlapping chromatographic peaks subject to random delay offsets ($\beta_i$) and flow rate stretches ($\alpha_i$).
  * [x] Implemented the `ChromaPETN` model in [model.py](./src/chroma/model.py) using a **differentiable warping head** and a custom **differentiable 1D linear interpolation** layer.
  * [x] Implemented the **mean-centering constraint** ($\sum \alpha_i = 0, \sum \beta_i = 0$) inside `project_constraints()` to resolve the shift-translation and scaling ambiguities.
  * [x] Coded unit and integration tests in `tests/chroma/` (all passing cleanly).
  * [x] Added visual reporting in [train.py](./src/chroma/train.py) that saves resolved profiles, TIC alignment comparisons, and parameter correlations to `notebooks/chroma/`.

### Phase B2: Non-Linear Warping (Splines & Quadratic Upgrade) | ✅ COMPLETED
* **Goal:** Handle non-linear elution shifting common in gradient HPLC runs.
* **Tasks:**
  * [x] Upgraded the warping head in [model.py](./src/chroma/model.py) to support **quadratic warping** ($\Delta_i(t) = \alpha_i t^2 + \beta_i t + \gamma_i$) and **uniform piecewise linear spline warping** (defined by $M$ control points).
  * [x] Enforced strict monotonicity on both quadratic warping (via parameter boundaries) and spline warping (via parameterized log-increments $\theta_{i,k}$).
  * [x] Added unit tests verifying monotonicity, constraint projections, and forward passes in [test_model.py](./tests/chroma/test_model.py) (all passing cleanly).
  * [ ] **Future Refinement (Step B2.1): Physics-Informed M** (Backlog)
    * Support non-uniform control point placement based on the mobile phase gradient or temperature ramp profiles from instrument method files to prevent overfitting.

### Phase B3: Real-World Chromatographic Validation | 📋 BACKLOG
* **Goal:** Ingest and align real instrument runs (HPLC-DAD or GC-MS).
* **Tasks:**
  * Preprocess standard datasets (e.g., polycyclic aromatic hydrocarbons (PAHs) or organic acids).
  * Train Chroma-PETN to align raw peaks and extract pure UV-vis/mass spectra.
  * Compare resolved scores against reference concentrations.

### Phase B4: Automated Peak Identification | 📋 BACKLOG
* **Goal:** Integrate structural database matching directly into training.
* **Tasks:**
  * Add a spectral database mapping layer (e.g., matching resolved mass spectra embeddings $C$ against NIST or Wiley database spectra).
  * Compute similarity matching on the fly during training to automatically annotate chemical components.
