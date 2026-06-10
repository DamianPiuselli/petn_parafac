This step-by-step implementation backlog is structured like an agile development plan. We start with the absolute simplest version of the system to ensure the baseline math works, and then systematically layer on the real-world complexities (scattering, non-linearities, and dynamic masks).

---

## Phase 1: The Minimum Viable Product (MVP)

**Goal:** Build a clean, synthetic, perfectly linear data generator and confirm that our custom deep learning architecture can replicate standard PARAFAC results. No scattering, no IFE yet.

### Step 1.1: Develop the Pure Synthetic Generator

* **Task:** Write a Python script using `NumPy` to generate a three-way tensor $\mathcal{X}$ ($I \times J \times K$) from true latent components ($R=3$).
* **Details:** Use pure Gaussian profiles for the excitation and emission dimensions, and a random matrix for sample intensities. Add a tiny amount of homoscedastic Gaussian noise.
* **Definition of Done (DoD):** A script that outputs an array of shape, for example, `(20, 60, 100)` along with the ground-truth arrays so you can benchmark against them later.

### Step 1.2: Build the Trilinear Core Model

* **Task:** Implement the embedding-based trilinear core in `TensorFlow` or `PyTorch`.
* **Details:** Map inputs `(sample_idx, ex_idx, em_idx)` to their respective non-negative embedding layers. The model should output the dot product of these three embeddings.
* **DoD:** The model compiles, accepts a batch of coordinate triplets, and outputs a predicted intensity scalar for each triplet.

### Step 1.3: Train and Validate the MVP

* **Task:** Flatten your synthetic tensor into a coordinate dataframe `[sample_idx, ex_idx, em_idx, true_intensity]` and train the model.
* **Details:** Extract the trained weights from the embedding layers after convergence. Plot them against the ground-truth Gaussians generated in Step 1.1.
* **DoD:** An $R^2 > 0.99$ correlation between predicted weights and true spectral shapes.

---

## Phase 2: Simulating and Masking Instrument Artifacts

**Goal:** Introduce scattering lines to the data generator and teach the model to ignore them using a predefined mask.

### Step 2.1: Add Scattering to the Generator

* **Task:** Update the synthetic generator script to inject 1st-order Rayleigh ($\lambda_{em} = \lambda_{ex}$) and Water Raman scatter lines into the tensor.
* **Details:** Make these lines high-intensity ridges that completely overwrite the underlying fluorescence signal at those specific pixel coordinates.
* **DoD:** Visual inspection of a simulated EEM heatmap showing distinct diagonal bands cutting through the chemical peaks.

### Step 2.2: Implement the Static Masked Loss Function

* **Task:** Code a custom loss function (`masked_mse_loss`) that reads a hardcoded binary mask matrix $W$.
* **Details:** $W$ equals `0` on the scattering diagonals and `1` everywhere else. The loss function must multiply the squared errors by this mask before calculating the mean, effectively blinding the backpropagation step to the scattering zones.
* **DoD:** Successful training on corrupted data where the model recovers the pure chemical profiles *underneath* the scattering zones without capturing the artifacts.

---

## Phase 3: The Non-Linear Upgrade (Inner Filter Effects)

**Goal:** Introduce the chemical non-linearity (absorbance attenuation) and build the hybrid gray-box head to fix it.

### Step 3.1: Inject IFE into the Simulation

* **Task:** Update the generator to apply the Lakowicz geometric correction formula.
* **Details:** Define an independent absorbance matrix for the sample background. Multiply the true tensor elements by $10^{-(A_{ex} + A_{em})}$. This will visibly skew and suppress the emission peaks at higher concentrations.
* **DoD:** Synthetic data that explicitly breaks the linear PARAFAC assumptions (i.e., if you try to fit Phase 1's model to this data, the spectral recovery fails).

### Step 3.2: Construct the Hybrid "Gray-Box" Network

* **Task:** Add the parallel Dense Neural Network (the "Black-Box" head) to the architecture.
* **Details:** This sub-network takes `(ex_idx, em_idx)` as inputs, passes them through dense layers with ReLU activations, ends with a Sigmoid layer, and element-wise multiplies its output with the trilinear core output.
* **DoD:** The network architecture accurately represents the physical equation: $\hat{I}_{obs} = I_{true} \times \gamma_{ife}$.

### Step 3.3: Evaluate the Complete Synthetic Pipeline

* **Task:** Train the hybrid network on the IFE-corrupted, scattered dataset.
* **Details:** Verify that the deep learning head accurately maps the non-linear attenuation landscape while the trilinear embedding layer successfully extracts the true unattenuated chemical components.
* **DoD:** Reaching equivalent accuracy ($R^2 > 0.95$) on highly non-linear data as achieved in the simple Phase 1 baseline.

---

## Phase 4: Autonomy and Real-World Validation

**Goal:** Transition from synthetic data to real-world datasets, automating the discovery of parameters.

### Step 4.1: Develop Dynamic Scattering Detection (Optional / Advanced)

* **Task:** Replace the predefined static mask with a dynamic loss-weighting network.
* **Details:** Build a secondary network head that identifies spatial anomalies in the residuals and dynamically sets $W \rightarrow 0$ for high-error, non-trilinear regions during early epochs.
* **DoD:** The model successfully detects and isolates scattering lines on a completely new synthetic layout without manual configuration.

### Step 4.2: Import and Preprocess Real Benchmarks

* **Task:** Write a data pipeline to ingest open-access `.mzML`, `.CDF`, or `.csv` files from the University of Copenhagen repository (e.g., the Honey or Micropollutants dataset).
* **Details:** Parse the files into the required coordinate format (`[Sample, Ex, Em, Intensity]`) used by the deep learning framework.
* **DoD:** Successful loading and batching of real instrument data through the training loop.

### Step 4.3: Final Benchmarking & Slide Preparation

* **Task:** Run the completed model against the real dataset. Compare its performance, execution speed, and component resolution clarity against classical PARAFAC.
* **DoD:** Production of high-resolution comparative plots (True vs. Recovered Profiles) ready to be embedded into the final symposium presentation.

---

## Phase 5: Future Work & Production Features (Backlog)

### Step 5.1: Semi-Supervised Training Guidance
* **Task:** Implement an auxiliary supervised loss term $\lambda \mathcal{L}_{\text{supervised}}$ to constrain the score embeddings ($A_{std}$) of calibration standards to their known physical concentrations during training.
* **Benefits:** Guides component separation in complex mixtures and resolves absolute physical units for molar absorptivity ($\alpha_r$) and scores.

### Step 5.2: Frozen Projection Inference Utility
* **Task:** Implement a single-sample inference helper `predict_unknown(new_eem, model)` that freezes the trained loading parameters ($B$, $C$, $\alpha$) and solves a non-linear optimization problem to predict concentrations ($a_{\text{new}}$) for new unknown EEM samples.

---

Which phase or specific tool within this backlog would you like to start drafting first? We can begin writing the core Python code for the synthetic generator or design the neural network structure.
