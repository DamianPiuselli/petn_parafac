# Physics-Informed Neural Network (PINN) for EEM Spectroscopy

A hybrid, Gray-Box Physics-Informed Neural Network (PINN) designed to resolve Excitation-Emission Matrix (EEM) fluorescence spectroscopy data under scattering artifacts and non-linear Inner Filter Effects (IFE). 

This project bridges classical multi-way chemometrics (like PARAFAC) with modern deep learning, restricting the neural network's hypothesis space using physical laws to achieve complete mathematical interpretability and extreme data efficiency.

---

## 1. Motivations & Physical Principles

Traditional chemometric calibration (e.g., linear PARAFAC) assumes a strict trilinear structure ($I \times J \times K$) of clean chemical signals. However, real-world laboratory EEM data violates this assumption due to two major physical interferences:
1.  **Optical Scattering:** 1st and 2nd order Rayleigh scattering ($\lambda_{em} = \lambda_{ex}$ and $\lambda_{em} = 2\lambda_{ex}$) and solvent Raman scattering create high-intensity diagonal bands that corrupt underlying chemical data.
2.  **Inner Filter Effect (IFE):** Matrix absorption attenuates both excitation and emission light, causing a non-linear concentration-dependent suppression and distortion of fluorescence intensity.

### The Physics-Informed Cuvette Architecture
The model embeds physical laws directly into the network graph:
*   **Trilinear Core (White-Box):** Maps inputs to non-negative embedding tables representing Sample Scores ($A$), Excitation Loadings ($B$), and Emission Loadings ($C$), combined via a tensor outer product:
    $$I_{\text{true}}(i,j,k) = \sum_{r=1}^{R} a_{ir} \cdot b_{jr} \cdot c_{kr}$$
*   **Cuvette Attenuation Layer (Gray-Box):** Evaluates the Beer-Lambert and Lakowicz equations using a learnable molar absorptivity scale ($\alpha_r$) and registered solvent background absorbances ($Abs_{bg}$):
    $$Abs_{ex, i}(j) = \sum_{r=1}^R a_{ir} \cdot (\alpha_r \cdot B_{jr}) + Abs_{bg, ex}(j)$$
    $$Abs_{em, i}(k) = Abs_{bg, em}(k) \quad (\text{Emission absorptivity } M = 0 \text{ due to Stokes Shift})$$
    $$\gamma_i(j, k) = 10^{-(Abs_{ex, i}(j) + Abs_{em, i}(k))}$$
    $$\hat{I}_{\text{obs}}(i, j, k) = I_{\text{true}}(i, j, k) \times \gamma_i(j, k)$$
*   **Custom Masked Loss:** Accepts a binary mask ($W$) which equals `0` on the Rayleigh/Raman scattering diagonals. Gradients on these diagonals are element-wise multiplied by $W$ during backpropagation, blinding the trilinear core to the artifacts and forcing it to smoothly interpolate the true chemical signal underneath.

### Resolving the Scaling Degeneracy
By registering the background solvent absorbances ($ex\_bg$ and $em\_bg$) as **fixed buffers** (simulating standard laboratory blank subtraction), we break a mathematical scaling degeneracy:
$$C_{kr} \rightarrow C_{kr} \cdot 10^{\delta_k} \quad \text{and} \quad Abs_{bg, em}(k) \rightarrow Abs_{bg, em}(k) + \delta_k$$
This anchors the scores ($A$) and loading profiles ($B, C$) preventing component warping and ensuring unique, physically interpretable solutions.

---

## 2. Repository Structure

```
pinn_parafac/
├── data/
│   └── raw/
│       └── amino.mat              # Experimental Amino Acids benchmark dataset
├── notebooks/
│   ├── phase3_eem_heatmaps.png    # Heatmaps showing clean, corrupted, and reconstructed EEMs
│   ├── phase3_resolved_profiles.png # Resolved loadings vs true loadings (synthetic)
│   ├── phase3_resolved_absorptivities.png # Molar absorptivities (synthetic)
│   └── real_resolved_profiles.png # Resolved spectra for the experimental dataset
├── src/
│   ├── generator.py               # EEM synthetic generator with scatter & Lakowicz IFE
│   ├── loss.py                    # Custom masked MSE loss implementation
│   ├── model.py                   # PINNParafac custom model class in PyTorch
│   ├── train.py                   # Synthetic pipeline training loop
│   ├── train_real.py              # Experimental benchmark validation script
│   └── utils.py                   # Visualizations and plotting helpers
├── tests/
│   ├── test_generator.py          # Unit tests for the synthetic data generator
│   ├── test_loss.py               # Unit tests for masked loss
│   └── test_model.py              # Unit tests for PINN custom model
├── requirements.txt               # Project python package requirements
└── README.md                      # Project documentation
```

---

## 3. Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/username/pinn_parafac.git
    cd pinn_parafac
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 4. Usage Instructions

### Run Automated Unit Tests
To verify all model properties, constraints, and generator shapes:
```bash
pytest
```

### Run Synthetic Benchmark Pipeline (Phase 3)
Generates a synthetic EEM dataset with combined scattering and IFE corruption, trains the model jointly, and saves resolved loading comparisons:
```bash
PYTHONPATH=. python src/train.py
```
*   **Outputs:** Plots saved to `notebooks/phase3_resolved_profiles.png`, `notebooks/phase3_resolved_absorptivities.png`, and `notebooks/phase3_eem_heatmaps.png`.

### Run Real-World Validation (Phase 4)
Downloads, preprocesses, and trains the model on the experimental **Amino Acids** benchmark dataset containing mixtures of Tryptophan, Tyrosine, and Phenylalanine:
```bash
# 1. Download the raw benchmark
python -c "import urllib.request, zipfile, io; zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(urllib.request.Request('https://sid.erda.dk/share_redirect/AWqL5T6JCZ', headers={'User-Agent': 'Mozilla/5.0'})).read())).extractall('data/')"
python -c "import zipfile, os; zipfile.ZipFile('data/Amino_Acid_fluo.zip').extractall('data/raw/')"

# 2. Run calibration training
PYTHONPATH=. python src/train_real.py
```
*   **Outputs:** 
    *   Prints concentration recovery $R^2$ scores (yielding an average score recovery **$R^2 \approx 0.973$**).
    *   Verifies excitation and emission spectra peaks match literature standards.
    *   Saves resolved spectra plots to `notebooks/real_resolved_profiles.png`.
