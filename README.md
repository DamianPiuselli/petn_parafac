# Physics-Embedded Tensor Networks (PETN) for Analytical Chemistry

This repository houses a hybrid, "Gray-Box" deep learning library designed to resolve complex multi-way calibration datasets in analytical chemistry. By embedding physical and chemical laws directly into PyTorch network routing, the models maintain complete mathematical interpretability, resolve rotational ambiguity, and require very small sample batches (e.g., 5-10 physical samples) to train.

---

## 🚀 Supported Analytical Domains

The library is organized into two self-contained subpackages targeting distinct physical interferences:

### 1. Excitation-Emission Matrix (EEM) Fluorescence Spectroscopy (`src/eem/`)
* **Physical Target:** Solves the non-linear **Inner Filter Effect (IFE)** and corruptive **Optical Scattering** (Rayleigh and Raman).
* **Architecture Highlights:**
  * **White-Box Core:** Trilinear PARAFAC embeddings with hardcoded non-negative constraints.
  * **Gray-Box Attenuation Head:** Implements Beer-Lambert and Lakowicz cuvette equations to predict sample-specific attenuation factors $\gamma \in [0, 1]$.
  * **Custom Masked Loss:** Blinds backpropagation to scattering zones, forcing the rigid trilinear core to smoothly interpolate the true underlying chemical signal.
* **Documentation:** Detailed guide available at [src/eem/README.md](./src/eem/README.md).

### 2. Chromatography with Retention Time Shifting (`src/chroma/`)
* **Physical Target:** Resolves non-trilinearity in GC-MS or HPLC-DAD datasets caused by column flow rate drift, temperature fluctuations, and injection delay shifts.
* **Architecture Highlights:**
  * **Differentiable Warping Head:** Parameterizes sample-specific time stretching ($\alpha_i$) and shifting ($\beta_i$) coefficients (supporting linear, quadratic, and spline warping functions).
  * **Differentiable 1D Interpolation:** Natively queries a common/aligned chromatography profile embedding ($B$) at warped continuous coordinates.
  * **Differentiable Savitzky-Golay Derivative Layer:** Implements a convolutional layer using Savitzky-Golay filters to compute analytical derivatives (e.g. second derivatives) on the fly during training, resolving severe peak overlaps and background drifts.
  * **Mean-Centering Constraint:** Resolves translation/scaling degeneracy by forcing the learned warping parameters to average to zero, anchoring the canonical profile coordinates.
  * **Variance-Scaled Early Stopping:** Scales the minimum convergence delta dynamically based on the target signal variance ($y_{\text{target}}$ variance) to ensure scale-invariant training convergence.
* **Documentation:** Detailed guide available at [src/chroma/README.md](./src/chroma/README.md).

---

## 📁 Repository Structure

```
petn_parafac/
├── data/
│   ├── eem/                # EEM datasets (CDOM, Honey, Aminoacids, etc.)
│   └── chroma/             # Chromatography datasets (GC-MS, HPLC-DAD)
├── notebooks/
│   ├── eem/                # EEM analysis notebooks
│   └── chroma/             # Chromatography alignment notebooks
├── src/
│   ├── common/             # Shared utilities (plotting, base classes)
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── eem/                # EEM Spectroscopy Subpackage
│   │   ├── __init__.py
│   │   ├── model.py        # PETN-PARAFAC model with IFE attenuation head
│   │   ├── loss.py         # Custom masked MSE loss
│   │   ├── generator.py    # Synthetic EEM dataset generator
│   │   ├── train.py        # EEM training script
│   │   ├── benchmark.py    # EEM comparative benchmark script
│   │   └── README.md       # EEM-specific detailed documentation
│   └── chroma/             # Chromatography Subpackage
│       ├── __init__.py
│       ├── base.py         # Base class (BaseChromaPETN) with core parameters and warping equations
│       ├── hplc.py         # HPLC-DAD specific PETN (HPLC_PETN) with baseline and derivatives
│       ├── gcms.py         # GC-MS specific PETN (GCMS_PETN) with sparse losses and shape residuals
│       ├── generator.py    # Synthetic GC-MS/HPLC-DAD data simulator
│       ├── train.py        # Training and alignment/evaluation script
│       └── README.md       # Chromatography-specific detailed documentation
├── tests/
│   ├── eem/                # EEM unit and import tests
│   └── chroma/             # Chromatography unit and integration tests
├── pyproject.toml          # Build configuration
├── requirements.txt        # Package dependencies
└── AGENTS.md               # Architecture and physical rules constraints
```

---

## ⚡ Training Optimizations

The chromatography subpackage supports three major performance optimizations to handle large-scale chemical datasets efficiently:
1. **Dynamic Batch Size (Option 1)**: Supports coordinate-based training segmented in mini-batches to prevent GPU/CPU Out-Of-Memory (OOM) on very large coordinate lists.
2. **Grid-Based Tensor Reconstruction (Option 2)**: Reconstructs predictions directly on the time/spectral grid using batched `einsum` and 1D Savitzky-Golay convolutions. Evaluates coordinate alignment equations only once per sample rather than per coordinate, speeding up training by **up to 250x** for derivative-based runs. Enabled by default (`batch_size=None`).
3. **Model Graph Compilation (Option 3)**: Compiles the network graph with `torch.compile` where supported for optimal hardware performance. Enabled by default (`compile_model=True`).

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DamianPiuselli/petn_parafac.git
   cd petn_parafac
   ```

2. **Activate the Conda environment or install requirements:**
   ```bash
   conda activate petn_parafac
   # Or install via pip:
   pip install -r requirements.txt
   ```

---

## 🏃 Quick Start

### Running EEM Spectroscopy Benchmarks
To train and validate on experimental/botanical datasets:
```bash
# Copenhagen Honey benchmark
PYTHONPATH=. python3 src/eem/train_honey.py

# Experimental Amino Acids benchmark
PYTHONPATH=. python3 src/eem/train_aminoacids.py

# Comparative Monte Carlo run
PYTHONPATH=. python3 src/eem/benchmark.py
```

### Running Chromatography Alignment Benchmarks
To run the alignment benchmarks on simulated and real-world datasets:
```bash
# Solidago Root Extracts benchmark (HPLC-DAD)
PYTHONPATH=. python3 src/chroma/benchmark_solidago.py

# Copenhagen Apple Wine benchmark (GC-MS)
PYTHONPATH=. python3 src/chroma/benchmark_applewine.py

# Lignin Phenols benchmark (HPLC-DAD, 13 components)
PYTHONPATH=. python3 src/chroma/benchmark_lignin.py

# UCPH Simulated GC-MS benchmark
PYTHONPATH=. python3 src/chroma/benchmark_simulated.py
```

### Running Tests
To execute all test suites (covering both EEM and chromatography):
```bash
pytest
```

