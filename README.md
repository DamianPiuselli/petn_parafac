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
* **Documentation:** Detailed guide available at [src/eem/README.md](file:///home/damianp/Proyectos/pinn_parafac/src/eem/README.md).

### 2. Chromatography with Retention Time Shifting (`src/chroma/`)
* **Physical Target:** Resolves non-trilinearity in GC-MS or HPLC-DAD datasets caused by column flow rate drift, temperature fluctuations, and injection delay shifts.
* **Architecture Highlights:**
  * **Differentiable Warping Head:** Parameterizes sample-specific time stretching ($\alpha_i$) and shifting ($\beta_i$) coefficients.
  * **Differentiable 1D Interpolation:** Natively queries a common/aligned chromatography profile embedding ($B$) at warped continuous coordinates.
  * **Mean-Centering Constraint:** Resolves translation/scaling degeneracy by forcing the learned warping parameters to average to zero, anchoring the canonical profile coordinates.

---

## 📁 Repository Structure

```
pinn_parafac/
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
│   │   └── README.md       # EEM-specific detailed documentation
│   └── chroma/             # Chromatography Subpackage
│       ├── __init__.py
│       ├── model.py        # Chroma-PETN model with warping layer
│       ├── generator.py    # Synthetic GC-MS/HPLC data simulator
│       └── train.py        # Chromatography training script
├── tests/
│   ├── eem/                # EEM unit and import tests
│   └── chroma/             # Chromatography unit and integration tests
├── pyproject.toml          # Build configuration
├── requirements.txt        # Package dependencies
└── AGENTS.md               # Architecture and physical rules constraints
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DamianPiuselli/pinn_parafac.git
   cd pinn_parafac
   ```

2. **Activate the Conda environment or install requirements:**
   ```bash
   conda activate pinn_parafac
   # Or install via pip:
   pip install -r requirements.txt
   ```

---

## 🏃 Quick Start

### Running the EEM Benchmark
To train the PETN-PARAFAC model on synthetic EEM data and compare it with traditional PARAFAC:
```bash
python -m src.eem.benchmark
```

### Running the Chromatography Demo
To train the Chroma-PETN model and verify retention time alignment recovery:
```bash
python -m src.chroma.train
```

### Running Tests
To execute all test suites (covering both EEM and chromatography):
```bash
pytest
```
