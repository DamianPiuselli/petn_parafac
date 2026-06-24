# EEM Spectroscopy Datasets Context Documentation

This directory contains the Excitation-Emission Matrix (EEM) spectroscopy datasets downloaded and used for testing and validation of the **EEM-PETN** (Physics-Embedded Tensor Network) model.

---

## 1. Dataset Inventory & Summary

| Dataset Name | Local Path | Format & File Count | Size on Disk | Chemical & Instrumental Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Aromatic Amino Acids Mixture** | `data/eem/aminoacids/` | 1 `.mat` workspace | ~274 KB | Mixtures of Tryptophan (Trp), Tyrosine (Tyr), and Phenylalanine (Phe) with severe Rayleigh/Raman scattering diagonals. |
| **Copenhagen Honey Botanical Origins** | `data/eem/honey/` | 1 `.mat` file | ~1.4 MB | Organic fluorophores in 110 honey samples across 5 botanical classes (used to identify origin and adulteration). |

---

## 2. Aromatic Amino Acids Dataset

### A. Chemical & Physical Profile
The dataset contains fluorescence measurements of the three naturally fluorescent aromatic amino acids in aqueous mixtures:
1. **Tryptophan (Trp):** Excitation peak ~280 nm, Emission peak ~350 nm.
2. **Tyrosine (Tyr):** Excitation peak ~275 nm, Emission peak ~303 nm.
3. **Phenylalanine (Phe):** Excitation peak ~260 nm, Emission peak ~282 nm.

*   **Non-trilinear source:** High-intensity Rayleigh and Raman scattering bands that corrupt the chemical signatures along the diagonals where $\lambda_{em} \approx \lambda_{ex}$ and $\lambda_{em} \approx 2\lambda_{ex}$.
*   **Dimensions:** 5 mixture runs, 61 excitation wavelengths (240 nm to 300 nm, 1 nm step), 201 emission wavelengths (250 nm to 450 nm, 1 nm step).

### B. Layout & Schema
The variables inside the MAT file `amino.mat`:
*   `X`: Flattened matrix of dimensions `(5, 12261)`. When reshaped in column-major (Fortran) order, it yields a 3D tensor of shape `(5, 61, 201)`.
*   `y`: Reference concentrations of the 3 amino acids in each of the 5 samples.
*   `ExAx`: 1D array of 61 excitation wavelengths.
*   `EmAx`: 1D array of 201 emission wavelengths.

### C. Python Loading Recipe
```python
import os
import scipy.io
import numpy as np

def load_aminoacids_dataset(data_dir):
    """
    Loads and reshapes the Amino Acids EEM dataset.
    """
    mat_path = os.path.join(data_dir, "amino.mat")
    mat = scipy.io.loadmat(mat_path)
    
    ex_wavelens = mat['ExAx'].squeeze()
    em_wavelens = mat['EmAx'].squeeze()
    X_flat = mat['X']  # (5, 12261)
    y_true = mat['y']  # (5, 3)
    
    num_samples = X_flat.shape[0]
    num_ex = len(ex_wavelens)
    num_em = len(em_wavelens)
    
    # Reshape from Fortran column-major order to C-order
    X = X_flat.reshape(num_samples, num_ex, num_em)
    
    return X, y_true, ex_wavelens, em_wavelens
```

---

## 3. Copenhagen Honey Dataset

### A. Botanical Origin & Physical Profile
Organic fluorophores (phenolics, flavonoids, amino acids) are used to characterize the botanical origin of honey samples.
*   **Non-trilinear source:** Severe Inner Filter Effect (IFE) due to sample absorbance of excitation/emission light in concentrated cuvette solutions, which attenuates the expected trilinear signal.
*   **Dimensions:** 110 samples, 52 excitation wavelengths (260 nm to 410 nm, 3 nm step), 741 emission wavelengths (310 nm to 680 nm, 0.5 nm step). Typically downsampled by 4 on emission to $186$ channels.

### B. Layout & Schema
Inside `HoneyEEM.mat`, a nested struct named `X` stores:
*   `dataset['data']`: 3D raw data array of dimensions `(110, 741, 52)`.
*   `dataset['axisscale'][1, 0]`: 1D emission wavelengths grid.
*   `dataset['axisscale'][2, 0]`: 1D excitation wavelengths grid.
*   `dataset['class']`: Botanical origin class labels (grouped into Acacia, Linden, Sunflower, Meadow, and Adulterated Fake Honey).

### C. Python Loading Recipe
```python
import os
import scipy.io
import numpy as np

def load_honey_dataset(data_dir):
    """
    Loads and downsamples the Honey EEM dataset.
    """
    mat_path = os.path.join(data_dir, "HoneyEEM.mat")
    mat = scipy.io.loadmat(mat_path)
    
    dataset = mat['X'][0, 0]
    X_raw = dataset['data']  # (110, 741, 52)
    em_wavelens = dataset['axisscale'][1, 0].squeeze()
    ex_wavelens = dataset['axisscale'][2, 0].squeeze()
    class_ids = dataset['class'].squeeze().astype(int)
    
    # Downsample emission grid by 4 to speed up training
    X = X_raw[:, ::4, :]
    em_wavelens = em_wavelens[::4]
    
    return X, ex_wavelens, em_wavelens, class_ids
```
