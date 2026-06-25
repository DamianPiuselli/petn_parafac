# EEM-PETN Copenhagen Honey Experiment Report

## 1. Summary of Model Performance & Diagnostics
Below are the botanical classification accuracies achieved by applying different supervised and unsupervised models to the resolved concentration scores (A) under leave-one-out cross-validation.

### Supervised Classification Accuracy (Multiclass, Autoscale)
- **PLS-DA:** 61.82%
- **SVM (Linear):** 70.91%
- **SVM (RBF):** 69.09%
- **Logistic Regression:** 71.82%
- **Random Forest:** 75.45%

### Binary Classification Accuracy (Authentic vs. Adulterated)
- **SVM (Linear):** 100.00%
- **SVM (RBF):** 100.00%
- **Logistic Regression:** 100.00%

### Attenuation Head Diagnostics
- **Average Attenuation Coefficient (Gamma):** 0.6946 (Min: 0.0000, Max: 0.9684)
- **Learned Component Molar Absorptivities (Alpha):** 3.0128e-03, 2.7265e-02, 7.2959e-02, 5.5320e-03, 7.3258e-03, 4.5796e-02

## 2. Visualization Artifacts
The following plots have been generated and saved to the EEM output folder:
1. **[Resolved Profiles](honey_resolved_profiles.png)**: Visualizes resolved excitation (B) and emission (C) loadings separated by component.
2. **[PCA Score Clusters](honey_pca_separation.png)**: Visualizes separation of botanical classes using principal components of the resolved scores.
3. **[Resolved Absorptivities](honey_resolved_absorptivities.png)**: Visualizes resolved excitation (E) molar absorptivity curves.
