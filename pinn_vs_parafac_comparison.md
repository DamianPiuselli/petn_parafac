# PINN vs. Traditional PARAFAC Comparison

This document serves as a reference comparing the Physics-Informed Neural Network (PINN) approach with the traditional multi-way calibration PARAFAC implementation.

| Feature | Traditional PARAFAC | Our PyTorch PINN |
| :--- | :--- | :--- |
| **Optimization** | **Alternating Least Squares (ALS)**: Solves for one matrix (e.g., $A$) at a time while keeping the others ($B, C$) fixed, iterating sequentially. | **Simultaneous Gradient Descent**: Updates all matrices ($A, B, C$) at the same time using the `Adam` optimizer gradients. |
| **Scattering & Missing Data** | Requires complex mathematical workarounds or iterative imputation (filling in missing values) to handle scattering diagonals. | **Naturally Ignored**: We simply set the loss weight of the scattering pixels to `0` using a binary mask. The network trains on valid pixels, and the rigid trilinear core naturally interpolates the chemical signal underneath. |
| **Non-Linearities (e.g., IFE)** | **Strictly Linear**: Cannot model attenuation or self-absorption (Inner Filter Effects) without external chemical pre-processing. | **Hybrid Architecture**: We can route a parallel Dense network to predict a non-linear attenuation factor $\gamma \in [0,1]$ that scales the trilinear core output. |
| **Memory Scaling** | Processes the entire 3D tensor at once. Can suffer from memory bottlenecks on very high-resolution scans. | **Mini-batching**: Can train on subsets of coordinate triplets, scaling easily to massive datasets. |

## How Embeddings Replicate Loadings
In this framework, the loading matrices of PARAFAC are stored as weights inside PyTorch `nn.Embedding` layers:
1. **Lookup Tables**:
   * Sample embedding matrix $A$ of shape `(num_samples, R)`
   * Excitation embedding matrix $B$ of shape `(num_ex, R)`
   * Emission embedding matrix $C$ of shape `(num_em, R)`
2. **Indexing**: When we pass an index (e.g., sample index `3`), the embedding layer retrieves the 3rd row of its weight matrix.
3. **Weight Updating**: The weights in these tables *are* the spectral profiles and sample concentrations. Backpropagation calculates how to adjust these lookup tables so that their combined product reconstructs the measured fluorescence.
