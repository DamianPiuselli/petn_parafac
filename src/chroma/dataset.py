"""
Custom PyTorch Dataset and DataLoader wrappers for Chromatography PETN models.
Handles coordinate list generation (COO) for dense and sparse data formats.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ChromatographyCOODataset(Dataset):
    """
    A custom PyTorch Dataset representing chromatography data as coordinate triplets.
    Only includes coordinate triplets where signal is above a threshold for sparse data (GC-MS),
    otherwise includes the full dense grid (HPLC).
    """
    def __init__(self, X, threshold=None):
        """
        Args:
            X: Data matrix of shape (I, J, K) - NumPy array or PyTorch Tensor.
            threshold: If specified, only coordinates where intensity > threshold are loaded.
        """
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X.clone().float()
            
        I, J, K = self.X.shape
        self.num_samples = I
        self.num_time = J
        self.num_spec = K
        
        # 1. Generate active coordinate triplets
        if threshold is not None:
            # Filter sparse coordinates (Shape: [N, 3] where N is number of active points)
            self.coords = torch.nonzero(self.X > threshold)
        else:
            # Generate complete dense coordinate grid (Shape: [I * J * K, 3])
            coords_i, coords_j, coords_k = torch.meshgrid(
                torch.arange(I), torch.arange(J), torch.arange(K), indexing='ij'
            )
            self.coords = torch.stack([coords_i.flatten(), coords_j.flatten(), coords_k.flatten()], dim=1)
            
        # 2. Extract targets for the active coordinates (Shape: [N])
        self.targets = self.X[self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]]

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        # Returns coordinates: [sample_idx, time_idx, spec_idx] and target_intensity
        return self.coords[idx], self.targets[idx]


def get_chroma_dataloader(X, batch_size=4096, shuffle=True, threshold=None, pin_memory=True, num_workers=0):
    """
    Utility function to instantiate ChromatographyCOODataset and wrap it in a DataLoader.
    
    Args:
        X: Data matrix of shape (I, J, K)
        batch_size: Mini-batch size for coordinate triplets
        shuffle: Whether to shuffle coordinate batches during epoch training
        threshold: Intensity threshold for sparse mode
        pin_memory: If True, copies Tensors into CUDA pinned memory for faster GPU transfer
        num_workers: Number of subprocesses to use for data loading
        
    Returns:
        dataloader: PyTorch DataLoader instance
        dataset: ChromatographyCOODataset instance
    """
    dataset = ChromatographyCOODataset(X, threshold=threshold)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    return dataloader, dataset
