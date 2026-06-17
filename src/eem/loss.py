"""
Custom Loss Functions.
Implements the masked MSE loss to ignore Rayleigh/Raman scattering diagonals.
"""
import torch

def masked_mse_loss(y_pred, y_true, mask=None):
    """
    Computes Mean Squared Error (MSE) only over elements where mask is 1 (or True).
    Args:
        y_pred: Tensor of shape (BatchSize,) containing predicted intensities
        y_true: Tensor of shape (BatchSize,) containing ground truth intensities
        mask: Tensor of shape (BatchSize,) containing 1 for valid data, 0 for masked data.
              If None, standard MSE is computed.
    Returns:
        loss: Scalar tensor containing the masked MSE loss
    """
    squared_errors = (y_pred - y_true) ** 2
    
    if mask is None:
        return torch.mean(squared_errors)
        
    # Element-wise multiply errors with mask
    # Cast mask to match precision of squared_errors
    mask = mask.to(dtype=squared_errors.dtype)
    masked_errors = squared_errors * mask
    
    sum_mask = torch.sum(mask)
    if sum_mask == 0:
        # Avoid division by zero
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
    return torch.sum(masked_errors) / sum_mask
