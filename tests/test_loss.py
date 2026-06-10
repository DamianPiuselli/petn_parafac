"""
Tests for Custom Masked Loss function.
"""
import torch
import pytest
from src.loss import masked_mse_loss

def test_masked_loss_no_mask():
    """Verify that masked_mse_loss behaves like standard MSE when no mask is passed."""
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.1, 1.9, 3.2])
    
    loss = masked_mse_loss(y_pred, y_true, mask=None)
    expected = torch.mean((y_pred - y_true) ** 2)
    
    assert torch.isclose(loss, expected)

def test_masked_loss_with_mask():
    """Verify that elements with mask = 0 are ignored in loss calculation."""
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.1, 10.0, 3.2])  # huge error on element 1
    mask = torch.tensor([1.0, 0.0, 1.0])       # mask out element 1
    
    loss = masked_mse_loss(y_pred, y_true, mask=mask)
    
    # Expected: average error over elements 0 and 2
    expected = ((1.0 - 1.1) ** 2 + (3.0 - 3.2) ** 2) / 2.0
    
    assert torch.isclose(loss, torch.tensor(expected))

def test_masked_loss_zero_mask():
    """Verify that the loss doesn't crash (returns 0.0) if the mask is entirely zero."""
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.1, 1.9, 3.2])
    mask = torch.tensor([0.0, 0.0, 0.0])
    
    loss = masked_mse_loss(y_pred, y_true, mask=mask)
    assert loss.item() == 0.0
