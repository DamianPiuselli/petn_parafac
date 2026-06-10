"""
Smoke test to verify all standard libraries and custom modules can be imported.
"""

def test_library_imports():
    """Verify that all external libraries are installed and can be imported."""
    import torch
    import numpy
    import pandas
    import scipy
    import matplotlib
    import seaborn
    import tensorly
    
    assert torch.__version__ is not None
    assert numpy.__version__ is not None

def test_module_imports():
    """Verify that all internal project modules can be imported."""
    from src.generator import generate_synthetic_eem
    from src.model import PINNParafac
    from src.loss import masked_mse_loss
    from src.train import train_model
    from src.utils import plot_components
    
    assert generate_synthetic_eem is not None
    assert PINNParafac is not None
    assert masked_mse_loss is not None
    assert train_model is not None
    assert plot_components is not None
