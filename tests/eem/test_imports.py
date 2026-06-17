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
    from src.eem.generator import EEMGenerator
    from src.eem.model import PETNParafac
    from src.eem.loss import masked_mse_loss
    from src.eem.train import train_petn_mvp
    from src.common.utils import plot_resolved_vs_true_profiles
    
    assert EEMGenerator is not None
    assert PETNParafac is not None
    assert masked_mse_loss is not None
    assert train_petn_mvp is not None
    assert plot_resolved_vs_true_profiles is not None
