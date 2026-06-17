"""
Integration tests for Chroma-PETN Training Loop.
"""
from src.chroma.generator import ChromatographicDataGenerator
from src.chroma.train import train_chroma_petn, evaluate_chroma_alignment

def test_chroma_training_integration():
    # Generate a tiny dataset for quick integration test
    gen = ChromatographicDataGenerator(num_samples=4, num_time=20, num_spec=15, num_components=3, seed=123)
    dataset = gen.generate_dataset(noise_std=0.01, max_shift=0.02, max_stretch=0.03)
    
    # Train for a very small number of epochs
    model = train_chroma_petn(dataset, epochs=10, lr=0.05, warp_reg_coef=0.001)
    
    # Verify training completed and evaluate returns metrics
    metrics = evaluate_chroma_alignment(model, dataset)
    
    assert 'b_similarities' in metrics
    assert 'c_similarities' in metrics
    assert 'a_similarities' in metrics
    assert len(metrics['b_similarities']) == 3
    assert len(metrics['c_similarities']) == 3
    assert len(metrics['a_similarities']) == 3
    assert metrics['shift_correlation'] is not None
    assert metrics['mean_shift_error'] >= 0.0
