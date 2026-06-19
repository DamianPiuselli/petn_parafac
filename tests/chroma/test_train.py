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

def test_chroma_early_stopping(capsys):
    # Generate a tiny dataset
    gen = ChromatographicDataGenerator(num_samples=4, num_time=20, num_spec=15, num_components=3, seed=123)
    dataset = gen.generate_dataset(noise_std=0.01, max_shift=0.02, max_stretch=0.03)
    
    # Train with tol=1.0 (unachievably high threshold for improvement) and patience=2
    # It should stop early at epoch 2
    model = train_chroma_petn(dataset, epochs=50, lr=0.01, tol=1.0, patience=2)
    
    captured = capsys.readouterr()
    assert "Early stopping at epoch    2" in captured.out


def test_chroma_training_coordinate_and_compile():
    # Generate a tiny dataset
    gen = ChromatographicDataGenerator(num_samples=4, num_time=20, num_spec=15, num_components=3, seed=123)
    dataset = gen.generate_dataset(noise_std=0.01, max_shift=0.02, max_stretch=0.03)
    
    # Train with batch_size=200 to test coordinate-based mode
    model_coord = train_chroma_petn(dataset, epochs=5, lr=0.05, warp_reg_coef=0.001, batch_size=200)
    assert model_coord is not None

    # Train with compile_model=True to test compilation path
    model_compile = train_chroma_petn(dataset, epochs=5, lr=0.05, warp_reg_coef=0.001, compile_model=True)
    assert model_compile is not None



