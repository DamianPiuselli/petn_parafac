import streamlit as st
import numpy as np
import torch
import torch.optim as optim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from src.generator import EEMGenerator
from src.model import PINNParafac
from src.loss import masked_mse_loss
from src.train import match_and_align_components

# --- Page Configurations ---
st.set_page_config(
    page_title="PINN-PARAFAC Spectroscopy Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules
st.markdown("""
<style>
    /* Sleek font and container styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .css-1d391kg {
        background-color: #161b22;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Outfit', sans-serif;
    }
    .metric-card {
        background: #1f242c;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 5px 0;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        border-bottom: 2px solid transparent;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #58a6ff;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #58a6ff;
        border-bottom-color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 PINN-PARAFAC Gray-Box Cuvette Simulator")
st.markdown("""
This interactive simulator and solver demonstrates how a **Physics-Informed Neural Network (PINN)** resolves overlapping 
chemical fluorophore loadings, compensates for non-linear **Inner Filter Effects (IFE)** using Beer-Lambert and Lakowicz physical constraints, 
and blinds out high-intensity **Rayleigh & Raman optical scattering** lines in real-time.
""")

# --- Sidebar Configuration Panels ---
st.sidebar.header("1. Data Simulation Settings")

noise_std = st.sidebar.slider("Measurement Noise (Std Dev)", min_value=0.0001, max_value=0.05, value=0.005, step=0.001, format="%.4f")
corrupt_scatter = st.sidebar.checkbox("Corrupt with Scattering (Rayleigh/Raman)", value=True)
corrupt_ife = st.sidebar.checkbox("Corrupt with Inner Filter Effect (IFE)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("2. Solver Configuration")

model_type = st.sidebar.selectbox("Model Architecture", ["PINN-PARAFAC", "Pure PARAFAC"])
lr = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.05, value=0.008, step=0.001, format="%.3f")
total_epochs = st.sidebar.number_input("Total Training Epochs", min_value=100, max_value=10000, value=2000, step=500)
epochs_per_update = st.sidebar.slider("Epochs per UI Update", min_value=10, max_value=500, value=100, step=10)

st.sidebar.markdown("---")
st.sidebar.header("3. Live Operations")

col_op1, col_op2 = st.sidebar.columns(2)
btn_generate = col_op1.button("Generate Dataset", use_container_width=True)
btn_reset = col_op2.button("Reset Solver", use_container_width=True)

btn_train_toggle = st.sidebar.empty()

# --- Initialize session state variables ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.epoch = 0
    st.session_state.is_training = False
    st.session_state.losses = []
    st.session_state.r2_a = []
    st.session_state.r2_b = []
    st.session_state.r2_c = []
    st.session_state.model = None
    st.session_state.optimizer = None
    st.session_state.dataset = None
    st.session_state.generator = None
    st.session_state.aligned_A = None
    st.session_state.aligned_B = None
    st.session_state.aligned_C = None
    st.session_state.aligned_E = None
    st.session_state.aligned_M = None
    st.session_state.pred_ex_bg = None
    st.session_state.pred_em_bg = None

# --- Generate Dataset Action ---
if btn_generate:
    st.session_state.is_training = False
    
    # Generate data
    generator = EEMGenerator(num_samples=20, num_ex=60, num_em=100, num_components=3, seed=42)
    dataset = generator.generate_dataset(noise_std=noise_std, corrupt_scatter=corrupt_scatter, corrupt_ife=corrupt_ife)
    
    st.session_state.generator = generator
    st.session_state.dataset = dataset
    
    # Background buffers
    lambda_0 = 240.0
    A_bg_ex = 0.10 * np.exp(-0.010 * (generator.ex_wavelens - lambda_0))
    A_bg_em = 0.10 * np.exp(-0.010 * (generator.em_wavelens - lambda_0))
    
    # Initialize PyTorch Model
    torch.manual_seed(42)
    np.random.seed(42)
    
    if model_type == "PINN-PARAFAC":
        model = PINNParafac(
            num_samples=generator.num_samples,
            num_ex=generator.num_ex,
            num_em=generator.num_em,
            ex_wavelens=generator.ex_wavelens,
            em_wavelens=generator.em_wavelens,
            ex_bg=A_bg_ex,
            em_bg=A_bg_em,
            num_components=3
        )
    else: # Pure PARAFAC
        model = PINNParafac(
            num_samples=generator.num_samples,
            num_ex=generator.num_ex,
            num_em=generator.num_em,
            ex_wavelens=generator.ex_wavelens,
            em_wavelens=generator.em_wavelens,
            ex_bg=None,
            em_bg=None,
            num_components=3
        )
        # Turn off IFE by freezing alpha at zero
        model.alpha.data.fill_(0.0)
        model.alpha.requires_grad = False
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    st.session_state.model = model
    st.session_state.optimizer = optimizer
    st.session_state.epoch = 0
    st.session_state.losses = []
    st.session_state.r2_a = []
    st.session_state.r2_b = []
    st.session_state.r2_c = []
    st.session_state.aligned_A = None
    st.session_state.aligned_B = None
    st.session_state.aligned_C = None
    st.session_state.aligned_E = None
    st.session_state.aligned_M = None
    st.session_state.pred_ex_bg = None
    st.session_state.pred_em_bg = None
    st.session_state.initialized = True
    
    st.rerun()

# --- Reset Solver Action ---
if btn_reset and st.session_state.initialized:
    st.session_state.is_training = False
    st.session_state.epoch = 0
    st.session_state.losses = []
    st.session_state.r2_a = []
    st.session_state.r2_b = []
    st.session_state.r2_c = []
    
    # Re-initialize weights
    st.session_state.model.reset_parameters()
    if model_type == "Pure PARAFAC":
        st.session_state.model.alpha.data.fill_(0.0)
        st.session_state.model.alpha.requires_grad = False
        
    st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=lr)
    st.session_state.aligned_A = None
    st.session_state.aligned_B = None
    st.session_state.aligned_C = None
    st.session_state.aligned_E = None
    st.session_state.aligned_M = None
    st.session_state.pred_ex_bg = None
    st.session_state.pred_em_bg = None
    st.rerun()

# --- Training Logic Loop ---
def run_training_step(num_epochs, lr_val, model_arch):
    # Dynamically update optimizer learning rate
    for param_group in st.session_state.optimizer.param_groups:
        param_group['lr'] = lr_val
        
    st.session_state.model.train()
    generator = st.session_state.generator
    dataset = st.session_state.dataset
    
    # Coordinate grids
    sample_grid, ex_grid, em_grid = np.meshgrid(
        np.arange(generator.num_samples),
        np.arange(generator.num_ex),
        np.arange(generator.num_em),
        indexing='ij'
    )
    sample_indices = torch.tensor(sample_grid.reshape(-1), dtype=torch.long)
    ex_indices = torch.tensor(ex_grid.reshape(-1), dtype=torch.long)
    em_indices = torch.tensor(em_grid.reshape(-1), dtype=torch.long)
    intensities = torch.tensor(dataset['X'].reshape(-1), dtype=torch.float32)
    
    if model_arch == "PINN-PARAFAC" and dataset['mask'] is not None:
        mask_3d = dataset['mask'][np.newaxis, :, :].repeat(generator.num_samples, axis=0)
        mask_values = torch.tensor(mask_3d.reshape(-1), dtype=torch.float32)
    else:
        mask_values = torch.ones_like(intensities)
        
    for _ in range(num_epochs):
        st.session_state.optimizer.zero_grad()
        predictions = st.session_state.model(sample_indices, ex_indices, em_indices)
        loss = masked_mse_loss(predictions, intensities, mask_values)
        loss.backward()
        st.session_state.optimizer.step()
        st.session_state.model.project_constraints()
        
    st.session_state.epoch += num_epochs
    loss_val = loss.item()
    st.session_state.losses.append(loss_val)
    
    # Extract weights & align components
    st.session_state.model.eval()
    with torch.no_grad():
        pred_A = st.session_state.model.sample_embeddings.weight.cpu().numpy()
        pred_B = st.session_state.model.ex_embeddings.weight.cpu().numpy()
        pred_C = st.session_state.model.em_embeddings.weight.cpu().numpy()
        pred_E, pred_M = st.session_state.model.get_learned_absorptivities()
        
    aligned_A, aligned_B, aligned_C, metrics = match_and_align_components(
        dataset['A'], dataset['B'], dataset['C'], pred_A, pred_B, pred_C
    )
    
    avg_r2_a = np.max([0.0, np.mean(metrics['r2_A'])])
    avg_r2_b = np.max([0.0, np.mean(metrics['r2_B'])])
    avg_r2_c = np.max([0.0, np.mean(metrics['r2_C'])])
    
    st.session_state.r2_a.append(avg_r2_a)
    st.session_state.r2_b.append(avg_r2_b)
    st.session_state.r2_c.append(avg_r2_c)
    
    # Rescale molar absorptivities using inverse scaling factors (Abs = A * E)
    pred_ind = metrics['pred_ind']
    s_factors = metrics['scale_factors']
    aligned_E = pred_E[:, pred_ind].copy()
    aligned_M = pred_M[:, pred_ind].copy()
    for r in range(generator.num_components):
        aligned_E[:, r] = aligned_E[:, r] / (s_factors[r] + 1e-8)
        aligned_M[:, r] = aligned_M[:, r] / (s_factors[r] + 1e-8)
        
    st.session_state.aligned_A = aligned_A
    st.session_state.aligned_B = aligned_B
    st.session_state.aligned_C = aligned_C
    st.session_state.aligned_E = aligned_E
    st.session_state.aligned_M = aligned_M
    
    with torch.no_grad():
        st.session_state.pred_ex_bg = st.session_state.model.ex_bg.cpu().numpy()
        st.session_state.pred_em_bg = st.session_state.model.em_bg.cpu().numpy()
        
    return loss_val, avg_r2_a, avg_r2_b, avg_r2_c


# --- Main Dashboard Render ---
if not st.session_state.initialized:
    st.info("👋 Welcome! Set your parameters and click **'Generate Dataset'** in the sidebar to create mock laboratory mixtures and initialize the solver.")
    
    # Render static educational info
    st.markdown("### How the Gray-Box Model Resolves Spectra Under Interference")
    
    col_ed1, col_ed2 = st.columns(2)
    with col_ed1:
        st.markdown("""
        **1. Unmasking Scattering Artifacts**
        * Rayleigh and Raman scattering create massive diagonal bands of light intensity.
        * Standard linear PARAFAC attempts to fit this scattering as separate chemical components, completely corrupting the resolved loadings.
        * Our PINN applies a custom **Masked Loss** ($W$) that sets backpropagation gradients to $0$ on these diagonals.
        * Blinded to the scattering bands, the model uses its rigid outer product constraints to smoothly interpolate the true signals underneath.
        """)
    with col_ed2:
        st.markdown("""
        **2. Reversing non-linear Inner Filter Effects (IFE)**
        * Highly concentrated mixtures absorb light in the cuvette, reducing fluorescence intensity non-linearly (attenuation).
        * The PINN embeds a **Cuvette Attenuation Head** inside the computational graph.
        * It learns molar absorptivities ($\alpha_r$) and computes a physical attenuation factor ($\gamma_i(j,k) = 10^{-\text{Abs}}$) dynamically.
        * The network calculates observed intensity as $\hat{I}_{obs} = I_{true} \times \gamma$, separating attenuation from the pure emission.
        """)
else:
    # Set up Start/Pause Button in Sidebar
    if st.session_state.is_training:
        if btn_train_toggle.button("Pause Solver", use_container_width=True):
            st.session_state.is_training = False
            st.rerun()
    else:
        if st.session_state.epoch < total_epochs:
            if btn_train_toggle.button("Start Solver", use_container_width=True):
                st.session_state.is_training = True
                st.rerun()
        else:
            btn_train_toggle.button("Training Complete ✅", disabled=True, use_container_width=True)

    # Trigger Training Loop
    if st.session_state.is_training:
        if st.session_state.epoch < total_epochs:
            loss_val, r2_a, r2_b, r2_c = run_training_step(epochs_per_update, lr, model_type)
            st.rerun()
        else:
            st.session_state.is_training = False
            st.success("Training fully complete!")
            st.rerun()

    # --- Upper Metric Panel ---
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    current_loss = st.session_state.losses[-1] if st.session_state.losses else 0.0
    r2_score_val = st.session_state.r2_a[-1] if st.session_state.r2_a else 0.0
    r2_ex_val = st.session_state.r2_b[-1] if st.session_state.r2_b else 0.0
    r2_em_val = st.session_state.r2_c[-1] if st.session_state.r2_c else 0.0
    
    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Epoch</div><div class="metric-val">{st.session_state.epoch} / {total_epochs}</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Masked MSE Loss</div><div class="metric-val">{current_loss:.6f}</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Scores R² (Concentration)</div><div class="metric-val">{r2_score_val:.4f}</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Excitation R² (Loadings B)</div><div class="metric-val">{r2_ex_val:.4f}</div></div>', unsafe_allow_html=True)
    with col_m5:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Emission R² (Loadings C)</div><div class="metric-val">{r2_em_val:.4f}</div></div>', unsafe_allow_html=True)

    # --- Visualisation Tabs ---
    tab_fitting, tab_heatmaps, tab_loadings, tab_absorbance = st.tabs([
        "📈 Fitting Metrics", "🗺️ EEM Heatmaps Comparison", "🧬 Resolved Loadings", "🧪 Cuvette & Absorptivities"
    ])
    
    generator = st.session_state.generator
    dataset = st.session_state.dataset

    # TAB 1: Fitting metrics (Loss, R2 history curves)
    with tab_fitting:
        st.subheader("Convergence & Alignment Curves")
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            # Loss curve
            fig_loss = go.Figure()
            epochs_x = [i * epochs_per_update for i in range(1, len(st.session_state.losses) + 1)]
            if not epochs_x:
                epochs_x = [0]
                losses_y = [0]
            else:
                losses_y = st.session_state.losses
                
            fig_loss.add_trace(go.Scatter(x=epochs_x, y=losses_y, mode='lines', name='Loss', line=dict(color='#58a6ff', width=2.5)))
            fig_loss.update_layout(
                title="Training Loss Curve (MSE)",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                yaxis_type="log",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            
        with col_f2:
            # R2 Recovery Curves
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(x=epochs_x, y=st.session_state.r2_a, mode='lines', name='Scores (A)', line=dict(color='#ff7f0e', width=2)))
            fig_r2.add_trace(go.Scatter(x=epochs_x, y=st.session_state.r2_b, mode='lines', name='Excitation (B)', line=dict(color='#2ca02c', width=2)))
            fig_r2.add_trace(go.Scatter(x=epochs_x, y=st.session_state.r2_c, mode='lines', name='Emission (C)', line=dict(color='#d62728', width=2)))
            
            fig_r2.update_layout(
                title="Ground Truth Component Recovery Profile (R²)",
                xaxis_title="Epoch",
                yaxis_title="R² Score Similarity",
                yaxis_range=[-0.05, 1.05],
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400
            )
            st.plotly_chart(fig_r2, use_container_width=True)

    # TAB 2: Heatmaps Comparison
    with tab_heatmaps:
        st.subheader("2D Excitation-Emission Matrix (EEM) Heatmap Profiles")
        
        sample_idx = st.slider("Select Sample Index to Display", min_value=0, max_value=generator.num_samples - 1, value=0)
        
        # Extract data for this sample
        X_true_sample = dataset['X_true'][sample_idx]
        X_obs_sample = dataset['X'][sample_idx]
        
        # Calculate model reconstructions
        if st.session_state.aligned_A is not None:
            # 1. Aligned clean reconstruction (unattenuated)
            pred_true_sample = np.einsum('r,jr,kr->jk', 
                                         st.session_state.aligned_A[sample_idx], 
                                         st.session_state.aligned_B, 
                                         st.session_state.aligned_C)
            
            # 2. Predicted observed reconstruction (with model's learned IFE attenuation)
            with torch.no_grad():
                # Prepare grid query
                sample_grid, ex_grid, em_grid = np.meshgrid(
                    np.array([sample_idx]),
                    np.arange(generator.num_ex),
                    np.arange(generator.num_em),
                    indexing='ij'
                )
                sample_indices_q = torch.tensor(sample_grid.reshape(-1), dtype=torch.long)
                ex_indices_q = torch.tensor(ex_grid.reshape(-1), dtype=torch.long)
                em_indices_q = torch.tensor(em_grid.reshape(-1), dtype=torch.long)
                
                pred_obs_flat = st.session_state.model(sample_indices_q, ex_indices_q, em_indices_q)
                pred_obs_sample = pred_obs_flat.reshape(generator.num_ex, generator.num_em).cpu().numpy()
        else:
            pred_true_sample = np.zeros_like(X_true_sample)
            pred_obs_sample = np.zeros_like(X_obs_sample)
            
        fig_heat = make_subplots(
            rows=2, cols=2, 
            subplot_titles=(
                "1. True Clean EEM (Chemical Target)", 
                "2. Lab Observed EEM (Corrupted with Scatter & IFE)", 
                "3. Reconstructed Observed EEM (Model Fit)", 
                "4. Recovered Clean EEM (Resolved by Model)"
            )
        )
        
        # Color scale limits
        max_val = float(np.max(X_true_sample))
        max_corrupt = float(np.max(X_obs_sample))
        
        # True Clean
        fig_heat.add_trace(go.Contour(z=X_true_sample.T, x=generator.ex_wavelens, y=generator.em_wavelens, colorscale="Viridis", zmin=0, zmax=max_val, showscale=False), row=1, col=1)
        # Observed Corrupted
        fig_heat.add_trace(go.Contour(z=X_obs_sample.T, x=generator.ex_wavelens, y=generator.em_wavelens, colorscale="Viridis", zmin=0, zmax=max_corrupt * 0.7, showscale=False), row=1, col=2)
        # Model Fit (Observed)
        fig_heat.add_trace(go.Contour(z=pred_obs_sample.T, x=generator.ex_wavelens, y=generator.em_wavelens, colorscale="Viridis", zmin=0, zmax=max_corrupt * 0.7, showscale=False), row=2, col=1)
        # Recovered Clean
        fig_heat.add_trace(go.Contour(z=pred_true_sample.T, x=generator.ex_wavelens, y=generator.em_wavelens, colorscale="Viridis", zmin=0, zmax=max_val, showscale=False), row=2, col=2)
        
        fig_heat.update_layout(
            template="plotly_dark",
            height=700,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        # Add labels
        for row in [1, 2]:
            for col in [1, 2]:
                fig_heat.update_xaxes(title_text="Excitation Wavelength (nm)", row=row, col=col)
                fig_heat.update_yaxes(title_text="Emission Wavelength (nm)", row=row, col=col)
                
        st.plotly_chart(fig_heat, use_container_width=True)

    # TAB 3: Resolved Loadings
    with tab_loadings:
        st.subheader("Component Loading Profiles Verification")
        
        if st.session_state.aligned_B is not None:
            fig_load = make_subplots(rows=1, cols=2, subplot_titles=("Excitation Loadings (B)", "Emission Loadings (C)"))
            
            colors_comp = ['#1f77b4', '#2ca02c', '#d62728'] # Blue, Green, Red
            comp_names = ["Component 1 (Phenanthrene-like)", "Component 2 (Anthracene-like)", "Component 3 (Humic-like)"]
            
            for r in range(generator.num_components):
                # Excitation True vs Resolved
                fig_load.add_trace(go.Scatter(x=generator.ex_wavelens, y=dataset['B'][:, r], mode='lines', name=f"True {comp_names[r]}", line=dict(color=colors_comp[r], width=1.5, dash='dash')), row=1, col=1)
                fig_load.add_trace(go.Scatter(x=generator.ex_wavelens, y=st.session_state.aligned_B[:, r], mode='lines', name=f"Resolved {comp_names[r]}", line=dict(color=colors_comp[r], width=2.5)), row=1, col=1)
                
                # Emission True vs Resolved
                fig_load.add_trace(go.Scatter(x=generator.em_wavelens, y=dataset['C'][:, r], mode='lines', name=f"True {comp_names[r]}", line=dict(color=colors_comp[r], width=1.5, dash='dash'), showlegend=False), row=1, col=2)
                fig_load.add_trace(go.Scatter(x=generator.em_wavelens, y=st.session_state.aligned_C[:, r], mode='lines', name=f"Resolved {comp_names[r]}", line=dict(color=colors_comp[r], width=2.5), showlegend=False), row=1, col=2)
                
            fig_load.update_layout(
                template="plotly_dark",
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Normalized Intensity"
            )
            fig_load.update_xaxes(title_text="Wavelength (nm)", row=1, col=1)
            fig_load.update_xaxes(title_text="Wavelength (nm)", row=1, col=2)
            fig_load.update_yaxes(title_text="Normalized Intensity", row=1, col=1)
            fig_load.update_yaxes(title_text="Normalized Intensity", row=1, col=2)
            
            st.plotly_chart(fig_load, use_container_width=True)
            
            # Scores (Concentration recovery)
            st.subheader("Relative Concentration Scores (A) Recovery Verification")
            fig_scores = go.Figure()
            for r in range(generator.num_components):
                fig_scores.add_trace(go.Scatter(x=dataset['A'][:, r], y=st.session_state.aligned_A[:, r], mode='markers', name=comp_names[r], marker=dict(color=colors_comp[r], size=10)))
                
            # Add identity diagonal line
            all_scores = np.concatenate([dataset['A'].flatten(), st.session_state.aligned_A.flatten()])
            min_sc, max_sc = float(np.min(all_scores)), float(np.max(all_scores))
            fig_scores.add_trace(go.Scatter(x=[min_sc, max_sc], y=[min_sc, max_sc], mode='lines', name='Ideal Recovery (y=x)', line=dict(color='#8b949e', dash='dot')))
            
            fig_scores.update_layout(
                title="True Concentration vs. Resolved Model Concentration (Scores A)",
                xaxis_title="True Prepared Score (Concentration)",
                yaxis_title="Model Resolved Aligned Score",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400
            )
            st.plotly_chart(fig_scores, use_container_width=True)
        else:
            st.info("Start solver training to display resolved loading profiles.")

    # TAB 4: Cuvette & Absorptivities
    with tab_absorbance:
        st.subheader("Physical Parameter Resolution Verification")
        
        if model_type == "Pure PARAFAC":
            st.warning("⚠️ Pure PARAFAC operates as a linear mathematical decomposition and is physically blind to the Cuvette Inner Filter Effect. No molar absorptivities or solvent background profiles are modeled.")
        elif st.session_state.aligned_E is not None:
            col_ab1, col_ab2 = st.columns(2)
            colors_comp = ['#1f77b4', '#2ca02c', '#d62728']
            comp_names = ["Component 1 (Phenanthrene-like)", "Component 2 (Anthracene-like)", "Component 3 (Humic-like)"]
            
            with col_ab1:
                st.subheader("Molar Absorptivity Scaling (E = α_r * B)")
                fig_abs = go.Figure()
                
                # Ground truth molar absorptivities
                # alpha true: [0.15, 0.10, 0.20]
                true_E = dataset['E']
                
                for r in range(generator.num_components):
                    fig_abs.add_trace(go.Scatter(x=generator.ex_wavelens, y=true_E[:, r], mode='lines', name=f"True α*B {comp_names[r]}", line=dict(color=colors_comp[r], width=1.5, dash='dash')))
                    fig_abs.add_trace(go.Scatter(x=generator.ex_wavelens, y=st.session_state.aligned_E[:, r], mode='lines', name=f"Resolved α*B {comp_names[r]}", line=dict(color=colors_comp[r], width=2.5)))
                    
                fig_abs.update_layout(
                    xaxis_title="Excitation Wavelength (nm)",
                    yaxis_title="Molar Absorptivity (L / mol / cm)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=400
                )
                st.plotly_chart(fig_abs, use_container_width=True)
                
            with col_ab2:
                st.subheader("Solvent Background Absorbance")
                fig_bg = go.Figure()
                
                # Ground truth background
                lambda_0 = 240.0
                A_bg_ex = 0.10 * np.exp(-0.010 * (generator.ex_wavelens - lambda_0))
                
                fig_bg.add_trace(go.Scatter(x=generator.ex_wavelens, y=A_bg_ex, mode='lines', name="True Excitation Solvent Abs_bg", line=dict(color='#ff7f0e', width=1.5, dash='dash')))
                fig_bg.add_trace(go.Scatter(x=generator.ex_wavelens, y=st.session_state.pred_ex_bg, mode='lines', name="Registered Excitation Solvent Abs_bg", line=dict(color='#ff7f0e', width=2.5)))
                
                fig_bg.update_layout(
                    xaxis_title="Excitation Wavelength (nm)",
                    yaxis_title="Absorbance Units",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=400
                )
                st.plotly_chart(fig_bg, use_container_width=True)
        else:
            st.info("Start solver training to display resolved physical absorptivities.")
