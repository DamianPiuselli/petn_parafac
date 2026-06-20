import streamlit as st
import numpy as np
import torch
import torch.optim as optim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from src.eem.generator import EEMGenerator
from src.eem.model import PETNParafac
from src.eem.loss import masked_mse_loss
from src.eem.train import match_and_align_components

from src.chroma.generator import ChromatographicDataGenerator
from src.chroma import HPLC_PETN


# --- Helper functions ---
def calculate_cosine_similarity(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    return np.max([np.dot(v1_norm, v2_norm), np.dot(v1_norm, -v2_norm)])

def match_and_align_chroma_components(A_true, B_true, C_true, A_pred, B_pred, C_pred):
    R = A_true.shape[1]
    sim_matrix = np.zeros((R, R))
    for r_pred in range(R):
        for r_true in range(R):
            sim_matrix[r_pred, r_true] = calculate_cosine_similarity(C_pred[:, r_pred], C_true[:, r_true])
            
    perm = []
    used = set()
    for r in range(R):
        best_sim = -1.0
        best_idx = 0
        for r_true in range(R):
            if r_true in used:
                continue
            sim = sim_matrix[r, r_true]
            if sim > best_sim:
                best_sim = sim
                best_idx = r_true
        perm.append(best_idx)
        used.add(best_idx)
        
    A_pred_ordered = np.zeros_like(A_pred)
    B_pred_ordered = np.zeros_like(B_pred)
    C_pred_ordered = np.zeros_like(C_pred)
    
    for r in range(R):
        true_idx = perm[r]
        A_pred_ordered[:, true_idx] = A_pred[:, r]
        B_pred_ordered[:, true_idx] = B_pred[:, r]
        C_pred_ordered[:, true_idx] = C_pred[:, r]
        
    # Scale ambiguity resolution
    for r in range(R):
        norm_b = np.linalg.norm(B_pred_ordered[:, r])
        norm_c = np.linalg.norm(C_pred_ordered[:, r])
        if norm_b > 0 and norm_c > 0:
            B_pred_ordered[:, r] /= norm_b
            C_pred_ordered[:, r] /= norm_c
            A_pred_ordered[:, r] *= (norm_b * norm_c)
            
    # Calculate recovery R^2 similarity
    b_sims = [calculate_cosine_similarity(B_pred_ordered[:, r], B_true[:, r]) for r in range(R)]
    c_sims = [calculate_cosine_similarity(C_pred_ordered[:, r], C_true[:, r]) for r in range(R)]
    a_sims = [calculate_cosine_similarity(A_pred_ordered[:, r], A_true[:, r]) for r in range(R)]
    
    return A_pred_ordered, B_pred_ordered, C_pred_ordered, {
        'r2_A': [s**2 for s in a_sims],
        'r2_B': [s**2 for s in b_sims],
        'r2_C': [s**2 for s in c_sims]
    }

# --- Page Configurations ---
st.set_page_config(
    page_title="PETN Gray-Box Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Plotly Styling Config ---
PLOTLY_THEME_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Plus Jakarta Sans, Outfit, sans-serif",
        color="#c9d1d9"
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.1)",
        zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color="#8b949e")
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.1)",
        zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color="#8b949e")
    ),
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(
        bgcolor="rgba(13,17,23,0.6)",
        bordercolor="rgba(255,255,255,0.05)",
        borderwidth=1
    )
)

# --- Inject Premium Custom CSS ---
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Body and Font Override */
    .stApp {
        background-color: #0b0f19;
        color: #c9d1d9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Card styling */
    .header-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8) 0%, rgba(13, 17, 23, 0.9) 100%);
        border: 1px solid rgba(88, 166, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .header-badge {
        display: inline-block;
        background: rgba(88, 166, 255, 0.12);
        color: #58a6ff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        border: 1px solid rgba(88, 166, 255, 0.2);
    }
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #58a6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #8b949e;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Status Bar styling */
    .status-bar {
        display: flex;
        align-items: center;
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 12px 18px;
        border-radius: 10px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 12px;
        display: inline-block;
    }
    .status-idle {
        background-color: #6e7681;
        box-shadow: 0 0 10px #6e7681;
    }
    .status-running {
        background-color: #58a6ff;
        box-shadow: 0 0 12px #58a6ff;
        animation: status-pulse 1.5s infinite;
    }
    .status-paused {
        background-color: #ff7f0e;
        box-shadow: 0 0 10px #ff7f0e;
    }
    .status-complete {
        background-color: #2ca02c;
        box-shadow: 0 0 12px #2ca02c;
    }
    .status-text {
        font-size: 0.85rem;
        font-weight: 700;
        color: #e5e7eb;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    @keyframes status-pulse {
        0% { transform: scale(0.95); opacity: 0.7; }
        50% { transform: scale(1.15); opacity: 1; box-shadow: 0 0 16px #58a6ff; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }
    
    /* Control Panel Section */
    .control-panel-header {
        font-size: 0.9rem;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Metrics Grid & Card styling */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 16px;
        margin: 20px 0;
    }
    @media (max-width: 1200px) {
        .metrics-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    @media (max-width: 768px) {
        .metrics-grid {
            grid-template-columns: repeat(1, 1fr);
        }
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: left;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
    }
    
    /* Metric specific color bars */
    .epoch-card::before { background: #58a6ff; }
    .loss-card::before { background: #8b949e; }
    .scores-card::before { background: #ff7f0e; }
    .ex-card::before { background: #2ca02c; }
    .em-card::before { background: #d62728; }
    
    .epoch-card:hover { border-color: rgba(88, 166, 255, 0.3); }
    .loss-card:hover { border-color: rgba(139, 148, 158, 0.3); }
    .scores-card:hover { border-color: rgba(255, 127, 14, 0.3); }
    .ex-card:hover { border-color: rgba(44, 160, 44, 0.3); }
    .em-card:hover { border-color: rgba(214, 39, 40, 0.3); }

    .metric-lbl {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 4px;
        font-family: 'Outfit', sans-serif;
    }
    .metric-total {
        font-size: 0.95rem;
        color: #8b949e;
        font-weight: 400;
    }
    .metric-trend {
        font-size: 0.72rem;
        color: #6e7681;
    }
    
    /* Interactive Streamlit Button Styling Overrides */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #1f2937, #111827) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #e5e7eb !important;
        border-radius: 8px !important;
        padding: 10px 18px !important;
        font-weight: 600 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15) !important;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #58a6ff !important;
        color: #58a6ff !important;
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(0px) !important;
    }
    
    /* Primary buttons (e.g. Start/Pause Solver) */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    }
    
    /* Onboarding / Educational Layout styling */
    .welcome-card {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.08), rgba(30, 41, 59, 0.35));
        border: 1px solid rgba(88, 166, 255, 0.18);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    .welcome-card h2 {
        margin-top: 0;
        color: #58a6ff !important;
        font-family: 'Outfit', sans-serif;
    }
    .welcome-card p {
        color: #c9d1d9;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    .interference-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 10px;
    }
    @media (max-width: 768px) {
        .interference-grid {
            grid-template-columns: 1fr;
        }
    }
    .interference-card {
        background: rgba(22, 27, 34, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
    }
    .interference-card:hover {
        transform: translateY(-3px);
        border-color: rgba(88, 166, 255, 0.15);
        box-shadow: 0 12px 40px rgba(88, 166, 255, 0.05);
    }
    .card-icon {
        font-size: 1.8rem;
        margin-bottom: 12px;
    }
    .interference-card h3 {
        margin-top: 0;
        font-size: 1.25rem;
        color: #58a6ff !important;
        font-family: 'Outfit', sans-serif;
    }
    .interference-card p {
        font-size: 0.9rem;
        line-height: 1.5;
        color: #c9d1d9;
        margin-bottom: 12px;
    }
    .card-tag {
        background: rgba(88, 166, 255, 0.12);
        color: #58a6ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        align-self: flex-start;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: 1px solid rgba(88, 166, 255, 0.15);
    }
    .card-action {
        font-size: 0.85rem !important;
        color: #8b949e !important;
        line-height: 1.4 !important;
        margin-bottom: 0 !important;
    }

    /* Warning card for Pure PARAFAC */
    .warning-card {
        display: flex;
        gap: 16px;
        background: rgba(255, 127, 14, 0.08);
        border: 1px solid rgba(255, 127, 14, 0.22);
        border-radius: 12px;
        padding: 20px;
        margin-top: 10px;
    }
    .warning-icon {
        font-size: 1.8rem;
    }
    .warning-content h3 {
        margin-top: 0;
        color: #ff7f0e !important;
        font-family: 'Outfit', sans-serif;
    }
    .warning-content p {
        margin: 0;
        color: #e5e7eb;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Streamlit Tabs Custom Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(22, 27, 34, 0.5) !important;
        border-radius: 10px !important;
        padding: 5px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        gap: 6px !important;
        margin-bottom: 20px !important;
    }
    .stTabs button[data-baseweb="tab"] {
        border-radius: 8px !important;
        background-color: transparent !important;
        border: none !important;
        color: #8b949e !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.25s ease !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
    }
    .stTabs button[data-baseweb="tab"]:hover {
        color: #58a6ff !important;
        background-color: rgba(88, 166, 255, 0.06) !important;
    }
    .stTabs button[data-baseweb="tab"][aria-selected="true"] {
        color: #0b0f19 !important;
        background: linear-gradient(135deg, #58a6ff, #1f77b4) !important;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.25) !important;
    }

    /* Sidebar Element Adjustments */
    [data-testid="stSidebar"] {
        background-color: #0d111b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #58a6ff !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        margin-top: 24px !important;
        margin-bottom: 12px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding-bottom: 6px !important;
        font-family: 'Outfit', sans-serif !important;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {
        background-color: rgba(22, 27, 34, 0.3);
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"]:hover {
        border-color: rgba(88, 166, 255, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# --- Top-Level Sidebar Track Selection ---
st.sidebar.markdown("<h2>📁 Development Track</h2>", unsafe_allow_html=True)
track_mode = st.sidebar.selectbox("Select Track Mode", ["EEM Spectroscopy", "Chromatography Warping"])

# --- Sidebar Configuration Panels ---
if track_mode == "EEM Spectroscopy":
    st.sidebar.markdown("<h2>1. Data Simulation</h2>", unsafe_allow_html=True)
    noise_std = st.sidebar.slider("Measurement Noise (Std Dev)", min_value=0.0001, max_value=0.05, value=0.005, step=0.001, format="%.4f")
    corrupt_scatter = st.sidebar.checkbox("Corrupt with Scattering (Rayleigh/Raman)", value=True)
    corrupt_ife = st.sidebar.checkbox("Corrupt with Inner Filter Effect (IFE)", value=True)
    
    st.sidebar.markdown("<h2>2. Solver Configuration</h2>", unsafe_allow_html=True)
    model_type = st.sidebar.selectbox("Model Architecture", ["PETN-PARAFAC", "Pure PARAFAC"])
    lr = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.05, value=0.008, step=0.001, format="%.3f")
    total_epochs = st.sidebar.number_input("Total Training Epochs", min_value=100, max_value=10000, value=2000, step=500)
    epochs_per_update = st.sidebar.slider("Epochs per UI Update", min_value=10, max_value=500, value=100, step=10)
    ui_delay = st.sidebar.slider("UI Update Delay (seconds)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
else:
    st.sidebar.markdown("<h2>1. Data Simulation</h2>", unsafe_allow_html=True)
    chroma_noise_std = st.sidebar.slider("Measurement Noise (Std Dev)", min_value=0.0001, max_value=0.05, value=0.01, step=0.001, format="%.4f")
    chroma_max_shift = st.sidebar.slider("Max Shift (Translation)", min_value=0.0, max_value=0.15, value=0.05, step=0.01, format="%.2f")
    chroma_max_stretch = st.sidebar.slider("Max Stretch (Scaling)", min_value=0.0, max_value=0.20, value=0.08, step=0.01, format="%.2f")
    chroma_sim_warp_type = st.sidebar.selectbox("Simulation Warp Type", ["linear", "quadratic", "spline"])
    
    st.sidebar.markdown("<h2>2. Solver Configuration</h2>", unsafe_allow_html=True)
    chroma_model_arch = st.sidebar.selectbox("Model Architecture", ["Chroma-PETN", "Pure PARAFAC"])
    chroma_warp_type = "linear"
    if chroma_model_arch == "Chroma-PETN":
        chroma_warp_type = st.sidebar.selectbox("Solver Warp Model Type", ["linear", "quadratic", "spline"])
    chroma_lr = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.05, value=0.015, step=0.001, format="%.3f")
    total_epochs = st.sidebar.number_input("Total Training Epochs", min_value=100, max_value=5000, value=600, step=100)
    epochs_per_update = st.sidebar.slider("Epochs per UI Update", min_value=5, max_value=100, value=25, step=5)
    ui_delay = st.sidebar.slider("UI Update Delay (seconds)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
    
    st.sidebar.markdown("<h2>3. Derivative Configuration</h2>", unsafe_allow_html=True)
    chroma_enable_deriv = st.sidebar.checkbox("Enable Savitzky-Golay Derivative", value=False)
    chroma_deriv_order = 1
    chroma_sg_window_size = 11
    if chroma_enable_deriv:
        chroma_deriv_order = st.sidebar.selectbox("Derivative Order", [1, 2], index=1)
        chroma_sg_window_size = st.sidebar.selectbox("SG Window Size", [5, 7, 9, 11, 13, 15, 17], index=3)
        
    chroma_warp_reg_coef = st.sidebar.slider("Warp Regularization Coef", min_value=0.0, max_value=0.01, value=0.001, step=0.0005, format="%.4f")

# --- Initialize Session State Variables ---
if 'track_mode' not in st.session_state or st.session_state.track_mode != track_mode:
    st.session_state.track_mode = track_mode
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

# --- Rebuild model on-the-fly if configuration changes ---
if track_mode == "EEM Spectroscopy":
    if 'model_type_in_state' in st.session_state and st.session_state.initialized and st.session_state.model_type_in_state != model_type:
        st.session_state.model_type_in_state = model_type
        generator = st.session_state.generator
        dataset = st.session_state.dataset
        
        lambda_0 = 240.0
        A_bg_ex = 0.10 * np.exp(-0.010 * (generator.ex_wavelens - lambda_0))
        A_bg_em = 0.10 * np.exp(-0.010 * (generator.em_wavelens - lambda_0))
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        if model_type == "PETN-PARAFAC":
            st.session_state.model = PETNParafac(
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
            st.session_state.model = PETNParafac(
                num_samples=generator.num_samples,
                num_ex=generator.num_ex,
                num_em=generator.num_em,
                ex_wavelens=generator.ex_wavelens,
                em_wavelens=generator.em_wavelens,
                ex_bg=None,
                em_bg=None,
                num_components=3
            )
            st.session_state.model.alpha.data.fill_(0.0)
            st.session_state.model.alpha.requires_grad = False
            
        st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=lr)
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
        st.sidebar.info(f"🔄 Rebuilt solver for {model_type}")
else:
    deriv_order = chroma_deriv_order if chroma_enable_deriv else 0
    sg_w = chroma_sg_window_size if chroma_enable_deriv else 11
    current_warp = chroma_warp_type if chroma_model_arch == "Chroma-PETN" else "linear"
    
    if st.session_state.initialized and (
        st.session_state.get('chroma_model_arch_in_state') != chroma_model_arch or
        st.session_state.get('chroma_warp_type_in_state') != current_warp or
        st.session_state.get('chroma_deriv_in_state') != deriv_order or
        st.session_state.get('chroma_sg_w_in_state') != sg_w or
        st.session_state.get('chroma_lr_in_state') != chroma_lr
    ):
        st.session_state.chroma_model_arch_in_state = chroma_model_arch
        st.session_state.chroma_warp_type_in_state = current_warp
        st.session_state.chroma_deriv_in_state = deriv_order
        st.session_state.chroma_sg_w_in_state = sg_w
        st.session_state.chroma_lr_in_state = chroma_lr
        
        generator = st.session_state.generator
        st.session_state.model = HPLC_PETN(
            num_samples=generator.num_samples,
            num_time=generator.num_time,
            num_spec=generator.num_spec,
            num_components=3,
            warp_type=current_warp,
            num_segments=4,
            derivative_order=deriv_order,
            sg_window_size=sg_w
        )
        if chroma_model_arch == "Pure PARAFAC":
            st.session_state.model.alpha_params.data.fill_(0.0)
            st.session_state.model.beta_params.data.fill_(0.0)
            st.session_state.model.alpha_params.requires_grad = False
            st.session_state.model.beta_params.requires_grad = False

            
        st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=chroma_lr)
        st.session_state.epoch = 0
        st.session_state.losses = []
        st.session_state.r2_a = []
        st.session_state.r2_b = []
        st.session_state.r2_c = []
        st.session_state.aligned_A = None
        st.session_state.aligned_B = None
        st.session_state.aligned_C = None
        st.sidebar.info(f"🔄 Rebuilt solver for {chroma_model_arch}")

# --- Helper functions to render custom UI structures ---
def render_metrics_grid(epoch, total_epochs, loss, r2_score, r2_b, r2_c, mode="EEM"):
    loss_text = f"{loss:.6f}" if loss > 0 else "0.000000"
    lbl_b = "Excitation R² (Loadings B)" if mode == "EEM" else "Retention Time R² (Loadings B)"
    lbl_c = "Emission R² (Loadings C)" if mode == "EEM" else "Spectral R² (Loadings C)"
    
    html_content = f"""
    <div class="metrics-grid">
        <div class="metric-card epoch-card">
            <div class="metric-lbl">Epoch</div>
            <div class="metric-val" style="color: #58a6ff;">{epoch} <span class="metric-total">/ {total_epochs}</span></div>
            <div class="metric-trend">Solver iterations</div>
        </div>
        <div class="metric-card loss-card">
            <div class="metric-lbl">Masked MSE Loss</div>
            <div class="metric-val" style="color: #c9d1d9;">{loss_text}</div>
            <div class="metric-trend">Error function target</div>
        </div>
        <div class="metric-card scores-card">
            <div class="metric-lbl">Scores R² (Concentration)</div>
            <div class="metric-val" style="color: #ff7f0e;">{r2_score:.4f}</div>
            <div class="metric-trend">Score vector accuracy</div>
        </div>
        <div class="metric-card ex-card">
            <div class="metric-lbl">{lbl_b}</div>
            <div class="metric-val" style="color: #2ca02c;">{r2_b:.4f}</div>
            <div class="metric-trend">Loading B profile shape</div>
        </div>
        <div class="metric-card em-card">
            <div class="metric-lbl">{lbl_c}</div>
            <div class="metric-val" style="color: #d62728;">{r2_c:.4f}</div>
            <div class="metric-trend">Loading C profile shape</div>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def render_status_bar(mode="EEM"):
    max_ep = total_epochs
    if not st.session_state.initialized:
        status_html = """
        <div class="status-bar">
            <span class="status-dot status-idle"></span>
            <span class="status-text">UNINITIALIZED — Configure parameters & Generate a dataset to begin</span>
        </div>
        """
    elif st.session_state.is_training:
        status_html = f"""
        <div class="status-bar">
            <span class="status-dot status-running"></span>
            <span class="status-text">TRAINING — Epoch {st.session_state.epoch} / {max_ep}</span>
        </div>
        """
    elif st.session_state.epoch >= max_ep:
        status_html = """
        <div class="status-bar">
            <span class="status-dot status-complete"></span>
            <span class="status-text">COMPLETED — Optimization converged successfully</span>
        </div>
        """
    else:
        status_html = f"""
        <div class="status-bar">
            <span class="status-dot status-paused"></span>
            <span class="status-text">PAUSED — Solver stopped at epoch {st.session_state.epoch} / {max_ep}</span>
        </div>
        """
    st.markdown(status_html, unsafe_allow_html=True)

# --- Actions Triggers ---
def action_generate_dataset():
    st.session_state.is_training = False
    
    if track_mode == "EEM Spectroscopy":
        generator = EEMGenerator(num_samples=20, num_ex=60, num_em=100, num_components=3, seed=42)
        dataset = generator.generate_dataset(noise_std=noise_std, corrupt_scatter=corrupt_scatter, corrupt_ife=corrupt_ife)
        
        st.session_state.generator = generator
        st.session_state.dataset = dataset
        
        lambda_0 = 240.0
        A_bg_ex = 0.10 * np.exp(-0.010 * (generator.ex_wavelens - lambda_0))
        A_bg_em = 0.10 * np.exp(-0.010 * (generator.em_wavelens - lambda_0))
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        if model_type == "PETN-PARAFAC":
            model = PETNParafac(
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
            model = PETNParafac(
                num_samples=generator.num_samples,
                num_ex=generator.num_ex,
                num_em=generator.num_em,
                ex_wavelens=generator.ex_wavelens,
                em_wavelens=generator.em_wavelens,
                ex_bg=None,
                em_bg=None,
                num_components=3
            )
            model.alpha.data.fill_(0.0)
            model.alpha.requires_grad = False
            
        optimizer = optim.Adam(model.parameters(), lr=lr)
        st.session_state.model_type_in_state = model_type
    else:
        generator = ChromatographicDataGenerator(
            num_samples=15, 
            num_time=80, 
            num_spec=60, 
            num_components=3, 
            seed=42
        )
        dataset = generator.generate_dataset(
            noise_std=chroma_noise_std, 
            max_shift=chroma_max_shift, 
            max_stretch=chroma_max_stretch, 
            warp_type=chroma_sim_warp_type
        )
        
        st.session_state.generator = generator
        st.session_state.dataset = dataset
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        deriv_order = chroma_deriv_order if chroma_enable_deriv else 0
        sg_w = chroma_sg_window_size if chroma_enable_deriv else 11
        current_warp = chroma_warp_type if chroma_model_arch == "Chroma-PETN" else "linear"
        
        model = HPLC_PETN(
            num_samples=generator.num_samples,
            num_time=generator.num_time,
            num_spec=generator.num_spec,
            num_components=3,
            warp_type=current_warp,
            num_segments=4,
            derivative_order=deriv_order,
            sg_window_size=sg_w
        )
        
        if chroma_model_arch == "Pure PARAFAC":
            model.alpha_params.data.fill_(0.0)
            model.beta_params.data.fill_(0.0)
            model.alpha_params.requires_grad = False
            model.beta_params.requires_grad = False

            
        optimizer = optim.Adam(model.parameters(), lr=chroma_lr)
        st.session_state.chroma_model_arch_in_state = chroma_model_arch
        st.session_state.chroma_warp_type_in_state = current_warp
        st.session_state.chroma_deriv_in_state = deriv_order
        st.session_state.chroma_sg_w_in_state = sg_w
        st.session_state.chroma_lr_in_state = chroma_lr
        
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

def action_reset_solver():
    st.session_state.is_training = False
    st.session_state.epoch = 0
    st.session_state.losses = []
    st.session_state.r2_a = []
    st.session_state.r2_b = []
    st.session_state.r2_c = []
    
    st.session_state.model.reset_parameters()
    
    if track_mode == "EEM Spectroscopy":
        if model_type == "Pure PARAFAC":
            st.session_state.model.alpha.data.fill_(0.0)
            st.session_state.model.alpha.requires_grad = False
        st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=lr)
    else:
        if chroma_model_arch == "Pure PARAFAC":
            st.session_state.model.alpha_params.data.fill_(0.0)
            st.session_state.model.beta_params.data.fill_(0.0)
            st.session_state.model.alpha_params.requires_grad = False
            st.session_state.model.beta_params.requires_grad = False

        st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=chroma_lr)
        
    st.session_state.aligned_A = None
    st.session_state.aligned_B = None
    st.session_state.aligned_C = None
    st.session_state.aligned_E = None
    st.session_state.aligned_M = None
    st.session_state.pred_ex_bg = None
    st.session_state.pred_em_bg = None

# --- Training Logic Step ---
def run_training_step(num_epochs, lr_val, model_arch):
    for param_group in st.session_state.optimizer.param_groups:
        param_group['lr'] = lr_val
        
    st.session_state.model.train()
    generator = st.session_state.generator
    dataset = st.session_state.dataset
    
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
    
    if dataset['mask'] is not None:
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
    
    pred_ind = metrics['pred_ind']
    s_factors = metrics['scale_factors']
    aligned_E = pred_E[:, pred_ind].copy()
    aligned_M = pred_M[:, pred_ind].copy()
    
    max_bs = [np.max(pred_B[:, r]) for r in range(generator.num_components)]
    max_cs = [np.max(pred_C[:, r]) for r in range(generator.num_components)]
    
    for r in range(generator.num_components):
        idx = pred_ind[r]
        max_b = max_bs[idx] if max_bs[idx] > 1e-8 else 1.0
        max_c = max_cs[idx] if max_cs[idx] > 1e-8 else 1.0
        scale_val = s_factors[r] * max_b * max_c
        aligned_E[:, r] = aligned_E[:, r] / (scale_val + 1e-8)
        aligned_M[:, r] = aligned_M[:, r] / (scale_val + 1e-8)
        
    st.session_state.aligned_A = aligned_A
    st.session_state.aligned_B = aligned_B
    st.session_state.aligned_C = aligned_C
    st.session_state.aligned_E = aligned_E
    st.session_state.aligned_M = aligned_M
    
    with torch.no_grad():
        st.session_state.pred_ex_bg = st.session_state.model.ex_bg.cpu().numpy()
        st.session_state.pred_em_bg = st.session_state.model.em_bg.cpu().numpy()
        
    return loss_val, avg_r2_a, avg_r2_b, avg_r2_c

def run_chroma_training_step(num_epochs, lr_val, model_arch):
    for param_group in st.session_state.optimizer.param_groups:
        param_group['lr'] = lr_val
        
    st.session_state.model.train()
    generator = st.session_state.generator
    dataset = st.session_state.dataset
    model = st.session_state.model
    
    I, J, K = generator.num_samples, generator.num_time, generator.num_spec
    
    coords_i, coords_j, coords_k = np.meshgrid(
        np.arange(I), np.arange(J), np.arange(K), indexing='ij'
    )
    coords_i_flat = torch.tensor(coords_i.flatten(), dtype=torch.long)
    coords_j_flat = torch.tensor(coords_j.flatten(), dtype=torch.long)
    coords_k_flat = torch.tensor(coords_k.flatten(), dtype=torch.long)
    
    if model.derivative_order > 0:
        from scipy.signal import savgol_filter
        X_deriv = savgol_filter(dataset['X'], window_length=model.sg_window_size, polyorder=2, deriv=model.derivative_order, axis=1)
        y_target = torch.tensor(X_deriv, dtype=torch.float32)[coords_i_flat, coords_j_flat, coords_k_flat]
    else:
        y_target = torch.tensor(dataset['X'], dtype=torch.float32)[coords_i_flat, coords_j_flat, coords_k_flat]
        
    for _ in range(num_epochs):
        st.session_state.optimizer.zero_grad()
        y_pred = model(coords_i_flat, coords_j_flat, coords_k_flat)
        loss_mse = torch.nn.functional.mse_loss(y_pred, y_target)
        
        if model.warp_type == 'linear':
            loss_warp_reg = chroma_warp_reg_coef * (torch.mean(model.alpha_params**2) + torch.mean(model.beta_params**2))
        elif model.warp_type == 'quadratic':
            loss_warp_reg = chroma_warp_reg_coef * (torch.mean(model.alpha_params**2) + torch.mean(model.beta_params**2) + torch.mean(model.gamma_params**2))
        elif model.warp_type == 'spline':
            loss_warp_reg = chroma_warp_reg_coef * (torch.mean(model.beta_params**2) + torch.mean(model.log_increments_params**2))
        else:
            loss_warp_reg = torch.tensor(0.0)

            
        loss = loss_mse + loss_warp_reg
        loss.backward()
        st.session_state.optimizer.step()
        model.project_constraints()
        
    st.session_state.epoch += num_epochs
    loss_val = loss_mse.item()
    st.session_state.losses.append(loss_val)
    
    model.eval()
    with torch.no_grad():
        pred_A = model.sample_embeddings.weight.cpu().numpy()
        pred_B = model.time_embeddings.weight.cpu().numpy()
        pred_C = model.spec_embeddings.weight.cpu().numpy()
        
    aligned_A, aligned_B, aligned_C, metrics = match_and_align_chroma_components(
        dataset['A'], dataset['B'], dataset['C'], pred_A, pred_B, pred_C
    )
    
    avg_r2_a = np.max([0.0, np.mean(metrics['r2_A'])])
    avg_r2_b = np.max([0.0, np.mean(metrics['r2_B'])])
    avg_r2_c = np.max([0.0, np.mean(metrics['r2_C'])])
    
    st.session_state.r2_a.append(avg_r2_a)
    st.session_state.r2_b.append(avg_r2_b)
    st.session_state.r2_c.append(avg_r2_c)
    
    st.session_state.aligned_A = aligned_A
    st.session_state.aligned_B = aligned_B
    st.session_state.aligned_C = aligned_C
    
    return loss_val, avg_r2_a, avg_r2_b, avg_r2_c

# --- Conditional Dashboard Body Render ---
if not st.session_state.initialized:
    if track_mode == "EEM Spectroscopy":
        st.markdown("""
        <div class="welcome-card">
            <h2>👋 Welcome to the Spectroscopy Gray-Box Simulator</h2>
            <p>
                Standard multi-way chemometrics (like PARAFAC) assumes ideal, linear trilinear systems.
                In actual chemical laboratory samples, measurements are heavily corrupted by <strong>optical scattering artifacts</strong> 
                and non-linear <strong>Inner Filter Effects (IFE)</strong>. 
                This simulator allows you to generate synthetic spectroscopy datasets with physical interferences, 
                and solve them using a Physics-Embedded Tensor Network (PETN) gray-box model.
            </p>
        </div>
        
        <div class="interference-grid">
            <div class="interference-card">
                <div class="card-icon">⚡</div>
                <h3>1. Rayleigh & Raman Scattering diagonals</h3>
                <p>
                    1st and 2nd order scattering create high-intensity lines across the diagonal. 
                    Traditional models attempt to fit this scattering as separate mathematical components, which warps the physical spectral signatures.
                </p>
                <div class="card-tag">Masked Loss Function</div>
                <p class="card-action">
                    The neural network uses a binary weight mask to blind the loss function on scattering diagonals. 
                    This forces the rigid outer-product trilinear layers to interpolate the true chemical spectra smoothly underneath the scattering.
                </p>
            </div>
            <div class="interference-card">
                <div class="card-icon">🧪</div>
                <h3>2. Cuvette Inner Filter Effect (IFE)</h3>
                <p>
                    Highly concentrated fluorophores absorb both excitation and emission light in the cuvette. 
                    This results in a non-linear attenuation of the observed fluorescence intensity, breaking the trilinear assumption.
                </p>
                <div class="card-tag">Gray-Box Cuvette Attenuation Head</div>
                <p class="card-action">
                    The PETN integrates a physical cuvette attenuation head inside its neural graph. 
                    It learns component-specific molar absorptivities and applies a physical attenuation factor strictly bounded between 0 and 1, mathematically reversing the suppression.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="welcome-card">
            <h2>👋 Welcome to the Chromatography Alignment Gray-Box Simulator</h2>
            <p>
                Standard multi-way chemometrics (like PARAFAC) assumes that chromatograms are perfectly aligned across all runs.
                In practice, flow rate fluctuations, pressure variations, and column aging cause peaks to elute earlier or later (retention time shifting).
                This simulator allows you to generate synthetic chromatography datasets with peak shifts and solve them using
                the **Chroma-PETN** gray-box model, which embeds differentiable warping and Savitzky-Golay filters.
            </p>
        </div>
        
        <div class="interference-grid">
            <div class="interference-card">
                <div class="card-icon">🌀</div>
                <h3>1. Differentiable Warping Head</h3>
                <p>
                    The model embeds sample-specific warping parameters (linear, quadratic, or spline segment-based) to continuously map observed time coordinates to canonical coordinates.
                </p>
                <div class="card-tag">Differentiable 1D Interpolation</div>
                <p class="card-action">
                    By doing linear interpolation over the time embedding using backpropagatable warped coordinates, gradients flow back directly to both alignment parameters and canonical profiles.
                </p>
            </div>
            <div class="interference-card">
                <div class="card-icon">📉</div>
                <h3>2. Savitzky-Golay Derivative Head</h3>
                <p>
                    When baseline drift or broad interferences corrupt the chromatography peaks, derivatives are used to isolate sharp chemical components.
                </p>
                <div class="card-tag">Savitzky-Golay Filter Layer</div>
                <p class="card-action">
                    By evaluating derivatives *inside* the forward pass, we apply constraints on the raw (non-negative) profiles rather than on the derivative profiles, resolving physical peak shapes.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Extract current state metrics
    current_loss = st.session_state.losses[-1] if st.session_state.losses else 0.0
    r2_score_val = st.session_state.r2_a[-1] if st.session_state.r2_a else 0.0
    r2_b_val = st.session_state.r2_b[-1] if st.session_state.r2_b else 0.0
    r2_c_val = st.session_state.r2_c[-1] if st.session_state.r2_c else 0.0

    # Render custom metric cards
    render_metrics_grid(
        st.session_state.epoch, 
        total_epochs, 
        current_loss, 
        r2_score_val, 
        r2_b_val, 
        r2_c_val,
        mode="EEM" if track_mode == "EEM Spectroscopy" else "Chroma"
    )

    # --- Setup Tabs ---
    tab_lbl_heatmap = "🗺️ EEM Heatmaps Comparison" if track_mode == "EEM Spectroscopy" else "🗺️ Chromatogram Heatmaps"
    tab_lbl_abs = "🧪 Cuvette & Absorptivities" if track_mode == "EEM Spectroscopy" else "⚙️ Warping Functions"
    
    tab_fitting, tab_heatmaps, tab_loadings, tab_absorbance = st.tabs([
        "📈 Fitting Metrics", tab_lbl_heatmap, "🧬 Resolved Loadings", tab_lbl_abs
    ])
    
    generator = st.session_state.generator
    dataset = st.session_state.dataset

    # TAB 1: Convergence History Curves
    with tab_fitting:
        st.subheader("Convergence & Alignment Curves")
        st.markdown("""
        These live graphs map the optimization history:
        * **Loss Curve (Left)**: Log-scale plot of the MSE loss converging towards zero.
        * **Recovery Profile (Right)**: Recovery of chemical parameters ($R^2$ similarity compared to true ground truth) for Sample Scores ($A$), loadings ($B$), and loadings ($C$).
        """)
        
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            fig_loss = go.Figure()
            epochs_x = [i * epochs_per_update for i in range(1, len(st.session_state.losses) + 1)]
            if not epochs_x:
                epochs_x = [0]
                losses_y = [0]
            else:
                losses_y = st.session_state.losses
                
            fig_loss.add_trace(go.Scatter(
                x=epochs_x, 
                y=losses_y, 
                mode='lines', 
                name='Loss', 
                line=dict(color='#58a6ff', width=2.5)
            ))
            fig_loss.update_layout(
                title="Training Loss Curve (MSE)",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400,
                **PLOTLY_THEME_LAYOUT
            )
            fig_loss.update_yaxes(type="log")
            st.plotly_chart(fig_loss, use_container_width=True)
            
        with col_f2:
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(x=epochs_x, y=st.session_state.r2_a, mode='lines', name='Scores (A)', line=dict(color='#ff7f0e', width=2)))
            fig_r2.add_trace(go.Scatter(x=epochs_x, y=st.session_state.r2_b, mode='lines', name='Loadings (B)', line=dict(color='#2ca02c', width=2)))
            fig_r2.add_trace(go.Scatter(x=epochs_x, y=st.session_state.r2_c, mode='lines', name='Loadings (C)', line=dict(color='#d62728', width=2)))
            
            fig_r2.update_layout(
                title="Ground Truth Component Recovery Profile (R²)",
                xaxis_title="Epoch",
                yaxis_title="R² Score Similarity",
                height=400,
                **PLOTLY_THEME_LAYOUT
            )
            fig_r2.update_yaxes(range=[-0.05, 1.05])
            st.plotly_chart(fig_r2, use_container_width=True)

    # TAB 2: Heatmaps Comparison
    with tab_heatmaps:
        if track_mode == "EEM Spectroscopy":
            st.subheader("2D Excitation-Emission Matrix (EEM) Heatmap Profiles")
            st.markdown("""
            Compare the resolved components for a selected sample in the generated batch:
            1. **True Clean EEM**: Target unattenuated fluorescence spectrum.
            2. **Lab Observed EEM**: Raw simulated measurement, distorted by scattering diagonals and cuvette attenuation.
            3. **Reconstructed Observed EEM**: Model's prediction. The network successfully ignores the scattering diagonals because of the custom masked loss, while fitting the valid data profile.
            4. **Recovered Clean EEM**: Pure chemical signal extracted by the model, demonstrating complete removal of scattering and mathematical inversion of the Inner Filter Effect.
            """)
            
            sample_idx = st.slider("Select Sample Index to Display", min_value=0, max_value=generator.num_samples - 1, value=0)
            
            X_true_sample = dataset['X_true'][sample_idx]
            X_obs_sample = dataset['X'][sample_idx]
            
            if st.session_state.aligned_A is not None:
                pred_true_sample = np.einsum('r,jr,kr->jk', 
                                             st.session_state.aligned_A[sample_idx], 
                                             st.session_state.aligned_B, 
                                             st.session_state.aligned_C)
                
                with torch.no_grad():
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
            
            max_val = float(np.max(X_true_sample)) if np.max(X_true_sample) > 0 else 1.0
            
            fig_heat.add_trace(go.Contour(
                z=X_true_sample.T, x=generator.ex_wavelens, y=generator.em_wavelens, 
                colorscale="Plasma", zmin=0, zmax=max_val, showscale=False, contours=dict(showlines=False)
            ), row=1, col=1)
            
            fig_heat.add_trace(go.Contour(
                z=X_obs_sample.T, x=generator.ex_wavelens, y=generator.em_wavelens, 
                colorscale="Plasma", zmin=0, zmax=max_val * 1.5, showscale=False, contours=dict(showlines=False)
            ), row=1, col=2)
            
            fig_heat.add_trace(go.Contour(
                z=pred_obs_sample.T, x=generator.ex_wavelens, y=generator.em_wavelens, 
                colorscale="Plasma", zmin=0, zmax=max_val, showscale=False, contours=dict(showlines=False)
            ), row=2, col=1)
            
            fig_heat.add_trace(go.Contour(
                z=pred_true_sample.T, x=generator.ex_wavelens, y=generator.em_wavelens, 
                colorscale="Plasma", zmin=0, zmax=max_val, showscale=False, contours=dict(showlines=False)
            ), row=2, col=2)
            
            fig_heat.update_layout(
                template="plotly_dark",
                height=680,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans, Outfit, sans-serif")
            )
            
            fig_heat.update_xaxes(title_text="Excitation Wavelength (nm)", showgrid=False, linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8b949e"))
            fig_heat.update_yaxes(title_text="Emission Wavelength (nm)", showgrid=False, linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8b949e"))
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.subheader("2D Chromatography (Retention Time vs Spectrum) Heatmaps")
            st.markdown("""
            Compare the resolved chromatograms for a selected sample in the generated batch:
            1. **True Clean Chromatogram**: Target aligned, noise-free chromatogram.
            2. **Lab Observed Chromatogram**: Raw simulated measurement, distorted by retention time shifts and noise.
            3. **Reconstructed Observed Chromatogram**: Model's prediction of the shifted profile.
            4. **Recovered Aligned Chromatogram**: Pure chemical signal aligned by the model (warping removed).
            """)
            
            sample_idx = st.slider("Select Sample Index to Display", min_value=0, max_value=generator.num_samples - 1, value=0)
            
            X_true_sample = dataset['X_true_unshifted'][sample_idx]
            X_obs_sample = dataset['X'][sample_idx]
            
            if st.session_state.aligned_A is not None:
                pred_true_sample = np.einsum('r,jr,kr->jk', 
                                             st.session_state.aligned_A[sample_idx], 
                                             st.session_state.aligned_B, 
                                             st.session_state.aligned_C)
                
                with torch.no_grad():
                    sample_grid, time_grid, spec_grid = np.meshgrid(
                        np.array([sample_idx]),
                        np.arange(generator.num_time),
                        np.arange(generator.num_spec),
                        indexing='ij'
                    )
                    sample_indices_q = torch.tensor(sample_grid.reshape(-1), dtype=torch.long)
                    time_indices_q = torch.tensor(time_grid.reshape(-1), dtype=torch.long)
                    spec_indices_q = torch.tensor(spec_grid.reshape(-1), dtype=torch.long)
                    
                    pred_obs_flat = st.session_state.model(sample_indices_q, time_indices_q, spec_indices_q)
                    pred_obs_sample = pred_obs_flat.reshape(generator.num_time, generator.num_spec).cpu().numpy()
            else:
                pred_true_sample = np.zeros_like(X_true_sample)
                pred_obs_sample = np.zeros_like(X_obs_sample)
                
            fig_heat = make_subplots(
                rows=2, cols=2, 
                subplot_titles=(
                    "1. True Clean Chromatogram (Aligned Target)", 
                    "2. Lab Observed Chromatogram (Shifted & Noisy)", 
                    "3. Reconstructed Observed Chromatogram (Model Fit)", 
                    "4. Recovered Aligned Chromatogram (Resolved by Model)"
                )
            )
            
            max_val = float(np.max(X_true_sample)) if np.max(X_true_sample) > 0 else 1.0
            
            fig_heat.add_trace(go.Contour(
                z=X_true_sample.T, x=generator.time_grid, y=generator.spec_grid, 
                colorscale="Plasma", zmin=0, zmax=max_val, showscale=False, contours=dict(showlines=False)
            ), row=1, col=1)
            
            fig_heat.add_trace(go.Contour(
                z=X_obs_sample.T, x=generator.time_grid, y=generator.spec_grid, 
                colorscale="Plasma", zmin=0, zmax=max_val * 1.5, showscale=False, contours=dict(showlines=False)
            ), row=1, col=2)
            
            fig_heat.add_trace(go.Contour(
                z=pred_obs_sample.T, x=generator.time_grid, y=generator.spec_grid, 
                colorscale="Plasma", zmin=0, zmax=max_val, showscale=False, contours=dict(showlines=False)
            ), row=2, col=1)
            
            fig_heat.add_trace(go.Contour(
                z=pred_true_sample.T, x=generator.time_grid, y=generator.spec_grid, 
                colorscale="Plasma", zmin=0, zmax=max_val, showscale=False, contours=dict(showlines=False)
            ), row=2, col=2)
            
            fig_heat.update_layout(
                template="plotly_dark",
                height=680,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans, Outfit, sans-serif")
            )
            
            fig_heat.update_xaxes(title_text="Retention Time (Normalized)", showgrid=False, linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8b949e"))
            fig_heat.update_yaxes(title_text="Wavelength (nm)", showgrid=False, linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8b949e"))
            st.plotly_chart(fig_heat, use_container_width=True)

    # TAB 3: Resolved Loadings
    with tab_loadings:
        if track_mode == "EEM Spectroscopy":
            st.subheader("Component Loading Profiles Verification")
            st.markdown("""
            Compare model-resolved loadings (solid lines) vs. ground truth clean chemical loadings (dashed lines) for the three components:
            * **Excitation Loadings (Left)**: Recovered components in the Excitation dimension ($B$).
            * **Emission Loadings (Right)**: Recovered components in the Emission dimension ($C$).
            * **Concentration Scores (Bottom)**: True sample scores ($A$) plotted against resolved scores, showing linear reconstruction correlation.
            """)
            
            if st.session_state.aligned_B is not None:
                fig_load = make_subplots(rows=1, cols=2, subplot_titles=("Excitation Loadings (B)", "Emission Loadings (C)"))
                
                colors_comp = ['#1f77b4', '#2ca02c', '#d62728']
                comp_names = ["Component 1 (Phenanthrene-like)", "Component 2 (Anthracene-like)", "Component 3 (Humic-like)"]
                
                for r in range(generator.num_components):
                    fig_load.add_trace(go.Scatter(
                        x=generator.ex_wavelens, y=dataset['B'][:, r], 
                        mode='lines', name=f"True {comp_names[r]}", 
                        line=dict(color=colors_comp[r], width=1.5, dash='dash')
                    ), row=1, col=1)
                    fig_load.add_trace(go.Scatter(
                        x=generator.ex_wavelens, y=st.session_state.aligned_B[:, r], 
                        mode='lines', name=f"Resolved {comp_names[r]}", 
                        line=dict(color=colors_comp[r], width=2.5)
                    ), row=1, col=1)
                    
                    fig_load.add_trace(go.Scatter(
                        x=generator.em_wavelens, y=dataset['C'][:, r], 
                        mode='lines', name=f"True {comp_names[r]}", 
                        line=dict(color=colors_comp[r], width=1.5, dash='dash'), 
                        showlegend=False
                    ), row=1, col=2)
                    fig_load.add_trace(go.Scatter(
                        x=generator.em_wavelens, y=st.session_state.aligned_C[:, r], 
                        mode='lines', name=f"Resolved {comp_names[r]}", 
                        line=dict(color=colors_comp[r], width=2.5), 
                        showlegend=False
                    ), row=1, col=2)
                    
                fig_load.update_layout(
                    template="plotly_dark",
                    height=450,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans, Outfit, sans-serif")
                )
                fig_load.update_xaxes(title_text="Wavelength (nm)", showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8b949e"))
                fig_load.update_yaxes(title_text="Normalized Intensity", showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8b949e"))
                st.plotly_chart(fig_load, use_container_width=True)
                
                # Scores Scatter
                st.subheader("Relative Concentration Scores (A) Recovery Verification")
                fig_scores = go.Figure()
                for r in range(generator.num_components):
                    fig_scores.add_trace(go.Scatter(
                        x=dataset['A'][:, r], y=st.session_state.aligned_A[:, r], 
                        mode='markers', name=comp_names[r], 
                        marker=dict(color=colors_comp[r], size=10, line=dict(width=1, color='white'))
                    ))
                    
                all_scores = np.concatenate([dataset['A'].flatten(), st.session_state.aligned_A.flatten()])
                min_sc, max_sc = float(np.min(all_scores)), float(np.max(all_scores))
                fig_scores.add_trace(go.Scatter(
                    x=[min_sc, max_sc], y=[min_sc, max_sc], 
                    mode='lines', name='Ideal Recovery (y=x)', 
                    line=dict(color='#8b949e', dash='dot')
                ))
                
                fig_scores.update_layout(
                    title="True Concentration vs. Resolved Model Concentration (Scores A)",
                    xaxis_title="True Prepared Score (Concentration)",
                    yaxis_title="Model Resolved Aligned Score",
                    height=400,
                    **PLOTLY_THEME_LAYOUT
                )
                st.plotly_chart(fig_scores, use_container_width=True)
            else:
                st.info("Start solver training to display resolved loading profiles.")
        else:
            st.subheader("Component Chromatography and Spectral Profiles Verification")
            st.markdown("""
            Compare model-resolved profiles (solid lines) vs. ground truth clean chemical profiles (dashed lines):
            * **Chromatography Profiles (Left)**: Recovered aligned chromatograms in the Retention Time dimension ($B$).
            * **Spectral Profiles (Right)**: Recovered components in the Spectral dimension ($C$).
            * **Concentration Scores (Bottom)**: True sample scores ($A$) plotted against resolved scores, showing linear reconstruction correlation.
            """)
            
            if st.session_state.aligned_B is not None:
                fig_load = make_subplots(rows=1, cols=2, subplot_titles=("Retention Time Profiles (B)", "Spectral Profiles (C)"))
                
                colors_comp = ['#1f77b4', '#2ca02c', '#d62728']
                comp_names = ["Component 1", "Component 2", "Component 3"]
                
                for r in range(generator.num_components):
                    fig_load.add_trace(go.Scatter(
                        x=generator.time_grid, y=dataset['B'][:, r], 
                        mode='lines', name=f"True {comp_names[r]}", 
                        line=dict(color=colors_comp[r], width=1.5, dash='dash')
                    ), row=1, col=1)
                    fig_load.add_trace(go.Scatter(
                        x=generator.time_grid, y=st.session_state.aligned_B[:, r], 
                        mode='lines', name=f"Resolved {comp_names[r]}", 
                        line=dict(color=colors_comp[r], width=2.5)
                    ), row=1, col=1)
                    
                    fig_load.add_trace(go.Scatter(
                        x=generator.spec_grid, y=dataset['C'][:, r], 
                        mode='lines', name=f"True {comp_names[r]}", 
                        line=dict(color=colors_comp[r], width=1.5, dash='dash'), 
                        showlegend=False
                    ), row=1, col=2)
                    fig_load.add_trace(go.Scatter(
                        x=generator.spec_grid, y=st.session_state.aligned_C[:, r], 
                        mode='lines', name=f"Resolved {comp_names[r]}", 
                        line=dict(color=colors_comp[r], width=2.5), 
                        showlegend=False
                    ), row=1, col=2)
                    
                fig_load.update_layout(
                    template="plotly_dark",
                    height=450,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans, Outfit, sans-serif")
                )
                fig_load.update_xaxes(title_text="Normalized Time / Wavelength", showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8b949e"))
                fig_load.update_yaxes(title_text="Normalized Intensity", showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#8b949e"))
                st.plotly_chart(fig_load, use_container_width=True)
                
                # Scores Scatter
                st.subheader("Relative Concentration Scores (A) Recovery Verification")
                fig_scores = go.Figure()
                for r in range(generator.num_components):
                    fig_scores.add_trace(go.Scatter(
                        x=dataset['A'][:, r], y=st.session_state.aligned_A[:, r], 
                        mode='markers', name=comp_names[r], 
                        marker=dict(color=colors_comp[r], size=10, line=dict(width=1, color='white'))
                    ))
                    
                all_scores = np.concatenate([dataset['A'].flatten(), st.session_state.aligned_A.flatten()])
                min_sc, max_sc = float(np.min(all_scores)), float(np.max(all_scores))
                fig_scores.add_trace(go.Scatter(
                    x=[min_sc, max_sc], y=[min_sc, max_sc], 
                    mode='lines', name='Ideal Recovery (y=x)', 
                    line=dict(color='#8b949e', dash='dot')
                ))
                
                fig_scores.update_layout(
                    title="True Concentration vs. Resolved Model Concentration (Scores A)",
                    xaxis_title="True Prepared Score (Concentration)",
                    yaxis_title="Model Resolved Aligned Score",
                    height=400,
                    **PLOTLY_THEME_LAYOUT
                )
                st.plotly_chart(fig_scores, use_container_width=True)
            else:
                st.info("Start solver training to display resolved loading profiles.")

    # TAB 4: Warping / Cuvette
    with tab_absorbance:
        if track_mode == "EEM Spectroscopy":
            st.subheader("Physical Parameter Resolution Verification")
            st.markdown("""
            Verify the physical cuvette attributes learned by the gray-box model:
            * **Molar Absorptivity (Left)**: Resolved component molar absorptivities ($E = \\alpha_r \\cdot B$). This is the learned physical absorption scaling parameter.
            * **Solvent Background (Right)**: Model-extracted excitation solvent background profile ($Abs_{bg, ex}$) anchoring embedding scale.
            """)
            
            if model_type == "Pure PARAFAC":
                st.markdown("""
                <div class="warning-card">
                    <div class="warning-icon">⚠️</div>
                    <div class="warning-content">
                        <h3>Pure PARAFAC Physical Blindness</h3>
                        <p>
                            Pure PARAFAC operates as a linear mathematical decomposition and is physically blind to the Cuvette Inner Filter Effect. 
                            No molar absorptivities or solvent background profiles are modeled. 
                            Switch to the <strong>PETN-PARAFAC</strong> architecture in the sidebar to enable gray-box attenuation modeling.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.aligned_E is not None:
                col_ab1, col_ab2 = st.columns(2)
                colors_comp = ['#1f77b4', '#2ca02c', '#d62728']
                comp_names = ["Component 1 (Phenanthrene-like)", "Component 2 (Anthracene-like)", "Component 3 (Humic-like)"]
                
                with col_ab1:
                    st.subheader("Molar Absorptivity Scaling (E = α*B)")
                    fig_abs = go.Figure()
                    
                    true_E = dataset['E']
                    if true_E is None:
                        true_E = dataset['B'] * np.array([0.15, 0.10, 0.20])
                    
                    for r in range(generator.num_components):
                        fig_abs.add_trace(go.Scatter(
                            x=generator.ex_wavelens, y=true_E[:, r], 
                            mode='lines', name=f"True α*B {comp_names[r]}", 
                            line=dict(color=colors_comp[r], width=1.5, dash='dash')
                        ))
                        fig_abs.add_trace(go.Scatter(
                            x=generator.ex_wavelens, y=st.session_state.aligned_E[:, r], 
                            mode='lines', name=f"Resolved α*B {comp_names[r]}", 
                            line=dict(color=colors_comp[r], width=2.5)
                        ))
                        
                    fig_abs.update_layout(
                        xaxis_title="Excitation Wavelength (nm)",
                        yaxis_title="Molar Absorptivity (L / mol / cm)",
                        height=400,
                        **PLOTLY_THEME_LAYOUT
                    )
                    st.plotly_chart(fig_abs, use_container_width=True)
                    
                with col_ab2:
                    st.subheader("Solvent Background Absorbance")
                    fig_bg = go.Figure()
                    
                    lambda_0 = 240.0
                    A_bg_ex = 0.10 * np.exp(-0.010 * (generator.ex_wavelens - lambda_0))
                    
                    fig_bg.add_trace(go.Scatter(
                        x=generator.ex_wavelens, y=A_bg_ex, 
                        mode='lines', name="True Excitation Solvent Abs_bg", 
                        line=dict(color='#ff7f0e', width=1.5, dash='dash')
                    ))
                    fig_bg.add_trace(go.Scatter(
                        x=generator.ex_wavelens, y=st.session_state.pred_ex_bg, 
                        mode='lines', name="Registered Excitation Solvent Abs_bg", 
                        line=dict(color='#ff7f0e', width=2.5)
                    ))
                    
                    fig_bg.update_layout(
                        xaxis_title="Excitation Wavelength (nm)",
                        yaxis_title="Absorbance Units",
                        height=400,
                        **PLOTLY_THEME_LAYOUT
                    )
                    st.plotly_chart(fig_bg, use_container_width=True)
            else:
                st.info("Start solver training to display resolved physical absorptivities.")
        else:
            st.subheader("Retention Time Shift Warping Functions")
            st.markdown("""
            Verify the retention time alignment profiles learned by the model:
            * **Shift Perturbations (y = t' - t)**: Displays the shift offset across the chromatogram.
              Solid lines represent the model's learned warping function for each sample, and dashed lines represent the true simulated warpings.
              The more closely the solid lines track the dashed lines, the more accurately the model has recovered the alignment physical parameters.
            """)
            
            if chroma_model_arch == "Pure PARAFAC":
                st.markdown("""
                <div class="warning-card">
                    <div class="warning-icon">⚠️</div>
                    <div class="warning-content">
                        <h3>Pure PARAFAC Alignment Blindness</h3>
                        <p>
                            Pure PARAFAC assumes that peak shapes are perfectly aligned across samples and operates as a linear mathematical decomposition.
                            It does not model coordinate warping.
                            Switch to the <strong>Chroma-PETN</strong> architecture in the sidebar to enable differentiable coordinate warping.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.aligned_B is not None:
                fig_warp = go.Figure()
                t_obs = generator.time_grid
                colors_samples = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f']
                
                model = st.session_state.model
                num_plots = min(5, generator.num_samples)
                
                for i in range(num_plots):
                    sim_warp = dataset['warp_type']
                    if sim_warp == 'linear':
                        t_true_warped = (t_obs - dataset['shifts'][i]) / (1.0 + dataset['stretches'][i])
                    elif sim_warp == 'quadratic':
                        alpha = dataset['alphas'][i]
                        beta = dataset['betas'][i]
                        gamma = dataset['gammas'][i]
                        t_true_warped = t_obs - (alpha * (t_obs ** 2) + beta * t_obs + gamma)
                    elif sim_warp == 'spline':
                        shift = dataset['shifts'][i]
                        stretch = dataset['stretches'][i] * 0.5
                        t_true_warped = t_obs - (shift + stretch * np.sin(np.pi * t_obs))
                        
                    true_diff = t_true_warped - t_obs
                    
                    if model.warp_type == 'linear':
                        stretch_pred = model.alpha_params[i].mean().item()
                        shift_pred = model.beta_params[i].mean().item()
                        t_pred_warped = t_obs - (stretch_pred * t_obs + shift_pred)
                    elif model.warp_type == 'quadratic':
                        alpha_pred = model.alpha_params[i].mean().item()
                        beta_pred = model.beta_params[i].mean().item()
                        gamma_pred = model.gamma_params[i].mean().item()
                        t_pred_warped = t_obs - (alpha_pred * (t_obs**2) + beta_pred * t_obs + gamma_pred)
                    elif model.warp_type == 'spline':
                        shift_pred = model.beta_params[i].mean().item()
                        log_inc_pred = model.log_increments_params[i].mean(dim=-1).detach().cpu().numpy()
                        inc_pred = (1.0 / model.num_segments) * np.exp(log_inc_pred)
                        w_pred = shift_pred + np.cumsum(np.concatenate([[0.0], inc_pred]))
                        val = t_obs * model.num_segments
                        k = np.clip(np.floor(val).astype(int), 0, model.num_segments - 1)
                        u = val - k
                        t_pred_warped = (1.0 - u) * w_pred[k] + u * w_pred[k + 1]

                        
                    pred_diff = t_pred_warped - t_obs
                    
                    true_diff_centered = true_diff - true_diff.mean()
                    pred_diff_centered = pred_diff - pred_diff.mean()
                    
                    fig_warp.add_trace(go.Scatter(
                        x=t_obs, y=true_diff_centered, 
                        mode='lines', name=f"Sample {i+1} True Shift", 
                        line=dict(color=colors_samples[i % len(colors_samples)], width=1.5, dash='dash')
                    ))
                    fig_warp.add_trace(go.Scatter(
                        x=t_obs, y=pred_diff_centered, 
                        mode='lines', name=f"Sample {i+1} Learned Shift", 
                        line=dict(color=colors_samples[i % len(colors_samples)], width=2.5)
                    ))
                    
                fig_warp.update_layout(
                    title="True vs. Learned Retention Time Shift Deviation (Centered)",
                    xaxis_title="Retention Time (Normalized)",
                    yaxis_title="Shift Offset (t' - t)",
                    height=450,
                    **PLOTLY_THEME_LAYOUT
                )
                st.plotly_chart(fig_warp, use_container_width=True)
            else:
                st.info("Start solver training to display warping profiles.")

# --- Training Loop Execution (must run at the end to allow state rendering) ---
if st.session_state.is_training:
    if st.session_state.epoch < total_epochs:
        if track_mode == "EEM Spectroscopy":
            loss_val, r2_a, r2_b, r2_c = run_training_step(epochs_per_update, lr, model_type)
        else:
            loss_val, r2_a, r2_b, r2_c = run_chroma_training_step(epochs_per_update, chroma_lr, chroma_model_arch)
            
        if ui_delay > 0:
            time.sleep(ui_delay)
        st.rerun()
    else:
        st.session_state.is_training = False
        st.success("Training fully complete!")
        st.rerun()
