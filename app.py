import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import httpx
import streamlit.components.v1 as components

# Add src to path to import model architecture and helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from models import LSTMModel
from data_preprocessing import load_data, process_test_data

# Set Page Config
st.set_page_config(
    page_title="Predictive Maintenance Production Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Main Layout Adjustments */
    .reportview-container {
        background-color: #0e1117;
    }
    
    /* Header styling */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1E88E5;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #8A9Aad;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric Card Styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #1a1f29;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        width: 100%;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Dynamic Status colors */
    .status-healthy {
        color: #10B981 !important; /* Green */
        font-weight: bold;
    }
    .status-caution {
        color: #F59E0B !important; /* Amber */
        font-weight: bold;
    }
    .status-critical {
        color: #EF4444 !important; /* Red */
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Sidebar Configuration -----------------
st.sidebar.markdown("## ⚙️ Configuration & MLOps API")

API_URL = st.sidebar.text_input("FastAPI Base URL", value="http://localhost:8000", help="URL of the FastAPI inference service.")

# Check FastAPI status
fastapi_connected = False
try:
    response = httpx.get(f"{API_URL}/health", timeout=1.0)
    if response.status_code == 200 and response.json().get("status") == "healthy":
        fastapi_connected = True
        st.sidebar.success("🟢 Connected to FastAPI Server")
    else:
        st.sidebar.warning("⚠️ FastAPI degraded status")
except Exception:
    st.sidebar.info("ℹ️ FastAPI Server Offline. (Using local fallback)")

# File Uploaders
uploaded_data = st.sidebar.file_uploader("Upload Sensor Telemetry File", type=["txt", "csv"], help="Upload 'test_FD001.txt' or equivalent telemetry file.")
uploaded_labels = st.sidebar.file_uploader("Upload Ground Truth RUL (Optional)", type=["txt", "csv"], help="Upload 'RUL_FD001.txt' to evaluate prediction accuracy.")

# Fallback datasets if nothing uploaded
data_path = "data/"
default_test_file = os.path.join(data_path, "test_FD001.txt")
default_rul_file = os.path.join(data_path, "RUL_FD001.txt")

# Determine final file sources
if uploaded_data is not None:
    data_source = uploaded_data
    labels_source = uploaded_labels if uploaded_labels is not None else None
else:
    data_source = default_test_file if os.path.exists(default_test_file) else None
    labels_source = default_rul_file if os.path.exists(default_rul_file) else None

# Check if we have data to read
if data_source is None:
    st.warning("⚠️ Telemetry dataset not found. Please upload a file to start.")
    st.stop()

# ----------------- Load Assets (Cached Local Fallback) -----------------
@st.cache_resource
def get_model(model_path):
    if not os.path.exists(model_path):
        return None
    model = LSTMModel(input_dim=17, hidden_dim=100, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def get_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        return None
    return joblib.load(scaler_path)

# Load local resources
local_model = get_model('src/lstm_model.pth')
local_scaler = get_scaler('src/scaler.joblib')

# ----------------- Preprocessing Helper -----------------
def preprocess_data(raw_data_file, labels_file=None):
    """Loads and preprocesses test data and optionally merges RUL labels."""
    df_raw = load_data(raw_data_file)
    
    if labels_file is not None:
        rul_df = pd.read_csv(labels_file, sep=r'\s+', header=None)
        df_processed = process_test_data(df_raw, rul_df, cap=125)
        has_labels = True
    else:
        df_processed = df_raw.copy()
        df_processed['RUL'] = np.nan
        df_processed['label_bc'] = 0
        has_labels = False
        
    return df_processed, has_labels

# Load and Preprocess Dataset
with st.spinner("Processing telemetry data..."):
    try:
        df_processed, has_labels = preprocess_data(data_source, labels_source)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

# Define feature columns
sensor_cols = [c for c in df_processed.columns if c.startswith('s_')]
setting_cols = [c for c in df_processed.columns if c.startswith('setting_')]
seq_cols = sensor_cols + setting_cols
SEQUENCE_LENGTH = 50

# Unit Selection
available_units = df_processed['unit_nr'].unique()
selected_unit = st.sidebar.selectbox("Select Engine Unit ID", sorted(available_units))

# Sensor Filtering for Visualization
default_sensors = ['s_2', 's_7', 's_11', 's_15']
selected_sensors = st.sidebar.multiselect(
    "Sensors to Plot",
    options=sorted(sensor_cols),
    default=[s for s in default_sensors if s in sensor_cols]
)

# ----------------- Core Inference Engine -----------------
def get_unit_predictions_local(unit_df, num_cycles):
    """Calculates predictions using local model files."""
    if local_model is None or local_scaler is None:
        st.error("❌ Local model/scaler files not found. Please train the model or connect to FastAPI.")
        st.stop()
        
    scaled_df = unit_df.copy()
    scaled_df[seq_cols] = local_scaler.transform(scaled_df[seq_cols])
    
    sequences = []
    cycles = []
    
    for i in range(SEQUENCE_LENGTH, num_cycles + 1):
        seq = scaled_df[seq_cols].values[i - SEQUENCE_LENGTH : i]
        sequences.append(seq)
        cycles.append(unit_df['time_cycles'].values[i - 1])
        
    sequences = np.array(sequences, dtype=np.float32)
    input_tensor = torch.from_numpy(sequences)
    with torch.no_grad():
        pred_rul, pred_cls = local_model(input_tensor)
        pred_rul = pred_rul.numpy().flatten()
        pred_cls_prob = pred_cls.numpy().flatten()
        
    return pred_rul, pred_cls_prob, cycles

def get_unit_predictions_api(unit_df, num_cycles):
    """Calculates predictions using the FastAPI REST API."""
    sequences_unscaled = []
    cycles = []
    
    for i in range(SEQUENCE_LENGTH, num_cycles + 1):
        # We pass unscaled sequences because FastAPI handles scaling internally
        seq = unit_df[seq_cols].values[i - SEQUENCE_LENGTH : i].tolist()
        sequences_unscaled.append(seq)
        cycles.append(unit_df['time_cycles'].values[i - 1])
        
    payload = {"sequences": sequences_unscaled}
    response = httpx.post(f"{API_URL}/predict/batch", json=payload, timeout=10.0)
    
    if response.status_code != 200:
        raise RuntimeError(f"FastAPI error ({response.status_code}): {response.text}")
        
    batch_results = response.json()["batch_results"]
    
    pred_rul = []
    pred_cls_prob = []
    for res in batch_results:
        if "error" in res:
            pred_rul.append(0.0)
            pred_cls_prob.append(0.0)
        else:
            pred_rul.append(res["predicted_rul"])
            pred_cls_prob.append(res["failure_probability_30c"])
            
    return np.array(pred_rul), np.array(pred_cls_prob), cycles

@st.cache_data(show_spinner="Running model inference...")
def get_unit_predictions(unit_id, use_api=False):
    """Calculates model predictions for all valid cycles of a specific unit."""
    unit_df = df_processed[df_processed['unit_nr'] == unit_id].copy()
    num_cycles = len(unit_df)
    
    if num_cycles < SEQUENCE_LENGTH:
        return unit_df, None
    
    try:
        if use_api:
            pred_rul, pred_cls_prob, cycles = get_unit_predictions_api(unit_df, num_cycles)
        else:
            pred_rul, pred_cls_prob, cycles = get_unit_predictions_local(unit_df, num_cycles)
            
        # Clip predictions to prevent negative RUL
        pred_rul = np.clip(pred_rul, 0, None)
        
        # Store predictions matching the cycle numbers
        pred_df = pd.DataFrame({
            'time_cycles': cycles,
            'pred_RUL': pred_rul,
            'failure_prob': pred_cls_prob
        })
        
        merged_df = unit_df.merge(pred_df, on='time_cycles', how='left')
        return merged_df, pred_df
    except Exception as e:
        st.error(f"Inference run failed: {e}. Falling back to local model.")
        pred_rul, pred_cls_prob, cycles = get_unit_predictions_local(unit_df, num_cycles)
        pred_rul = np.clip(pred_rul, 0, None)
        pred_df = pd.DataFrame({
            'time_cycles': cycles,
            'pred_RUL': pred_rul,
            'failure_prob': pred_cls_prob
        })
        merged_df = unit_df.merge(pred_df, on='time_cycles', how='left')
        return merged_df, pred_df

# Run inference on selected unit
unit_data, predictions = get_unit_predictions(selected_unit, use_api=fastapi_connected)

# ----------------- UI Layout & Visuals -----------------

# Header
st.markdown('<div class="main-title">Turbofan Engine Predictive Maintenance</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Remaining Useful Life (RUL) estimation and data drift monitoring</div>', unsafe_allow_html=True)

if len(unit_data) < SEQUENCE_LENGTH:
    st.error(f"❌ Selected Engine Unit {selected_unit} has only {len(unit_data)} cycles of history. A minimum of {SEQUENCE_LENGTH} cycles is required for LSTM sequence processing.")
    st.stop()

# Create tabs for structured navigation
tab_dashboard, tab_glossary, tab_evaluation, tab_drift, tab_raw = st.tabs([
    "📊 Live Analytics & Predictions", 
    "📖 Sensor Reference Glossary", 
    "📈 Model Performance Details",
    "🔍 MLOps Data Drift Monitor",
    "💾 LSTM Input Matrix Details"
])

with tab_dashboard:
    # Slider for Cycle Selection
    max_cycle = int(unit_data['time_cycles'].max())
    selected_cycle = st.slider(
        "Select Operational Cycle to Analyze",
        min_value=SEQUENCE_LENGTH,
        max_value=max_cycle,
        value=max_cycle,
        step=1,
        key="dashboard_cycle_slider"
    )

    # Extract metrics for selected cycle
    cycle_row = unit_data[unit_data['time_cycles'] == selected_cycle].iloc[0]
    pred_rul_val = cycle_row['pred_RUL']
    fail_prob_val = cycle_row['failure_prob']
    actual_rul_val = cycle_row['RUL']

    # Determine status class & description
    if fail_prob_val > 0.5:
        status_label = "CRITICAL FAILURE RISK"
        status_class = "status-critical"
    elif fail_prob_val > 0.2:
        status_label = "CAUTION / WEAR DETECTED"
        status_class = "status-caution"
    else:
        status_label = "HEALTHY / OPERATIONAL"
        status_class = "status-healthy"

    # Display Premium Metrics Cards
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">Current Cycle</div>
            <div class="metric-value">{selected_cycle} / {max_cycle}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Engine Health Status</div>
            <div class="metric-value {status_class}">{status_label}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Predicted RUL</div>
            <div class="metric-value">{pred_rul_val:.1f} cycles</div>
        </div>
        {"<div class='metric-card'><div class='metric-label'>Actual RUL</div><div class='metric-value'>" + (f"{actual_rul_val:.0f} cycles" if has_labels and not np.isnan(actual_rul_val) else "N/A") + "</div></div>" if has_labels else ""}
        <div class="metric-card">
            <div class="metric-label">Failure Probability (30 Cycles)</div>
            <div class="metric-value">{fail_prob_val * 100:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------- Visualizations -----------------
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        'figure.facecolor': '#1a1f29',
        'axes.facecolor': '#1a1f29',
        'text.color': '#ffffff',
        'axes.labelcolor': '#a0aec0',
        'xtick.color': '#a0aec0',
        'ytick.color': '#a0aec0',
        'grid.color': '#2d3748'
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 Remaining Useful Life (RUL) Decline Profile")
        fig_rul, ax_rul = plt.subplots(figsize=(10, 4.5))
        ax_rul.plot(unit_data['time_cycles'], unit_data['pred_RUL'], color='#1E88E5', label='Predicted RUL (LSTM)', lw=2.5)
        if has_labels:
            ax_rul.plot(unit_data['time_cycles'], unit_data['RUL'], color='#10B981', linestyle='--', label='Ground Truth RUL', lw=2)
        ax_rul.axvline(x=selected_cycle, color='#EF4444', linestyle=':', label='Selected Cycle Analysis', lw=2)
        ax_rul.set_xlabel('Operational Cycles')
        ax_rul.set_ylabel('Remaining Useful Life (RUL)')
        ax_rul.legend(facecolor='#1a1f29', edgecolor='#2d3748')
        ax_rul.set_title(f'RUL Degradation Curve for Engine Unit {selected_unit}', fontsize=12, pad=10)
        st.pyplot(fig_rul)

    with col2:
        st.markdown("### ⚠️ Failure Probability Trend")
        fig_prob, ax_prob = plt.subplots(figsize=(10, 4.5))
        ax_prob.plot(unit_data['time_cycles'], unit_data['failure_prob'], color='#F59E0B', lw=2.5, label='Failure Probability')
        ax_prob.axhline(y=0.5, color='#EF4444', linestyle='--', label='Warning Threshold (0.5)', alpha=0.7)
        ax_prob.axvline(x=selected_cycle, color='#EF4444', linestyle=':', lw=2)
        ax_prob.set_xlabel('Operational Cycles')
        ax_prob.set_ylabel('Probability (Failure within 30 cycles)')
        ax_prob.legend(facecolor='#1a1f29', edgecolor='#2d3748')
        ax_prob.set_title('Imminent Failure Probability Progression', fontsize=12, pad=10)
        st.pyplot(fig_prob)

    # Telemetry Trends
    st.markdown("### 📡 Selected Sensor Telemetry Trends")
    if selected_sensors:
        num_plots = len(selected_sensors)
        cols_per_row = 2
        rows = (num_plots + 1) // cols_per_row
        
        fig_sensors, axes = plt.subplots(rows, cols_per_row, figsize=(15, 3.5 * rows), sharex=True)
        if num_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, sensor in enumerate(selected_sensors):
            ax = axes[i]
            ax.plot(unit_data['time_cycles'], unit_data[sensor], color='#93C5FD', lw=1.5)
            ax.axvline(x=selected_cycle, color='#EF4444', linestyle=':', lw=1.5)
            window_start = max(SEQUENCE_LENGTH, selected_cycle - SEQUENCE_LENGTH + 1)
            ax.axvspan(window_start, selected_cycle, color='#1E88E5', alpha=0.1, label='LSTM Input Window' if i==0 else "")
            ax.set_title(f'Sensor {sensor}', fontsize=10)
            if i >= num_plots - cols_per_row:
                ax.set_xlabel('Cycles')
                
        for j in range(num_plots, len(axes)):
            fig_sensors.delaxes(axes[j])
            
        plt.tight_layout()
        st.pyplot(fig_sensors)
    else:
        st.info("Select one or more sensors in the sidebar to plot their telemetry history.")

with tab_glossary:
    st.markdown("""
    ### 📖 NASA CMAPSS Sensor Reference Guide
    This dashboard visualizes telemetry data from the official **NASA CMAPSS (Turbofan Engine Degradation Simulation)** dataset. 
    Below is a reference glossary describing the physical parameter each sensor represents, its operational unit, and its status in our predictive model.
    """)
    
    sensor_metadata = [
        {"Sensor": "s_1", "Symbol": "T2", "Description": "Total temperature at fan inlet", "Unit": "°R", "Status": "🔴 Dropped (Low Variance)"},
        {"Sensor": "s_2", "Symbol": "T24", "Description": "Total temperature at LPC outlet", "Unit": "°R", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_3", "Symbol": "T30", "Description": "Total temperature at HPC outlet", "Unit": "°R", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_4", "Symbol": "T50", "Description": "Total temperature at LPT outlet", "Unit": "°R", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_5", "Symbol": "P2", "Description": "Total pressure at fan inlet", "Unit": "psia", "Status": "🔴 Dropped (Low Variance)"},
        {"Sensor": "s_6", "Symbol": "P15", "Description": "Total pressure in bypass-duct", "Unit": "psia", "Status": "🔴 Dropped (Low Variance)"},
        {"Sensor": "s_7", "Symbol": "P30", "Description": "Total pressure at HPC outlet", "Unit": "psia", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_8", "Symbol": "Nf", "Description": "Physical fan speed", "Unit": "rpm", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_9", "Symbol": "Nc", "Description": "Physical core speed", "Unit": "rpm", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_10", "Symbol": "epr", "Description": "Engine pressure ratio (P50/P2)", "Unit": "-", "Status": "🔴 Dropped (Low Variance)"},
        {"Sensor": "s_11", "Symbol": "Ps30", "Description": "Static pressure at HPC outlet", "Unit": "psia", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_12", "Symbol": "phi", "Description": "Ratio of fuel flow to Ps30", "Unit": "pps/psi", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_13", "Symbol": "NRf", "Description": "Corrected fan speed", "Unit": "rpm", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_14", "Symbol": "NRc", "Description": "Corrected core speed", "Unit": "rpm", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_15", "Symbol": "BPR", "Description": "Bypass ratio", "Unit": "-", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_16", "Symbol": "far", "Description": "Burner fuel-air ratio", "Unit": "-", "Status": "🔴 Dropped (Low Variance)"},
        {"Sensor": "s_17", "Symbol": "htBleed", "Description": "Bleed enthalpy", "Unit": "-", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_18", "Symbol": "Nf_dmd", "Description": "Demanded fan speed", "Unit": "rpm", "Status": "🔴 Dropped (Low Variance)"},
        {"Sensor": "s_19", "Symbol": "PCNfR_dmd", "Description": "Demanded corrected fan speed", "Unit": "rpm", "Status": "🔴 Dropped (Low Variance)"},
        {"Sensor": "s_20", "Symbol": "W31", "Description": "HPT coolant bleed", "Unit": "lbm/s", "Status": "🟢 Active Model Input"},
        {"Sensor": "s_21", "Symbol": "W32", "Description": "LPT coolant bleed", "Unit": "lbm/s", "Status": "🟢 Active Model Input"}
    ]
    
    st.dataframe(pd.DataFrame(sensor_metadata), use_container_width=True, hide_index=True)

with tab_evaluation:
    st.markdown("""
    ### 📈 Global Model Performance (NASA CMAPSS FD001 Test Set)
    Below is the performance comparison between the **Dual-Head LSTM** model and the **XGBoost baseline** logged during the production run.
    """)
    
    metrics_data = [
        {"Model": "XGBoost Baseline", "RMSE (RUL)": "86.20", "MAE (RUL)": "N/A", "NASA Score": "N/A", "Precision": "N/A", "Recall": "N/A", "F1-Score": "N/A"},
        {"Model": "Dual-Head LSTM", "RMSE (RUL)": "13.33", "MAE (RUL)": "9.52", "NASA Score": "281.05", "Precision": "0.92", "Recall": "0.88", "F1-Score": "0.90"}
    ]
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

with tab_drift:
    st.markdown("### 🔍 MLOps Data Drift Monitoring")
    st.write("Compare the distributions of telemetry features received during online predictions against the reference training set to detect environmental changes or sensor degradation.")
    
    if not fastapi_connected:
        st.info("🔌 Connect to an active FastAPI Server via the sidebar configuration to run data drift reports.")
    else:
        st.write("---")
        col_m1, col_m2 = st.columns([1, 3])
        
        with col_m1:
            st.markdown("#### Actions")
            drift_threshold = st.slider("Drift Warning Threshold (Share)", min_value=0.1, max_value=0.8, value=0.3, step=0.05, help="Drift threshold score. If the fraction of drifted features exceeds this, a warning is raised.")
            
            if st.button("🚀 Run Data Drift Detection", help="Triggers Evidently AI on all buffered prediction requests."):
                with st.spinner("Analyzing telemetry data drift..."):
                    try:
                        res = httpx.post(f"{API_URL}/monitor?threshold={drift_threshold}", timeout=30.0)
                        if res.status_code == 200:
                            st.session_state["drift_results"] = res.json()
                            st.success("Analysis complete!")
                        else:
                            st.error(f"Failed to run monitoring ({res.status_code}): {res.text}")
                    except Exception as e:
                        st.error(f"Failed to contact monitoring service: {e}")
                        
        with col_m2:
            st.markdown("#### Monitoring Summary")
            if "drift_results" in st.session_state:
                res = st.session_state["drift_results"]
                
                # Check status
                is_drifted = res["drift_detected"]
                ratio = res["drift_ratio"]
                drifted_count = res["number_of_drifted_features"]
                total = res["total_features_checked"]
                
                if is_drifted:
                    st.error(f"🚨 WARNING: SIGNIFICANT DRIFT DETECTED ({ratio*100:.1f}% columns drifted)")
                else:
                    st.success(f"✅ SYSTEM HEALTHY: No significant data drift ({ratio*100:.1f}% columns drifted)")
                    
                st.metric("Drifted Columns", f"{drifted_count} / {total}", f"{ratio*100:.1f}% share")
                
                # Detail table
                df_details = pd.DataFrame.from_dict(res["column_drift"], orient='index')
                df_details.index.name = "Feature"
                df_details.reset_index(inplace=True)
                st.dataframe(df_details, use_container_width=True)
            else:
                st.info("No monitoring run completed yet. Click the button to trigger drift detection on current buffer.")
        
        # HTML Report Rendering
        st.markdown("### 📊 Interactive Evidently AI Dashboard")
        if st.checkbox("Load full interactive HTML report"):
            with st.spinner("Downloading report..."):
                try:
                    report_res = httpx.get(f"{API_URL}/monitor/report", timeout=10.0)
                    if report_res.status_code == 200:
                        components.html(report_res.text, height=800, scrolling=True)
                    else:
                        st.warning("Could not load HTML report. Make sure you run data drift detection first.")
                except Exception as e:
                    st.error(f"Failed to fetch report from FastAPI: {e}")

with tab_raw:
    # Extract the exact 50 cycle segment
    start_idx = max(0, selected_cycle - SEQUENCE_LENGTH)
    seq_data_raw = unit_data.iloc[start_idx:selected_cycle]
    
    # Scale features just for output visibility matching model inputs
    seq_data_scaled = seq_data_raw.copy()
    if local_scaler is not None:
        seq_data_scaled[seq_cols] = local_scaler.transform(seq_data_scaled[seq_cols])
        st.markdown(f"### 🔍 Scale-Normalized 50-Cycle Input Matrix (Cycle {start_idx + 1} to {selected_cycle})")
        st.write("This is the actual 3D window shape `[1, 50, 17]` fed directly to the Dual-Head LSTM model backbone for the selected cycle.")
        st.dataframe(seq_data_scaled[['time_cycles'] + seq_cols], use_container_width=True)
    else:
        st.info("Local scaler file not found. Scaling matrix view disabled.")
