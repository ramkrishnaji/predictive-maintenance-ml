import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path to import model architecture and helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from models import LSTMModel
from data_preprocessing import load_data, process_test_data

# Set Page Config
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
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

# ----------------- Load Assets (Cached) -----------------
@st.cache_resource
def get_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run training first.")
        return None
    # Model input dimension is 17 (3 operational settings + 14 remaining sensors)
    model = LSTMModel(input_dim=17, hidden_dim=100, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def get_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at {scaler_path}. Please run training first.")
        return None
    return joblib.load(scaler_path)

# Load resources
model = get_model('src/lstm_model.pth')
scaler = get_scaler('src/scaler.joblib')

# ----------------- Preprocessing Helper -----------------
def preprocess_data(raw_data_file, labels_file=None):
    """Loads and preprocesses test data and optionally merges RUL labels."""
    df_raw = load_data(raw_data_file)
    
    if labels_file is not None:
        rul_df = pd.read_csv(labels_file, sep=r'\s+', header=None)
        df_processed = process_test_data(df_raw, rul_df, cap=125)
        has_labels = True
    else:
        # If no labels, we calculate a dummy/placeholder RUL or leave it empty
        df_processed = df_raw.copy()
        df_processed['RUL'] = np.nan
        df_processed['label_bc'] = 0
        has_labels = False
        
    return df_processed, has_labels

# ----------------- Sidebar Configuration -----------------
st.sidebar.markdown("## ⚙️ Configuration & Data")

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
    # Use uploaded labels if available
    labels_source = uploaded_labels if uploaded_labels is not None else None
else:
    data_source = default_test_file if os.path.exists(default_test_file) else None
    labels_source = default_rul_file if os.path.exists(default_rul_file) else None

# Check if we have data to read
if data_source is None:
    st.warning("⚠️ Telemetry dataset not found. Please upload a file to start.")
    st.stop()

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
@st.cache_data
def get_unit_predictions(_model, _scaler, _df, unit_id):
    """Calculates model predictions for all valid cycles of a specific unit."""
    unit_df = _df[_df['unit_nr'] == unit_id].copy()
    num_cycles = len(unit_df)
    
    if num_cycles < SEQUENCE_LENGTH:
        return unit_df, None
    
    # Scale variables
    scaled_df = unit_df.copy()
    scaled_df[seq_cols] = _scaler.transform(scaled_df[seq_cols])
    
    # Generate rolling sequences starting from cycle 50
    sequences = []
    cycles = []
    
    for i in range(SEQUENCE_LENGTH, num_cycles + 1):
        seq = scaled_df[seq_cols].values[i - SEQUENCE_LENGTH : i]
        sequences.append(seq)
        cycles.append(unit_df['time_cycles'].values[i - 1])
        
    sequences = np.array(sequences, dtype=np.float32)
    
    # Predict in batch
    input_tensor = torch.from_numpy(sequences)
    with torch.no_grad():
        pred_rul, pred_cls = _model(input_tensor)
        pred_rul = pred_rul.numpy().flatten()
        pred_cls_prob = pred_cls.numpy().flatten()
        
    # Clip predictions to prevent negative RUL
    pred_rul = np.clip(pred_rul, 0, None)
    
    # Store predictions matching the cycle numbers
    pred_df = pd.DataFrame({
        'time_cycles': cycles,
        'pred_RUL': pred_rul,
        'failure_prob': pred_cls_prob
    })
    
    # Merge back with the original unit df
    merged_df = unit_df.merge(pred_df, on='time_cycles', how='left')
    return merged_df, pred_df

# Run inference on selected unit
unit_data, predictions = get_unit_predictions(model, scaler, df_processed, selected_unit)

# ----------------- UI Layout & Visuals -----------------

# Header
st.markdown('<div class="main-title">Turbofan Engine Predictive Maintenance</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Remaining Useful Life (RUL) estimation and failure risk classification</div>', unsafe_allow_html=True)

if len(unit_data) < SEQUENCE_LENGTH:
    st.error(f"❌ Selected Engine Unit {selected_unit} has only {len(unit_data)} cycles of history. A minimum of {SEQUENCE_LENGTH} cycles is required for LSTM sequence processing.")
    st.stop()

# Create tabs for structured navigation
tab_dashboard, tab_glossary, tab_evaluation, tab_raw = st.tabs([
    "📊 Live Analytics & Predictions", 
    "📖 Sensor & Parameter Glossary", 
    "📈 Global Model Performance",
    "🔍 LSTM Input Matrix Details"
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
        
        # Plot predicted RUL
        ax_rul.plot(unit_data['time_cycles'], unit_data['pred_RUL'], color='#1E88E5', label='Predicted RUL (LSTM)', lw=2.5)
        
        # Plot actual RUL if labels exist
        if has_labels:
            ax_rul.plot(unit_data['time_cycles'], unit_data['RUL'], color='#10B981', linestyle='--', label='Ground Truth RUL', lw=2)
            
        # Vertical line for the current selected cycle
        ax_rul.axvline(x=selected_cycle, color='#EF4444', linestyle=':', label='Selected Cycle Analysis', lw=2)
        
        ax_rul.set_xlabel('Operational Cycles')
        ax_rul.set_ylabel('Remaining Useful Life (RUL)')
        ax_rul.legend(facecolor='#1a1f29', edgecolor='#2d3748')
        ax_rul.set_title(f'RUL Degradation Curve for Engine Unit {selected_unit}', fontsize=12, pad=10)
        st.pyplot(fig_rul)

    with col2:
        st.markdown("### ⚠️ Failure Probability Trend")
        fig_prob, ax_prob = plt.subplots(figsize=(10, 4.5))
        
        # Plot failure probability
        ax_prob.plot(unit_data['time_cycles'], unit_data['failure_prob'], color='#F59E0B', lw=2.5, label='Failure Probability')
        ax_prob.axhline(y=0.5, color='#EF4444', linestyle='--', label='Warning Threshold (0.5)', alpha=0.7)
        
        # Vertical line for selected cycle
        ax_prob.axvline(x=selected_cycle, color='#EF4444', linestyle=':', lw=2)
        
        ax_prob.set_xlabel('Operational Cycles')
        ax_prob.set_ylabel('Probability (Failure within 30 cycles)')
        ax_prob.legend(facecolor='#1a1f29', edgecolor='#2d3748')
        ax_prob.set_title('Imminent Failure Probability Progression', fontsize=12, pad=10)
        st.pyplot(fig_prob)

    # ----------------- Telemetry Trends -----------------
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
            # Highlight values inside the current 50-cycle window
            window_start = max(SEQUENCE_LENGTH, selected_cycle - SEQUENCE_LENGTH + 1)
            ax.axvspan(window_start, selected_cycle, color='#1E88E5', alpha=0.1, label='LSTM Input Window' if i==0 else "")
            
            ax.set_title(f'Sensor {sensor}', fontsize=10)
            if i >= num_plots - cols_per_row:
                ax.set_xlabel('Cycles')
                
        # Hide unused subplots
        for j in range(num_plots, len(axes)):
            fig_sensors.delaxes(axes[j])
            
        plt.tight_layout()
        st.pyplot(fig_sensors)
    else:
        st.info("Select one or more sensors in the sidebar to plot their telemetry history.")

    # ----------------- CSV Report Download -----------------
    st.markdown("---")
    st.markdown("### 📥 Export Maintenance Report")
    st.markdown("Download a CSV record of the telemetry predictions and failure risk history for the selected engine unit to assist maintenance planning.")
    
    # Create export dataframe
    export_df = unit_data[['unit_nr', 'time_cycles', 'pred_RUL', 'failure_prob']].copy()
    if has_labels:
        export_df['actual_RUL'] = unit_data['RUL']
        export_df['prediction_error'] = np.abs(unit_data['pred_RUL'] - unit_data['RUL'])
    export_df['imminent_failure_warning'] = export_df['failure_prob'] > 0.5
    
    # Convert to CSV bytes
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label=f"💾 Download Maintenance CSV (Engine Unit {selected_unit})",
        data=csv_bytes,
        file_name=f"maintenance_report_engine_{selected_unit}.csv",
        mime="text/csv",
        key="download_maintenance_report_btn"
    )

with tab_glossary:
    st.markdown("""
    ### 📖 NASA CMAPSS Sensor Reference Guide
    This dashboard visualizes telemetry data from the official **NASA CMAPSS (Turbofan Engine Degradation Simulation)** dataset. 
    Below is a reference glossary describing the physical parameter each sensor represents, its operational unit, and its status in our predictive model.
    
    *   **Active Model Inputs (🟢)** are scaled using a min-max scaler and fed to the Dual-Head LSTM model.
    *   **Dropped Sensors (🔴)** are constant or show negligible variance in the `FD001` subset, and were filtered out during preprocessing to reduce noise.
    """)
    
    # Metadata for all 21 sensors
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
    
    df_glossary = pd.DataFrame(sensor_metadata)
    
    # Render table beautifully using st.dataframe with custom styling or st.table
    st.dataframe(
        df_glossary,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("""
    ### ⚙️ Operational Settings Reference
    In addition to the sensors, the dataset records **three operational settings** representing the environment and engine flight configurations:
    1.  **Setting 1 (Altitude)**: Altitude of the flight simulation environment.
    2.  **Setting 2 (Mach Number)**: Flight speed relative to the speed of sound.
    3.  **Setting 3 (Throttle Resolver Angle)**: Throttle control input defining power demand.
    """)

with tab_evaluation:
    st.markdown("""
    ### 📈 Global Model Performance (NASA CMAPSS FD001 Test Set)
    Below is the performance comparison between the **Dual-Head LSTM** model and the **XGBoost baseline** evaluated on the official test set.
    
    *   **Regression** evaluates Remaining Useful Life (RUL) accuracy.
    *   **Classification** evaluates imminent failure warnings ($RUL \\le 30$ cycles).
    """)
    
    # Metrics table
    metrics_data = [
        {"Model": "XGBoost Baseline", "RMSE (RUL)": "86.20", "MAE (RUL)": "N/A", "NASA Score": "N/A", "Precision": "N/A", "Recall": "N/A", "F1-Score": "N/A"},
        {"Model": "Dual-Head LSTM", "RMSE (RUL)": "13.33", "MAE (RUL)": "9.52", "NASA Score": "281.05", "Precision": "0.92", "Recall": "0.88", "F1-Score": "0.90"}
    ]
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    st.markdown("""
    *   **NASA Score**: A domain-specific asymmetric scoring metric where late predictions are penalized exponentially more than early predictions (to avoid catastrophic in-flight failures). A lower score indicates better performance.
    
    ### 🧠 Model Architecture & Optimization Details
    *   **Shared Temporal Backbone**: 2-layer LSTM with 100 hidden units per layer and 0.2 dropout regularization.
    *   **Multi-Task Loss Function**: Optimized using a custom weighted multi-task objective:
        $$Loss = Loss_{MSE} + 10 \times Loss_{BCE}$$
        The Binary Cross Entropy (BCE) loss of the classification head is scaled 10x higher than the Mean Squared Error (MSE) to maximize sensitivity (Recall) to imminent failure conditions.
    """)

with tab_raw:
    # Extract the exact 50 cycle segment
    start_idx = max(0, selected_cycle - SEQUENCE_LENGTH)
    seq_data_raw = unit_data.iloc[start_idx:selected_cycle]
    
    # Scale features just for output visibility matching model inputs
    seq_data_scaled = seq_data_raw.copy()
    seq_data_scaled[seq_cols] = scaler.transform(seq_data_scaled[seq_cols])
    
    st.markdown(f"### 🔍 Scale-Normalized 50-Cycle Input Matrix (Cycle {start_idx + 1} to {selected_cycle})")
    st.write(f"This is the actual 3D window shape `[1, 50, 17]` fed directly to the Dual-Head LSTM model backbone for the selected cycle.")
    
    st.dataframe(
        seq_data_scaled[['time_cycles'] + seq_cols],
        use_container_width=True
    )


