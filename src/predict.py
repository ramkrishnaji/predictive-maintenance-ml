import os
import sys
import joblib
import torch
import numpy as np
import mlflow
import mlflow.pytorch

# Add current directory to path to ensure models import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import LSTMModel

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "LSTMModel"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(LOCAL_DIR, "lstm_model.pth")
LOCAL_SCALER_PATH = os.path.join(LOCAL_DIR, "scaler.joblib")

# Globals for caching
_scaler = None
_model = None

def load_scaler():
    """Loads the scaler from the local path."""
    if os.path.exists(LOCAL_SCALER_PATH):
        return joblib.load(LOCAL_SCALER_PATH)
    raise FileNotFoundError(f"Scaler not found at {LOCAL_SCALER_PATH}")

def load_model():
    """Loads the PyTorch LSTM model from MLflow Registry or local fallback."""
    try:
        # Configure tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"Attempting to load model from MLflow Registry: models:/{MODEL_NAME}/Production at {MLFLOW_TRACKING_URI}")
        
        # Load directly using MLflow
        model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/Production")
        model.eval()
        print("Successfully loaded model from MLflow Model Registry.")
        return model
    except Exception as e:
        print(f"Could not load model from MLflow Registry ({e}). Falling back to local file: {LOCAL_MODEL_PATH}")
        if os.path.exists(LOCAL_MODEL_PATH):
            model = LSTMModel(input_dim=17, hidden_dim=100, num_layers=2, output_dim=1)
            model.load_state_dict(torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"Local model file not found at {LOCAL_MODEL_PATH}")

def get_scaler_and_model():
    """Fetches cached scaler and model or loads them if not cached."""
    global _scaler, _model
    if _scaler is None:
        _scaler = load_scaler()
    if _model is None:
        _model = load_model()
    return _scaler, _model

def predict_sequence(seq):
    """
    Runs inference for a single 50x17 sequence.
    Args:
        seq: list or np.ndarray of shape (50, 17)
    Returns:
        dict containing predicted RUL, failure probability, and failure warning flag.
    """
    scaler, model = get_scaler_and_model()
    seq_arr = np.array(seq).astype(np.float32)
    if seq_arr.shape != (50, 17):
        raise ValueError(f"Expected sequence shape (50, 17), got {seq_arr.shape}")
    
    # Scale sequence (scaler transforms 2D telemetry matrix directly)
    seq_scaled = scaler.transform(seq_arr)
    
    # Run prediction
    input_tensor = torch.from_numpy(seq_scaled).unsqueeze(0)  # Shape: (1, 50, 17)
    with torch.no_grad():
        rul_pred, cls_pred = model(input_tensor)
        rul = float(max(0, rul_pred.item()))
        fail_prob = float(cls_pred.item())
        
    return {
        "predicted_rul": round(rul, 2),
        "failure_probability_30c": round(fail_prob, 4),
        "imminent_failure": fail_prob > 0.5
    }
