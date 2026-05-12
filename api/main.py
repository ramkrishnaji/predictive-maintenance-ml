from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import joblib
import os
import sys

# Add src to path to import model architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from models import LSTMModel

app = FastAPI(title="Predictive Maintenance API", description="API for predicting Remaining Useful Life (RUL) of turbofan engines.")

# Load Model and Scaler
MODEL_PATH = "src/lstm_model.pth"
SCALER_PATH = "src/scaler.joblib"

# Global variables for model and scaler
model = None
scaler = None

class SensorData(BaseModel):
    # Expecting a list of 50 cycles, each cycle having 17 features (after dropping low-variance sensors)
    data: list

class BatchSensorData(BaseModel):
    # List of sequences
    sequences: list

@app.on_event("startup")
def load_assets():
    global model, scaler
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    
    if os.path.exists(MODEL_PATH):
        # input_dim is 17 (3 settings + 14 remaining sensors)
        model = LSTMModel(input_dim=17, hidden_dim=100, num_layers=2, output_dim=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()

@app.get("/health")
def health_check():
    status = "healthy" if model is not None and scaler is not None else "degraded"
    return {
        "status": status,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/")
def read_root():
    return {"message": "Predictive Maintenance API v2. Use /predict or /predict/batch."}

def process_single_sequence(seq_data):
    input_data = np.array(seq_data).astype(np.float32)
    if input_data.shape != (50, 17):
        raise ValueError(f"Expected shape (50, 17), got {input_data.shape}")
    
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.from_numpy(input_scaled).unsqueeze(0)
    
    with torch.no_grad():
        rul_pred, cls_pred = model(input_tensor)
        rul = float(max(0, rul_pred.item()))
        fail_prob = float(cls_pred.item())
        
    return {
        "predicted_rul": round(rul, 2),
        "failure_probability_30c": round(fail_prob, 4),
        "imminent_failure": fail_prob > 0.5
    }

@app.post("/predict")
async def predict(payload: SensorData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model assets not loaded.")
    
    try:
        result = process_single_sequence(payload.data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(payload: BatchSensorData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model assets not loaded.")
    
    results = []
    for i, seq in enumerate(payload.sequences):
        try:
            res = process_single_sequence(seq)
            res["index"] = i
            results.append(res)
        except Exception as e:
            results.append({"index": i, "error": str(e)})
            
    return {"batch_results": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
