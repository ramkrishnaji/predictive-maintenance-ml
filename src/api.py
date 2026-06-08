import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import predict_sequence, get_scaler_and_model
from monitor import DriftMonitor

app = FastAPI(
    title="Predictive Maintenance Production API", 
    description="FastAPI service for jet engine RUL prediction and data drift monitoring."
)

BUFFER_PATH = "data/inference_buffer.csv"

class SensorData(BaseModel):
    # Expecting a sequence of shape (50, 17)
    data: list

class BatchSensorData(BaseModel):
    # List of sequences of shape (50, 17)
    sequences: list

def append_to_buffer(sequences_list):
    """Appends incoming inference data to a buffer for drift tracking."""
    # NASA CMAPSS feature names (matching reference data)
    sensor_cols = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    setting_cols = ['setting_1', 'setting_2', 'setting_3']
    features = sensor_cols + setting_cols
    
    records = []
    for seq in sequences_list:
        for cycle_data in seq:
            records.append(cycle_data)
            
    df = pd.DataFrame(records, columns=features)
    
    os.makedirs(os.path.dirname(BUFFER_PATH), exist_ok=True)
    if not os.path.exists(BUFFER_PATH):
        df.to_csv(BUFFER_PATH, index=False)
    else:
        df.to_csv(BUFFER_PATH, mode='a', header=False, index=False)

@app.on_event("startup")
def load_assets():
    """Validates that ML model and scaler are loadable on startup."""
    try:
        scaler, model = get_scaler_and_model()
        print("FastAPI assets loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load model/scaler assets on startup ({e}). Will retry on request.")

@app.get("/")
def read_root():
    return {
        "message": "Predictive Maintenance MLOps API. Use /predict, /predict/batch, or /monitor.",
        "inference_buffer_size": len(pd.read_csv(BUFFER_PATH)) if os.path.exists(BUFFER_PATH) else 0
    }

@app.get("/health")
def health_check():
    try:
        scaler, model = get_scaler_and_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.post("/predict")
async def predict(payload: SensorData):
    """
    Predicts Remaining Useful Life (RUL) for a single engine sequence of shape (50, 17).
    """
    try:
        result = predict_sequence(payload.data)
        # Buffer the telemetry data for drift monitoring
        append_to_buffer([payload.data])
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(payload: BatchSensorData):
    """
    Predicts RUL for a batch of engine sequences.
    """
    results = []
    valid_sequences = []
    
    for i, seq in enumerate(payload.sequences):
        try:
            res = predict_sequence(seq)
            res["index"] = i
            results.append(res)
            valid_sequences.append(seq)
        except Exception as e:
            results.append({"index": i, "error": str(e)})
            
    if valid_sequences:
        # Buffer only valid sequences for drift detection
        append_to_buffer(valid_sequences)
        
    return {"batch_results": results, "count": len(results)}

@app.post("/monitor")
async def run_monitoring(threshold: float = 0.3):
    """
    Triggers Evidently AI drift detection on all buffered inference data since last run.
    Clears the buffer upon success.
    """
    if not os.path.exists(BUFFER_PATH):
        raise HTTPException(
            status_code=400, 
            detail="Inference buffer is empty. Submit some prediction requests first to build a batch."
        )
        
    try:
        current_df = pd.read_csv(BUFFER_PATH)
        
        # Verify if we have enough records to do a meaningful drift evaluation (at least 1 record)
        if len(current_df) < 1:
            raise HTTPException(status_code=400, detail="Inference buffer has no records.")
            
        monitor = DriftMonitor()
        drift_results = monitor.detect_drift(current_df, threshold=threshold)
        
        # Clear buffer
        os.remove(BUFFER_PATH)
        
        return drift_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitor/upload")
async def upload_for_monitoring(file: UploadFile = File(...), threshold: float = 0.3):
    """
    Manually upload a CSV batch of telemetry data to perform drift analysis.
    Does not clear the prediction buffer.
    """
    try:
        df = pd.read_csv(file.file)
        monitor = DriftMonitor()
        drift_results = monitor.detect_drift(df, threshold=threshold)
        return drift_results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded file: {str(e)}")

@app.get("/monitor/report")
async def get_drift_report():
    """
    Serves the HTML report generated by the latest drift analysis.
    """
    report_path = "assets/drift_report.html"
    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=404, 
            detail="No drift report HTML found. Please trigger /monitor or /monitor/upload first."
        )
    return FileResponse(report_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
