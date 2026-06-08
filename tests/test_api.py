import os
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from api import app, BUFFER_PATH

client = TestClient(app)

def test_read_root():
    """Tests the API root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "inference_buffer_size" in data

def test_health_check():
    """Tests the FastAPI health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]

def test_predict_endpoint_success():
    """Tests prediction with valid 50x17 sequence."""
    seq = np.random.rand(50, 17).tolist()
    response = client.post("/predict", json={"data": seq})
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rul" in data
    assert "failure_probability_30c" in data
    assert "imminent_failure" in data
    assert isinstance(data["predicted_rul"], (int, float))
    assert isinstance(data["imminent_failure"], bool)

def test_predict_endpoint_invalid_shape():
    """Tests prediction with invalid sequence shape."""
    seq = np.random.rand(40, 17).tolist()
    response = client.post("/predict", json={"data": seq})
    assert response.status_code == 400
    assert "detail" in response.json()

def test_predict_batch_endpoint():
    """Tests batch prediction endpoint."""
    seq1 = np.random.rand(50, 17).tolist()
    seq2 = np.random.rand(50, 17).tolist()
    
    response = client.post("/predict/batch", json={"sequences": [seq1, seq2]})
    assert response.status_code == 200
    data = response.json()
    assert "batch_results" in data
    assert data["count"] == 2
    assert len(data["batch_results"]) == 2
    assert data["batch_results"][0]["index"] == 0
    assert "predicted_rul" in data["batch_results"][0]

def test_monitor_empty_buffer():
    """Tests monitor endpoint behavior when buffer file does not exist."""
    # Ensure buffer file doesn't exist
    if os.path.exists(BUFFER_PATH):
        try:
            os.remove(BUFFER_PATH)
        except OSError:
            pass
            
    response = client.post("/monitor")
    assert response.status_code == 400
    assert "Inference buffer is empty" in response.json()["detail"]
