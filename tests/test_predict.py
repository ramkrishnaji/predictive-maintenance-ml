import os
import sys
import numpy as np
import pytest

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import predict_sequence

def test_predict_sequence_shape_error():
    """Verifies that invalid input shapes raise a ValueError."""
    # Invalid length (49 instead of 50)
    bad_length_seq = np.random.rand(49, 17).tolist()
    with pytest.raises(ValueError, match="Expected sequence shape"):
        predict_sequence(bad_length_seq)
        
    # Invalid features (16 instead of 17)
    bad_feat_seq = np.random.rand(50, 16).tolist()
    with pytest.raises(ValueError, match="Expected sequence shape"):
        predict_sequence(bad_feat_seq)

def test_predict_sequence_success():
    """Verifies that a valid 50x17 sequence returns the correct response schema."""
    # Create valid dummy sequence
    valid_seq = np.random.rand(50, 17).tolist()
    
    # Run prediction (will fall back to local model since MLflow is offline during tests)
    result = predict_sequence(valid_seq)
    
    assert isinstance(result, dict)
    assert "predicted_rul" in result
    assert "failure_probability_30c" in result
    assert "imminent_failure" in result
    
    assert isinstance(result["predicted_rul"], (int, float))
    assert result["predicted_rul"] >= 0
    assert 0.0 <= result["failure_probability_30c"] <= 1.0
    assert isinstance(result["imminent_failure"], bool)
