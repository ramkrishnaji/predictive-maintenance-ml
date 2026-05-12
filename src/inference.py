import torch
import numpy as np
import pandas as pd
import joblib
import os
from data_preprocessing import load_data, process_test_data
from models import LSTMModel

def run_sample_inference():
    # 1. Load Data
    data_path = "data/"
    test_file = os.path.join(data_path, "test_FD001.txt")
    rul_file = os.path.join(data_path, "RUL_FD001.txt")
    
    if not os.path.exists(test_file) or not os.path.exists(rul_file):
        print("Data not found.")
        return
        
    test_df = load_data(test_file)
    rul_df = pd.read_csv(rul_file, sep='\s+', header=None)
    test_df = process_test_data(test_df, rul_df)
    
    # 2. Load Assets
    scaler = joblib.load('src/scaler.joblib')
    model = LSTMModel(input_dim=24, hidden_dim=100, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load('src/lstm_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # 3. Pick a random unit from test data that has at least 50 cycles
    units = test_df['unit_nr'].unique()
    valid_units = [u for u in units if len(test_df[test_df['unit_nr']==u]) >= 50]
    
    if not valid_units:
        print("No units with enough cycles found.")
        return
        
    sample_unit = valid_units[0]
    unit_data = test_df[test_df['unit_nr'] == sample_unit]
    
    # Get the last 50 cycles
    sensor_cols = ['s_{}'.format(i) for i in range(1, 22)]
    seq_cols = sensor_cols + ['setting_1', 'setting_2', 'setting_3']
    
    last_seq = unit_data[seq_cols].values[-50:]
    last_seq_scaled = scaler.transform(last_seq)
    
    # Predict
    input_tensor = torch.from_numpy(last_seq_scaled).unsqueeze(0).float()
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_rul = prediction.item()
    
    actual_rul = unit_data['RUL'].values[-1]
    
    print(f"--- Sample Inference (Unit {sample_unit}) ---")
    print(f"Predicted RUL: {predicted_rul:.2f}")
    print(f"Actual RUL:    {actual_rul:.2f}")
    print(f"Error:         {abs(predicted_rul - actual_rul):.2f}")

if __name__ == "__main__":
    run_sample_inference()
