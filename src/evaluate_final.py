import torch
import numpy as np
import pandas as pd
import joblib
import os
from data_preprocessing import load_data, process_test_data
from models import LSTMModel
from utils import nasa_score, plot_training_history, plot_pred_vs_actual, plot_confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    data_path = "data/"
    test_df_raw = load_data(os.path.join(data_path, "test_FD001.txt"))
    rul_df_raw = pd.read_csv(os.path.join(data_path, "RUL_FD001.txt"), sep=r'\s+', header=None)
    
    # 2. Load Assets
    scaler = joblib.load('src/scaler.joblib')
    # input_dim is 17
    model = LSTMModel(input_dim=17, hidden_dim=100, num_layers=2)
    model.load_state_dict(torch.load('src/lstm_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Process Test Data
    test_df = process_test_data(test_df_raw, rul_df_raw, cap=125)
    
    sensor_cols = [c for c in test_df.columns if c.startswith('s_')]
    setting_cols = [c for c in test_df.columns if c.startswith('setting_')]
    seq_cols = sensor_cols + setting_cols
    
    test_df[seq_cols] = scaler.transform(test_df[seq_cols])
    
    # 4. Generate Test Sequences (Last cycle only for official RUL evaluation)
    x_test_last = []
    y_test_reg = []
    y_test_cls = []
    for id in test_df['unit_nr'].unique():
        unit_data = test_df[test_df['unit_nr'] == id]
        if len(unit_data) >= 50: # SEQUENCE_LENGTH
            x_test_last.append(unit_data[seq_cols].values[-50:])
            y_test_reg.append(unit_data['RUL'].values[-1])
            y_test_cls.append(unit_data['label_bc'].values[-1])
            
    x_test_last = np.array(x_test_last, dtype=np.float32)
    y_test_reg = np.array(y_test_reg, dtype=np.float32)
    y_test_cls = np.array(y_test_cls, dtype=np.float32)
    
    # 5. Predict
    with torch.no_grad():
        pred_reg, pred_cls_prob = model(torch.from_numpy(x_test_last).to(device))
        pred_reg = pred_reg.cpu().numpy().flatten()
        pred_cls_prob = pred_cls_prob.cpu().numpy().flatten()
        pred_cls = (pred_cls_prob > 0.5).astype(int)
        
    # 6. Calculate Metrics
    rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg))
    mae = mean_absolute_error(y_test_reg, pred_reg)
    score = nasa_score(y_test_reg, pred_reg)
    
    print(f"--- Final Evaluation Results ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"NASA Score: {score:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test_cls, pred_cls))
    
    # 7. Generate Plots
    plot_pred_vs_actual(y_test_reg, pred_reg)
    plot_confusion_matrix(y_test_cls, pred_cls)
    
    if os.path.exists('assets/training_history.csv'):
        history_df = pd.read_csv('assets/training_history.csv')
        plot_training_history(history_df)
        print("Generated assets: training_history.png, pred_vs_actual.png, confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
