import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support
import joblib
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from data_preprocessing import load_data, calculate_rul, process_test_data, gen_sequence, gen_labels, engine_train_test_split
from models import LSTMModel, XGBoostBaseline
from utils import nasa_score

# Constants
SEQUENCE_LENGTH = 50
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def train_model(model, train_loader, val_loader, device):
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_rmse = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_rmse': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y_reg, batch_y_cls in train_loader:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            batch_y_cls = batch_y_cls.to(device)
            
            optimizer.zero_grad()
            pred_reg, pred_cls = model(batch_x)
            
            loss_reg = criterion_reg(pred_reg, batch_y_reg)
            loss_cls = criterion_cls(pred_cls, batch_y_cls)
            
            # Weighted loss: prioritize RUL prediction
            loss = loss_reg + 10 * loss_cls 
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_rmse = 0
        with torch.no_grad():
            for val_x, val_y_reg, _ in val_loader:
                val_x, val_y_reg = val_x.to(device), val_y_reg.to(device)
                pred_reg, _ = model(val_x)
                val_rmse += criterion_reg(pred_reg, val_y_reg).item()
        
        val_rmse = np.sqrt(val_rmse / len(val_loader))
        train_loss /= len(train_loader)
        
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_rmse)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}')
        
        # Log epoch level metrics to MLflow
        if mlflow.active_run():
            mlflow.log_metric("epoch_train_loss", train_loss, step=epoch)
            mlflow.log_metric("epoch_val_rmse", val_rmse, step=epoch)
        
        # Early Stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'src/lstm_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    return history

def register_and_promote_model(model_uri, new_rmse):
    """Registers the model and promotes it to Production if its RMSE is better than current production."""
    client = MlflowClient()
    model_name = "LSTMModel"
    
    print(f"Registering model version from: {model_uri}")
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = mv.version
    except Exception as e:
        print(f"Could not register model: {e}")
        return
    
    # Check if there is an existing Production model
    production_version = None
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if latest_versions:
            production_version = latest_versions[0]
    except Exception as e:
        print(f"No existing production model or model name '{model_name}' not found: {e}")
        
    promote = False
    if production_version is None:
        print("No current Production model found. Promoting this model to Production.")
        promote = True
    else:
        prod_run_id = production_version.run_id
        try:
            prod_run = client.get_run(prod_run_id)
            prod_rmse = prod_run.data.metrics.get("test_rmse", float('inf'))
            print(f"Current Production model test RMSE: {prod_rmse:.4f}")
        except Exception as e:
            print(f"Could not retrieve metrics of current production run: {e}")
            prod_rmse = float('inf')
            
        if new_rmse < prod_rmse:
            print(f"New model RMSE ({new_rmse:.4f}) is better than Production model RMSE ({prod_rmse:.4f}). Promoting.")
            promote = True
        else:
            print(f"New model RMSE ({new_rmse:.4f}) is NOT better than Production model RMSE ({prod_rmse:.4f}). Stage set to Staging.")
            
    if promote:
        if production_version is not None:
            print(f"Demoting old production version {production_version.version} to Archived.")
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=production_version.version,
                    stage="Archived"
                )
            except Exception as e:
                print(f"Error demoting old production version: {e}")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        try:
            client.set_registered_model_alias(model_name, "Production", version)
            client.set_model_version_tag(model_name, version, "stage", "production")
        except Exception as e:
            print(f"Failed to set alias/tags: {e}")
        print(f"Model version {version} promoted to Production.")
    else:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        try:
            client.set_registered_model_alias(model_name, "Staging", version)
            client.set_model_version_tag(model_name, version, "stage", "staging")
        except Exception as e:
            print(f"Failed to set alias/tags: {e}")
        print(f"Model version {version} set to Staging.")

def main():
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Predictive_Maintenance_Turbofan")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    data_path = "data/"
    raw_df = load_data(os.path.join(data_path, "train_FD001.txt"))
    test_df_raw = load_data(os.path.join(data_path, "test_FD001.txt"))
    rul_df_raw = pd.read_csv(os.path.join(data_path, "RUL_FD001.txt"), sep=r'\s+', header=None)
    
    # Save training dataset as baseline reference for drift detection
    os.makedirs(data_path, exist_ok=True)
    reference_data_path = os.path.join(data_path, "reference_data.csv")
    raw_df.to_csv(reference_data_path, index=False)
    print(f"Saved reference dataset to {reference_data_path}")
    
    # 2. Preprocess & Split
    df_processed = calculate_rul(raw_df, cap=125)
    train_df, val_df, test_df_internal = engine_train_test_split(df_processed)
    
    # 3. Normalize
    sensor_cols = [c for c in train_df.columns if c.startswith('s_')]
    setting_cols = [c for c in train_df.columns if c.startswith('setting_')]
    seq_cols = sensor_cols + setting_cols
    
    scaler = MinMaxScaler()
    train_df[seq_cols] = scaler.fit_transform(train_df[seq_cols])
    val_df[seq_cols] = scaler.transform(val_df[seq_cols])
    test_df_internal[seq_cols] = scaler.transform(test_df_internal[seq_cols])
    
    joblib.dump(scaler, 'src/scaler.joblib')
    
    # 4. Generate Sequences
    def prepare_sequences(df):
        x, y_reg, y_cls = [], [], []
        for id in df['unit_nr'].unique():
            unit_data = df[df['unit_nr'] == id]
            if len(unit_data) >= SEQUENCE_LENGTH:
                x.extend(list(gen_sequence(unit_data, SEQUENCE_LENGTH, seq_cols)))
                y_reg.extend(gen_labels(unit_data, SEQUENCE_LENGTH, ['RUL']))
                y_cls.extend(gen_labels(unit_data, SEQUENCE_LENGTH, ['label_bc']))
        return np.array(x, dtype=np.float32), np.array(y_reg, dtype=np.float32), np.array(y_cls, dtype=np.float32)

    x_train, y_train_reg, y_train_cls = prepare_sequences(train_df)
    x_val, y_val_reg, y_val_cls = prepare_sequences(val_df)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_reg), torch.from_numpy(y_train_cls)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val_reg), torch.from_numpy(y_val_cls)), batch_size=BATCH_SIZE)
    
    # 5. Train LSTM with MLflow logging
    print("Training PyTorch LSTM...")
    with mlflow.start_run(run_name="LSTM_Model_Training") as run:
        # Log hyperparameters
        mlflow.log_params({
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "patience": PATIENCE,
            "optimizer": "Adam",
            "model_type": "LSTM",
            "device": str(device)
        })
        
        model = LSTMModel(input_dim=len(seq_cols), hidden_dim=100, num_layers=2).to(device)
        history = train_model(model, train_loader, val_loader, device)
        
        # 6. Final Evaluation on actual Test Set
        print("Final Evaluation on official test set...")
        model.load_state_dict(torch.load('src/lstm_model.pth'))
        model.eval()
        
        test_df = process_test_data(test_df_raw, rul_df_raw, cap=125)
        test_df[seq_cols] = scaler.transform(test_df[seq_cols])
        
        x_test_last = []
        y_test_reg = []
        y_test_cls = []
        for id in test_df['unit_nr'].unique():
            unit_data = test_df[test_df['unit_nr'] == id]
            if len(unit_data) >= SEQUENCE_LENGTH:
                x_test_last.append(unit_data[seq_cols].values[-SEQUENCE_LENGTH:])
                y_test_reg.append(unit_data['RUL'].values[-1])
                y_test_cls.append(unit_data['label_bc'].values[-1])
                
        x_test_last = np.array(x_test_last, dtype=np.float32)
        y_test_reg = np.array(y_test_reg, dtype=np.float32)
        y_test_cls = np.array(y_test_cls, dtype=np.float32)
        
        with torch.no_grad():
            pred_reg, pred_cls = model(torch.from_numpy(x_test_last).to(device))
            pred_reg = pred_reg.cpu().numpy().flatten()
            pred_cls_prob = pred_cls.cpu().numpy().flatten()
            pred_cls = (pred_cls_prob > 0.5).astype(int)
            
        rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg))
        mae = mean_absolute_error(y_test_reg, pred_reg)
        score = nasa_score(y_test_reg, pred_reg)
        
        # Log classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_cls, pred_cls, average='binary')
        
        # Log final metrics
        mlflow.log_metrics({
            "test_rmse": rmse,
            "test_mae": mae,
            "test_nasa_score": score,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        })
        
        print(f"LSTM Official Test RMSE: {rmse:.2f}")
        print(f"LSTM Official Test MAE: {mae:.2f}")
        print(f"LSTM NASA Score: {score:.2f}")
        
        # Save metrics for plotting/logging (simplified)
        os.makedirs('assets', exist_ok=True)
        pd.DataFrame(history).to_csv('assets/training_history.csv', index=False)
        
        # Log Artifacts
        mlflow.log_artifact('src/scaler.joblib', artifact_path="scaler")
        mlflow.log_artifact('assets/training_history.csv', artifact_path="metrics")
        
        # Log and Register LSTM model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="LSTMModel"
        )
        
        # Call the promotion logic
        model_uri = f"runs:/{run.info.run_id}/model"
        register_and_promote_model(model_uri, rmse)
        
    # 7. Baseline (XGBoost)
    print("Training XGBoost Baseline...")
    with mlflow.start_run(run_name="XGBoost_Baseline_Model"):
        x_train_xgb = train_df.groupby('unit_nr').last()[seq_cols]
        y_train_xgb = train_df.groupby('unit_nr').last()['RUL']
        
        x_test_xgb = test_df.groupby('unit_nr').last()[seq_cols]
        y_test_xgb = rul_df_raw['RUL_end'].values.flatten()
        
        xgb_model = XGBoostBaseline()
        
        mlflow.log_params({
            "model_type": "XGBoost",
            "objective": xgb_model.params['objective'],
            "max_depth": xgb_model.params['max_depth'],
            "eta": xgb_model.params['eta'],
            "subsample": xgb_model.params['subsample'],
            "colsample_bytree": xgb_model.params['colsample_bytree']
        })
        
        xgb_model.train(x_train_xgb, y_train_xgb)
        xgb_preds = xgb_model.predict(x_test_xgb)
        
        xgb_rmse = np.sqrt(mean_squared_error(y_test_xgb, xgb_preds))
        xgb_mae = mean_absolute_error(y_test_xgb, xgb_preds)
        xgb_score = nasa_score(y_test_xgb, xgb_preds)
        
        mlflow.log_metrics({
            "test_rmse": xgb_rmse,
            "test_mae": xgb_mae,
            "test_nasa_score": xgb_score
        })
        
        print(f"XGBoost Test RMSE: {xgb_rmse:.2f}")
        joblib.dump(xgb_model.model, 'src/xgb_model.joblib')
        mlflow.log_artifact('src/xgb_model.joblib', artifact_path="model")

if __name__ == "__main__":
    main()
