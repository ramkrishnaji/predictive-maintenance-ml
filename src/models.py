import torch
import torch.nn as nn
import xgboost as xgb

class LSTMModel(nn.Module):
    """Dual-Head LSTM for RUL regression and Failure Classification."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Shared LSTM Backbone
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Regression Head
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )
        
        # Classification Head (Binary)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        last_step_out = out[:, -1, :]
        
        rul_out = self.reg_head(last_step_out)
        cls_out = self.cls_head(last_step_out)
        
        return rul_out, cls_out

class XGBoostBaseline:
    """XGBoost model wrapper for baseline RUL prediction."""
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        else:
            self.params = params
        self.model = None

    def train(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, num_boost_round=100)

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)
