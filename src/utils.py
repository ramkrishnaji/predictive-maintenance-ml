import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report

def nasa_score(y_true, y_pred):
    """NASA scoring function for RUL prediction."""
    d = y_pred - y_true
    score = 0
    for val in d:
        if val < 0:
            score += np.exp(-val/13) - 1
        else:
            score += np.exp(val/10) - 1
    return score

def plot_training_history(history_df, save_path='assets/training_history.png'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_df['train_loss'], label='Train Loss')
    plt.title('Training Loss (Weighted)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df['val_rmse'], label='Val RMSE', color='orange')
    plt.title('Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pred_vs_actual(y_true, y_pred, save_path='assets/pred_vs_actual.png'):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='teal')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title('Predicted vs Actual RUL (Capped at 125)')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path='assets/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Fail Soon'], yticklabels=['Healthy', 'Fail Soon'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Failure Classification (RUL <= 30)')
    plt.savefig(save_path)
    plt.close()
