import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, calculate_rul

def save_visuals():
    sns.set_style('whitegrid')
    data_path = "data/"
    train_df = load_data(os.path.join(data_path, "train_FD001.txt"))
    train_df = calculate_rul(train_df)
    
    # 1. Degradation Curve
    unit_id = 1
    unit_data = train_df[train_df['unit_nr'] == unit_id]
    
    plt.figure(figsize=(12, 6))
    plt.plot(unit_data['time_cycles'], unit_data['s_7'], label='Sensor 7 (Pressure)')
    plt.plot(unit_data['time_cycles'], unit_data['s_11'], label='Sensor 11 (Temperature)')
    plt.title(f'Sensor Degradation Over Time (Engine Unit {unit_id})', fontsize=15)
    plt.xlabel('Operational Cycles', fontsize=12)
    plt.ylabel('Normalized Sensor Reading', fontsize=12)
    plt.legend()
    plt.savefig('notebooks/degradation_curve.png')
    print("Saved degradation_curve.png")
    
    # 2. RUL vs Time
    plt.figure(figsize=(10, 5))
    plt.plot(unit_data['time_cycles'], unit_data['RUL'], color='red', linestyle='--')
    plt.title('Ground Truth RUL Decline', fontsize=15)
    plt.xlabel('Cycles', fontsize=12)
    plt.ylabel('Remaining Useful Life', fontsize=12)
    plt.savefig('notebooks/rul_decline.png')
    print("Saved rul_decline.png")

if __name__ == "__main__":
    save_visuals()
